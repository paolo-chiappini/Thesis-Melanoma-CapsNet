import os
import random
from collections import Counter

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision.models import resnet18
from tqdm import tqdm

from datasets.augmentations import get_transforms
from datasets.loaders import get_dataset


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (https://arxiv.org/abs/2004.11362)
    """

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: shape (batch_size, feature_dim)
        labels: shape (batch_size)
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        # Remove diagonal (self-comparison)
        logits_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix[logits_mask].view(batch_size, -1)

        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(device)
        label_mask = label_mask[logits_mask].view(batch_size, -1)

        # Compute log_prob
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(
            exp_sim.sum(dim=1, keepdim=True) + 1e-9
        )

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (label_mask * log_prob).sum(dim=1) / (
            label_mask.sum(dim=1) + 1e-9
        )

        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss


# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        return (self.alpha * (1 - pt) ** self.gamma * BCE_loss).mean()


class LDAMMarginLoss(nn.Module):
    def __init__(self, cls_counts, max_m=0.5, base_loss=None):
        super().__init__()
        self.cls_counts = cls_counts
        self.base_loss = base_loss or nn.BCEWithLogitsLoss()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_counts))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)

    def forward(self, logits, targets):
        # Apply class-dependent margin
        margins = torch.zeros_like(logits)
        for i in range(len(logits)):
            cls = int(targets[i].item())
            margins[i] = self.m_list[cls]

        # Adjust logits with margin
        adjusted_logits = logits.clone()
        adjusted_logits[targets == 1] -= margins[targets == 1]
        adjusted_logits[targets == 0] += margins[targets == 0]

        return self.base_loss(adjusted_logits, targets)


class LogitMarginLoss(nn.Module):
    def __init__(self, margin=1.0, base_loss=None):
        super().__init__()
        self.margin = margin
        self.base_loss = base_loss or nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        base = self.base_loss(logits, targets)

        # Push positive logits > +margin
        pos_mask = targets == 1
        neg_mask = targets == 0

        pos_penalty = (
            F.relu(self.margin - logits[pos_mask]).mean() if pos_mask.any() else 0
        )
        neg_penalty = (
            F.relu(self.margin + logits[neg_mask]).mean() if neg_mask.any() else 0
        )

        margin_loss = pos_penalty + neg_penalty
        return base + margin_loss


# === Custom Dataset Wrapper ===
class BinaryImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset  # Assume it's something like ImageFolder or custom

    def __getitem__(self, index):
        dict = self.dataset[index]
        image = dict["images"]
        label = dict["malignancy_targets"]
        return image, label

    def __len__(self):
        return len(self.dataset)


# === Utility to Split Balanced Subset ===
def get_balanced_subset(dataset, positive_label=1):
    print("🔍 Indexing dataset to create balanced subset...")

    pos_indices = []
    neg_indices = []

    for i, (img, label) in tqdm(
        enumerate(dataset), total=len(dataset), desc="Indexing"
    ):
        if label == positive_label:
            pos_indices.append(i)
        else:
            neg_indices.append(i)

    if len(pos_indices) == 0:
        raise ValueError("No positive samples (label == 1) found in dataset.")
    if len(neg_indices) == 0:
        raise ValueError("No negative samples (label != 1) found in dataset.")

    num_pos = len(pos_indices)
    neg_indices = random.sample(neg_indices, min(num_pos, len(neg_indices)))

    balanced_indices = pos_indices + neg_indices
    random.shuffle(balanced_indices)

    print(
        f"✅ Balanced subset: {len(pos_indices)} positive, {len(neg_indices)} negative, total {len(balanced_indices)} samples"
    )
    return Subset(dataset, balanced_indices)


class ResNetWithFeatures(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base_model = resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            *list(base_model.children())[:-1]
        )  # all except final FC
        self.fc = nn.Linear(base_model.fc.in_features, 1)  # Binary classification

    def forward(self, x):
        x = self.backbone(x)  # Output shape: [B, 512, 1, 1]
        features = torch.flatten(x, 1)  # [B, 512]
        logits = self.fc(features)  # [B, 1]
        return features, logits


# === Model Definition ===
def get_model():
    # model = resnet18(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
    model = ResNetWithFeatures(pretrained=True)
    return model


# === Training Function ===
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        features, logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


# === Validation Function ===
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _, logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.unsqueeze(1).cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    auc = roc_auc_score(labels.numpy(), probs.numpy())
    report = classification_report(labels.numpy(), preds.numpy(), zero_division=0)

    return auc, report


def save_model(model, path, name="model.pth"):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, name)
    torch.save(model.state_dict(), full_path)
    print(f"💾 Model saved to {full_path}")


class ClassAwareSampler(Sampler):
    """
    Custom sampler that oversamples the minority class to ensure class balance.
    Ensures that each batch contains a desired ratio of minority class.
    """

    def __init__(self, labels, batch_size=32, minority_class=1, minority_fraction=0.25):
        """
        labels: list or array of dataset labels
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.minority_class = minority_class
        self.minority_fraction = minority_fraction

        # Get indices

        self.minority_indices = np.where(self.labels == minority_class)[0]
        self.majority_indices = np.where(self.labels != minority_class)[0]

        assert len(self.minority_indices) > 0, "No minority samples found."

    def __iter__(self):
        num_batches = len(self.labels) // self.batch_size
        indices = []

        for _ in range(num_batches):
            n_minority = int(self.batch_size * self.minority_fraction)
            n_majority = self.batch_size - n_minority

            minority_sample = np.random.choice(
                self.minority_indices, n_minority, replace=True
            )
            majority_sample = np.random.choice(
                self.majority_indices, n_majority, replace=False
            )

            batch = np.concatenate([minority_sample, majority_sample])
            np.random.shuffle(batch)
            indices.extend(batch.tolist())

        return iter(indices)

    def __len__(self):
        return len(self.labels)


class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        features, logits = self.model(x)
        return features, logits / self.temperature.to(logits.device)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        embeddings: tensor of shape (batch_size, embedding_dim)
        labels: tensor of shape (batch_size,)
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Compute pairwise distance matrix (Euclidean)
        # dist[i,j] = ||embeddings[i] - embeddings[j]||_2
        dist = torch.cdist(embeddings, embeddings, p=2)

        loss = 0.0
        triplets = 0

        for i in range(batch_size):
            anchor_label = labels[i]

            # Positive examples: same label, exclude anchor itself
            pos_mask = (labels == anchor_label) & (
                torch.arange(batch_size, device=device) != i
            )
            pos_indices = torch.where(pos_mask)[0]

            # Negative examples: different label
            neg_mask = labels != anchor_label
            neg_indices = torch.where(neg_mask)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue  # skip if no valid pos/neg

            anchor_dist = dist[i]

            for pos_idx in pos_indices:
                pos_dist = anchor_dist[pos_idx]

                # Semi-hard negative mining: negative with dist > pos_dist but within margin
                neg_dists = anchor_dist[neg_indices]
                semi_hard_neg_mask = (neg_dists > pos_dist) & (
                    neg_dists < pos_dist + self.margin
                )
                semi_hard_negatives = neg_indices[semi_hard_neg_mask]

                if len(semi_hard_negatives) == 0:
                    # If no semi-hard negatives, use hardest negative (smallest neg_dist)
                    hard_neg_dist, hard_neg_idx = torch.min(neg_dists, dim=0)
                    neg_dist = hard_neg_dist
                else:
                    # Randomly pick one semi-hard negative
                    neg_idx = semi_hard_negatives[
                        torch.randint(len(semi_hard_negatives), (1,))
                    ]
                    neg_dist = anchor_dist[neg_idx]

                triplet_loss = F.relu(pos_dist - neg_dist + self.margin).item()

                if triplet_loss > 0:
                    loss += triplet_loss
                    triplets += 1

        if triplets == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)
        else:
            return loss / triplets


def apply_soft_labels(labels, epsilon=0.1):
    return labels * (1 - epsilon) + 0.5 * epsilon


# === Main Curriculum Learning Training ===
def main():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_epochs_pretrain = 10
    num_epochs_finetune = 10

    # --- Transforms ---
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # ImageNet stats
    #         ),
    #     ]
    # )

    fake_config = {
        "dataset": {
            "name": "ISICDataset",
            "root": "./data/ISICDataset",
            "metadata_path": "./ISIC_2020_Training_GroundTruth_v2.csv",
            "image_extension": "png",
            "batch_size": batch_size,
            "val_size": 0.1,
            "test_size": 0.0001,
        },
        "preprocess": {
            "img_size": 256,
        },
        "model": {
            "name": "ResnetClassifier",
            "num_classes": 2,  # Binary classification: 0 = benign, 1 = malignant
        },
        "trainer": {
            "name": "ResnetClassifierTrainer",
            "epochs": 200,
            "learning_rate": 0.0001,
            "lr_decay": 0.95,
            "loss": {
                "components": {
                    "MalignancyLoss": {
                        "lambda": 1.0,
                        "params": {},
                    },
                    "FocalLoss": {
                        "lambda": 1.0,
                        "params": {},
                    },
                },
            },
        },
        "callbacks": [
            {"name": "PlotCallback"},
            {
                "name": "EarlyStoppingCallback",
                "patience": 20,
                "min_delta": 0.01,
                "restore_best_weights": True,
            },
        ],
        "system": {
            "seed": 123,
            "augment": True,
            "save_name": "resnet18_baseline",
            "save_path": "./checkpoints/",
            "has_visual_attributes": False,
            "use_weighted_metrics": True,
        },
        "evaluate": {
            "split_size": 0.2,
        },
    }

    print("Hey there...")

    # --- Load Dataset (replace with your custom loader if needed) ---
    original_dataset = get_dataset(
        config=fake_config["dataset"],
        transform=get_transforms(config=fake_config, is_train=True),
    )
    # loaders = build_dataloaders(
    #     config=fake_config,
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     num_workers=1,
    # )
    dataset = BinaryImageDataset(original_dataset)

    print("Doing something...")

    # --- Split into train and val ---
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(0.8 * len(dataset))
    train_indices, val_indices = indices[:split], indices[split:]

    print("Doing something... Again")

    from tqdm import tqdm

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === STEP 1: Pretrain on Balanced Subset ===
    print("⚙️ Pretraining on balanced subset...")
    # balanced_subset = get_balanced_subset(train_dataset)
    # print(f"✅ Balanced subset size: {len(balanced_subset)} samples")

    all_labels = [int(train_dataset[i][1].item()) for i in range(len(train_dataset))]
    print(len(all_labels))

    counts = Counter(all_labels)

    num_benign = counts[0]
    num_malignant = counts[1]
    cls_counts = [num_benign, num_malignant]

    print(f"Class counts: {cls_counts}")

    sampler = ClassAwareSampler(
        labels=all_labels,
        batch_size=batch_size,
        minority_class=1,
        minority_fraction=0.25,
    )
    balanced_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, sampler=sampler
    )

    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    con_loss_fn = SupConLoss()
    triplet_loss = TripletLoss()

    save_path = "./checkpoints/zioperonga"
    best_auc = 0.0  # or -float('inf')

    for epoch in range(num_epochs_pretrain):
        print(f"🌀 Starting Pretrain Epoch {epoch+1}")
        model.train()
        total_loss = 0.0
        pbar = tqdm(balanced_loader, desc=f"[Pretrain Epoch {epoch+1}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            labels = apply_soft_labels(labels)
            features, logits = model(images)

            loss = loss_fn(logits, labels)
            loss += con_loss_fn(logits, labels)
            tr_loss = triplet_loss(features, labels)
            loss += tr_loss * 0.2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(balanced_loader.dataset)
        print(f"[Pretrain Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

    auc, report = evaluate(model, val_loader, device)
    print(f"[Pretrain Epoch {epoch+1}] AUC: {auc:.4f}\n{report}")

    if auc > best_auc:
        best_auc = auc
        save_model(model, save_path, name="best_pretrain_model.pth")

    # === STEP 2: Fine-tune on Full Imbalanced Dataset ===
    print("\n🎯 Fine-tuning on full dataset with focal loss...")
    full_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    loss_fn_margin = LDAMMarginLoss(cls_counts=cls_counts)

    for epoch in range(num_epochs_finetune):
        model.train()
        total_loss = 0.0
        pbar = tqdm(full_loader, desc=f"[Finetune Epoch {epoch+1}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            _, logits = model(images)

            labels = apply_soft_labels(labels)
            loss = loss_fn(logits, labels) + loss_fn_margin(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        auc, report = evaluate(model, val_loader, device)
        print(f"[Finetune Epoch {epoch+1}] AUC: {auc:.4f}\n{report}")

        if auc > best_auc:
            best_auc = auc
            save_model(model, save_path, name="best_finetune_model.pth")

    auc, report = evaluate(model, val_loader, device)
    print(f"[Final Evaluation] AUC: {auc:.4f}\n{report}")

    model_path = "./checkpoints/zioperonga/best_finetune_model.pth"

    # Split into val only
    val_size = int(0.1 * len(dataset))
    indices = list(range(len(dataset)))
    np.random.seed(123)
    np.random.shuffle(indices)
    val_indices = indices[-val_size:]
    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    def find_best_threshold(probs, labels):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.01, 0.99, 100):
            preds = (probs > thresh).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh, best_f1

    # === Load model ===
    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model = TemperatureScaler(model)  # Ensure this model returns (features, logits)

    model.eval()
    print(f"✅ Loaded model from: {model_path}")

    # === Inference ===
    all_features = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Running inference"):
            images = images.to(device)
            features, logits = model(images)  # Must return both
            all_features.append(features.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels.unsqueeze(1))

    features = torch.cat(all_features).numpy()
    logits = torch.cat(all_logits).squeeze()
    labels = torch.cat(all_labels).squeeze()

    # === Predictions ===
    probs = torch.sigmoid(logits)
    best_thresh, best_f1 = find_best_threshold(probs.numpy(), labels.numpy())
    print(f"📈 Best threshold: {best_thresh:.2f}, F1: {best_f1:.4f}")
    preds = (probs > best_thresh).int()

    # === Confusion Matrix ===
    cm = confusion_matrix(labels.numpy(), preds.numpy())
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Benign", "Malignant"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # === Logits Distribution Plot ===
    malignant_logits = logits[labels == 1].numpy()
    benign_logits = logits[labels == 0].numpy()

    plt.figure(figsize=(8, 5))
    sns.histplot(
        benign_logits, color="blue", label="Benign", kde=True, stat="density", bins=50
    )
    sns.histplot(
        malignant_logits,
        color="red",
        label="Malignant",
        kde=True,
        stat="density",
        bins=50,
    )
    plt.axvline(x=0.0, color="black", linestyle="--", label="Logit = 0")
    plt.title("Logits Distribution")
    plt.xlabel("Logit Value (pre-sigmoid)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logits.png")
    plt.close()

    # === AUC (Optional) ===
    auc = roc_auc_score(labels.numpy(), probs.numpy())
    print(f"🔍 Validation AUC: {auc:.4f}")

    # === t-SNE Visualization ===

    print("📉 Running t-SNE on extracted features...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for label_value in np.unique(labels.numpy()):
        idx = labels.numpy() == label_value
        plt.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            label="Malignant" if label_value == 1 else "Benign",
            alpha=0.6,
        )
    plt.legend()
    plt.title("t-SNE of Penultimate Layer Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsne_features.png")
    plt.close()


if __name__ == "__main__":
    main()
