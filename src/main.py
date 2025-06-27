import config
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import transforms as T
from collections import Counter
from utils.loaders import get_dataset
from utils.losses.losses import CombinedLoss
from utils.callbacks import PlotCallback, ReconstructionCallback, CallbackManager
from trainers import trainer_with_attributes
from models.model_conv_attributes_32 import CapsuleNetworkWithAttributes32

trainer = trainer_with_attributes
size = 256

tensor_transform = T.Compose([T.Resize((500, 500)), T.ToTensor()])

epochs = 50
learning_rate = 1e-3
routing_steps = 3
lr_decay = 0.96
classes = range(2)  # Benign 0, Malignant 1


def main():
    args = config.args
    DATA_PATH = config.DATA_PATH
    batch_size = config.batch_size
    device = config.device
    multi_gpu = config.multi_gpu

    # switch between datasets
    dataset = get_dataset(
        args.dataset,
        DATA_PATH,
        transform=tensor_transform,
        metadata_path=args.meta,
        augment=args.augment,
    )
    if dataset is None:
        print(f"Dataset not found: {args.dataset}")
        exit()

    dataset.check_missing_files()

    class_counts = Counter(dataset.labels)
    print(f"Class counts: {class_counts}")

    train_idx, val_idx = next(
        config.splitter.split(np.zeros(len(dataset)), dataset.labels)
    )
    num_workers = 0 if not config.multi_gpu else 2

    train_transform = T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),  # Convert to tensor
        ]
    )

    val_transform = T.Compose(
        [
            T.Resize((size, size)),  # Resize to fixed size
            T.ToTensor(),
        ]
    )

    train_dataset = Subset(
        get_dataset(
            args.dataset,
            DATA_PATH,
            transform=train_transform,
            metadata_path=dataset.metadata_path,
        ),
        train_idx,
    )
    val_dataset = Subset(
        get_dataset(
            args.dataset,
            DATA_PATH,
            transform=val_transform,
            metadata_path=dataset.metadata_path,
        ),
        val_idx,
    )

    print(f"Len of train: {len(train_dataset)}")
    print(f"Len of validation: {len(val_dataset)}")

    class_counts = Counter(dataset.labels)
    counts = np.array(list(class_counts.values()), dtype=np.float32)
    class_weights = 1.0 / counts
    class_weights /= class_weights.sum()
    class_weights = torch.tensor(class_weights).to(device)
    print(f"Class counts: {class_counts}, Class weights: {class_weights}")

    loaders = {}
    loaders["train"] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    loaders["test"] = torch.utils.data.DataLoader(
        val_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    network = CapsuleNetworkWithAttributes32(
        img_shape=loaders["train"].dataset[0][0].numpy().shape,
        channels=3,
        primary_dim=8,
        num_classes=2,
        num_attributes=loaders["train"].dataset[0][2].shape[0],
        output_dim=16,
        routing_steps=routing_steps,
        device=device,
        routing_algorithm="sigmoid",
    )

    loss_criterion = CombinedLoss(class_weights=class_weights)

    caps_net = trainer.CapsNetTrainer(
        loaders,
        batch_size,
        learning_rate,
        lr_decay,
        network=network,
        device=device,
        multi_gpu=multi_gpu,
        criterion=loss_criterion,
    )

    callbacks = [
        PlotCallback(),
        ReconstructionCallback(frequency=3),
    ]
    callback_manager = CallbackManager(callbacks)

    caps_net.run(epochs, classes, callback_manager=callback_manager)
    print("=" * 10, "Run finished", "=" * 10)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
    torch.cuda.empty_cache()
