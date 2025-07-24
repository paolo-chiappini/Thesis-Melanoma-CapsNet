from utils.loaders import get_dataset
from models import get_model
from utils.commons import get_resize_transform
from config.device_config import get_device
import torch
from torch.utils.data import DataLoader
from utils.visualization.capsule_contribution import perturb_all_capsules
from tqdm import tqdm


def get_samples_from_classes(loader, num_samples=3, num_classes=2):
    classes = range(num_classes)
    class_images = {i: [] for i in classes}

    with tqdm(total=len(loader), desc="Collecting samples") as pbar:
        for batch in loader:
            images, labels = batch[0], batch[1]
            for img, label in zip(images, labels):
                label = label.item()
                if len(class_images[label]) < num_samples:
                    class_images[label].append(img)

            if all((len(class_images[l]) >= num_samples for l in classes)):
                break

            print([len(class_images[l]) for l in class_images])
            pbar.update(1)

    print("Finished collecting samples")

    for label in classes:
        class_images[label] = torch.stack(class_images[label])

    all_images = torch.cat(list(class_images.values()), dim=0)
    return all_images


def run_perturbation(config, model_path=None, cpu_override=False):
    visualization_config = config["perturbation"]
    dataset_config = config["dataset"]
    preprocess_config = config["preprocess"]
    model_config = config["model"]

    device, multi_gpu = get_device(cpu_override=cpu_override)

    tranform = get_resize_transform(preprocess_config["img_size"])
    dataset = get_dataset(dataset_config, transform=tranform)

    num_workers = 0 if not multi_gpu else 2

    batch_size = dataset_config["batch_size"]
    if device.type == "cuda" and multi_gpu:
        batch_size *= torch.cuda.device_count()

    loader = {
        "test": DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    }

    sampled_images = get_samples_from_classes(
        loader["test"], visualization_config["num_samples"], model_config["num_classes"]
    )

    model = get_model(model_config, data_loader=loader, device=device)
    model.load_state_dict(
        torch.load(
            visualization_config["model_name"],
            weights_only=False,
            map_location=torch.device(device),
        )
    )

    # generate global perturbations
    for i, image in enumerate(sampled_images):
        perturb_all_capsules(
            model.to(device),
            image,
            device=device,
            visual_attributes=dataset.visual_attributes,
            out_prefix=f"img{i}_global_label{i // visualization_config['num_samples']}",  # TODO: check assumption: loop in class order
            global_perturbation=visualization_config["is_global"],
        )
