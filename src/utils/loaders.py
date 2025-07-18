from .datasets.base_dataset import BaseDataset
from .datasets.ph2_dataset.ph2_loader import PH2Dataset
from .datasets.exham_dataset.exham_loader import EXHAMDataset
from .datasets.isic_dataset.isic_loader import ISICDataset

DATASET_REGISTRY = {cls.__name__: cls for cls in BaseDataset.__subclasses__()}


def get_dataset(
    config,
    transform=None,
):
    """
    Get the dataset class based on the dataset name.
    """
    name = config["name"]
    root = config["root"]
    metadata_path = config["metadata_path"]
    image_extension = config["image_extension"]
    augment = config["augment"]

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} not found in registry.")
    dataset_class = DATASET_REGISTRY[name]
    return dataset_class(
        root=root,
        transform=transform,
        image_extension=image_extension,
        metadata_path=metadata_path,
        augment=augment,
    )
