from .datasets.ph2_dataset.ph2_loader import PH2Dataset
from .datasets.exham_dataset.exham_loader import EXHAMDataset
from .datasets.isic_dataset.isic_loader import ISICDataset

DATASET_REGISTRY = {
    "PH2": PH2Dataset,
    "EXHAM": EXHAMDataset,
    "ISIC": ISICDataset,
    # Add other datasets here
}


def get_dataset(
    dataset_name,
    root,
    metadata_path=None,
    transform=None,
    image_extension=None,
    augment=False,
):
    """
    Get the dataset class based on the dataset name.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} not found in registry.")
    dataset_class = DATASET_REGISTRY[dataset_name]
    return dataset_class(
        root=root,
        transform=transform,
        image_extension=image_extension,
        metadata_path=metadata_path,
        augment=augment,
    )
