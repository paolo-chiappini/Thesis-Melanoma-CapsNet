import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentations(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            # ToTensorV2(),
        ]
    )


def get_val_augmentations(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            # ToTensorV2(), # TODO: review this (may be buggy), currently replaced by torchvision ToTensor()
        ]
    )


def get_transforms(config, is_train=True):
    img_size = config["preprocess"]["img_size"]
    augment = config["dataset"].get("augment", False)
    # if is_train and augment:
    if is_train:
        return get_train_augmentations(img_size)
    else:
        return get_val_augmentations(img_size)
