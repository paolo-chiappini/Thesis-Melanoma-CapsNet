import os
from PIL import Image
from torchvision import transforms
import pandas as pd
from collections import Counter

safe_augmentations = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
)


def augment_dataset(
    dataset,
    augmentations_per_sample=5,
    output_dir="",
    output_filename="",
    augmentations=safe_augmentations,
    try_balance=True,
):
    pass
