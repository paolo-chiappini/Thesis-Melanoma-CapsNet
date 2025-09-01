import os

import numpy as np
import pydicom as dicom
import torch
import torchvision.transforms as T
from PIL import Image

from ..base_dataset import BaseDataset

LABEL_MAP = {"benign": 0, "malignant": 1}

# TODO: make this better
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

normalize_tranform = T.Compose(
    [T.ToTensor(), T.Normalize(mean=imagenet_mean, std=imagenet_std)]
)


# TODO: Fix this
class ISICDataset(BaseDataset):
    def __init__(
        self,
        root,
        image_extension,
        metadata_path=None,
        transform=None,
        image_id="image_name",
        label="benign_malignant",
        load_segmentations=False,
    ):
        image_extension = image_extension if image_extension is not None else "jpg"

        super().__init__(
            root,
            (
                metadata_path
                if metadata_path is not None
                else "Datasets/metadata/metadata_ground_truth.csv"
            ),
            image_path="train_preprocessed",
            transform=transform,
            image_id=image_id,
            label=label,
            image_extension=image_extension,
        )

        self.labels = self.data[self.label]
        self.groups = self.data["patient_id"].values

        print("[ISIC 2020] Loaded dataset with", len(self.data), "rows")

    def __getitem__(self, index):
        if index >= len(self.data):
            raise IndexError(
                f"Index {index} out of bounds for dataset of size {len(self.data)}"
            )

        record = self.data.iloc[index]
        image_id = record[self.image_id]
        label = record[self.label]

        image_path = os.path.join(
            self.root,
            self.image_path,
            image_id + "." + self.image_extension,
        )

        image_path = os.path.normpath(image_path)
        if self.image_extension == "dcm":
            image = dicom.dcmread(image_path).pixel_array
        else:
            image = Image.open(image_path).convert("RGB")

        if self.transform:
            image_np = np.array(image).astype(np.uint8)
            transformed = self.transform(image=image_np)
            image = normalize_tranform(transformed["image"])

        label = torch.tensor(LABEL_MAP[label], dtype=torch.float)

        # return {"image": image, "label": label}
        return image, label

    def check_missing_files(self):
        full_image_path = lambda _: self.image_path

        super().check_missing_files(full_image_path, "image_id")
