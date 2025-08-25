import numpy as np
from ..base_dataset import BaseDataset
import os
import torch
from PIL import Image
import torchvision.transforms as T


LABEL_MAP = {"benign": 0, "malignant": 1}

# TODO: make this better
normalize_tranform = T.Compose([T.ToTensor()])


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
            image_path="train",
            transform=transform,
            image_id=image_id,
            label=label,
            image_extension=image_extension,
        )

        self.labels = self.data[self.label]

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
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image_np = np.array(image).astype(np.uint8)
            transformed = self.transform(image=image_np)
            image = normalize_tranform(transformed["image"])

        label = torch.tensor([LABEL_MAP[label]], dtype=torch.float)

        return (image, label)

    def check_missing_files(self):
        full_image_path = lambda _: self.image_path

        super().check_missing_files(full_image_path, "image_id")
