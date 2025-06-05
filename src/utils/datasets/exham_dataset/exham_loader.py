from ..base_dataset import BaseDataset
import os
import torch
from PIL import Image
from .augmentations import augment_dataset


class EXHAMDataset(BaseDataset):
    def __init__(
        self,
        root,
        image_extension,
        metadata_path=None,
        transform=None,
        image_id="image_id",
        label="benign_malignant",
        augment=False,
    ):
        image_extension = image_extension if image_extension is not None else "jpg"

        super().__init__(
            root,
            (
                metadata_path
                if metadata_path is not None
                else "Datasets/metadata/metadata_ground_truth.csv"
            ),
            image_path="images",
            transform=transform,
            image_id=image_id,
            label=label,
            image_extension=image_extension,
        )

        self.visual_attributes = [
            "APC",
            "BDG",
            "ESA",
            "GP",
            "MS",
            "MVP",
            "None",
            "OPC",
            "PDES",
            "PES",
            "PIF",
            "PLF",
            "PLR",
            "PRL",
            "PRLC",
            "PV",
            "SPC",
            "TRBL",
            "WLSA",
        ]
        self.labels = self.data[self.label]

        if augment:
            _, self.metadata_path = augment_dataset(self)
            self.load_metadata()  # Force metadata reload
            self.labels = self.data[self.label]

        print("[EXHAM] Loaded dataset with", len(self.data), "rows")

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
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.int)
        visual_features = record[self.visual_attributes].values.astype(float)
        visual_features = torch.tensor(visual_features, dtype=torch.float)

        return image, label, visual_features

    def check_missing_files(self):
        full_image_path = lambda _: self.image_path

        super().check_missing_files(full_image_path, "image_id")
