from ..base_dataset import BaseDataset
import os
import torch
from PIL import Image


class ISICDataset(BaseDataset):
    def __init__(
        self,
        root,
        image_extension,
        metadata_path=None,
        transform=None,
        image_id="image_name",
        label="benign_malignant",
        augment=False,
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

        # if augment:
        #     _, self.metadata_path = augment_dataset(self)
        #     self.load_metadata()  # Force metadata reload
        #     self.labels = self.data[self.label]

        # self.load_segmentations = load_segmentations
        # self.segmentations_path = "segmentations"
        # self.segmentation_extension = "png"

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

        # segmentation_path = os.path.join(
        #     self.root,
        #     self.segmentations_path,
        #     image_id + "_segmentation." + self.segmentation_extension,
        # )

        image_path = os.path.normpath(image_path)
        # segmentation_path = os.path.normpath(segmentation_path)

        image = Image.open(image_path).convert("RGB")
        # segmentation = (
        #     Image.open(segmentation_path).convert("L")
        #     if self.load_segmentations
        #     else None
        # )

        if self.transform:
            image = self.transform(image)
            # segmentation = (
            #     self.transform(segmentation) if segmentation is not None else None
            # )

        label = torch.tensor(label, dtype=torch.int)

        # return (image, label, visual_features, segmentation)
        return (image, label)

    def check_missing_files(self):
        full_image_path = lambda _: self.image_path

        super().check_missing_files(full_image_path, "image_id")
