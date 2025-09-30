import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from ..base_dataset import BaseDataset

# TODO: make this better
normalize_transform = T.Compose([T.ToTensor()])


class ToyDataset(BaseDataset):
    def __init__(
        self,
        root,
        image_extension,
        metadata_path=None,
        transform=None,
        image_id="file",
        label="label",
        load_segmentations=True,
    ):
        image_extension = image_extension if image_extension is not None else "png"

        super().__init__(
            root,
            (metadata_path if metadata_path is not None else "ground_truth.csv"),
            image_path="images",
            transform=transform,
            image_id=image_id,
            label=label,
            image_extension=image_extension,
        )

        self.visual_attributes = [
            "circle",
            "square",
            "triangle",
            "ring",
            "stripe",
            "None",
        ]
        self.labels = self.data[self.label]

        self.data["None"] = 1

        self.visual_features = torch.tensor(
            self.data[self.visual_attributes].values, dtype=torch.float
        )

        self.load_segmentations = load_segmentations
        self.segmentations_path = "segmentations"
        self.segmentation_extension = "png"

        print("[TOY] Loaded dataset with", len(self.data), "rows")

    def __getitem__(self, index):
        if index >= len(self.data):
            raise IndexError(
                f"Index {index} out of bounds for dataset of size {len(self.data)}"
            )

        record = self.data.iloc[index]
        image_id = record[self.image_id].strip(".png")
        label = record[self.label]

        image_path = os.path.join(
            self.root,
            self.image_path,
            f"{image_id}.{self.image_extension}",
        )

        segmentation_path = os.path.join(
            self.root,
            self.segmentations_path,
            f"{image_id}_segmentation.{self.segmentation_extension}",
        )

        image_path = os.path.normpath(image_path)
        segmentation_path = os.path.normpath(segmentation_path)

        image = Image.open(image_path).convert("RGB")
        segmentation = (
            Image.open(segmentation_path).convert("L")
            if self.load_segmentations
            else None
        )

        va_masks = self.load_va_masks(
            os.path.normpath(os.path.join(self.root, "va_masks")), image_id, image.size
        )

        if self.transform:
            image_np = np.array(image).astype(np.uint8)
            segmentation_np = (
                np.array(segmentation) if segmentation is not None else None
            ).astype(np.uint8)
            va_masks_np = [np.array(mask).astype(np.uint8) for mask in va_masks]

            transformed = self.transform(
                image=image_np, masks=[segmentation_np] + va_masks_np
            )

            image = normalize_transform(transformed["image"])
            masks = transformed["masks"]
            lesion_mask = normalize_transform(masks[0])
            attribute_masks = normalize_transform(np.stack(masks[1:], axis=-1))

            for mask in attribute_masks:
                assert (
                    mask.shape == image.shape[1:]
                ), f"Image and masks have different shapes for lesion {image_id}. Mask size {mask.shape}. Image size: {image.shape}."

        label = torch.tensor(label, dtype=torch.int)
        visual_attributes = record[self.visual_attributes].values.astype(float)
        visual_attributes = torch.tensor(visual_attributes, dtype=torch.float)

        # return (image, label, visual_features, segmentation)
        return {
            "images": image,
            "malignancy_targets": label,
            "visual_attributes_targets": visual_attributes,
            "lesion_masks": lesion_mask,
            "va_masks": attribute_masks,
        }

    def check_missing_files(self):
        def full_image_path(_):
            return self.image_path

        super().check_missing_files(full_image_path, "image_id")

    def load_va_masks(self, path, lesion_id, img_size):
        files_for_lesion = []
        for va in self.visual_attributes:
            if va == "None":
                # Create the None mask
                union_mask = np.any(files_for_lesion, axis=0).astype(np.uint8)
                inverted_mask = (1 - union_mask) * 255

                files_for_lesion.append(inverted_mask)
                continue

            file_path = os.path.join(
                path, f"{lesion_id}_{va}.{self.segmentation_extension}"
            )
            file_path = os.path.normpath(file_path)
            if os.path.exists(file_path):
                mask = Image.open(file_path).convert("L")
                mask = mask.resize(size=img_size)
            else:
                mask = Image.new("L", img_size, 0)  # create blank mask

            files_for_lesion.append(mask)

        return files_for_lesion
