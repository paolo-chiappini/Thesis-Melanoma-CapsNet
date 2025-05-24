from ..base_dataset import BaseDataset
import os
import torch
from PIL import Image
from .augmentations import augment_dataset


class PH2Dataset(BaseDataset):
    def __init__(
        self,
        root,
        metadata_path=None,
        transform=None,
        image_id="image_name",
        label="diagnosis_melanoma",
        image_extension="bmp",
        augment=False,
    ):
        super().__init__(
            root,
            (
                metadata_path
                if metadata_path is not None
                else "PH2_dataset_preprocessed.csv"
            ),
            "PH2_Dataset",
            transform,
            image_id,
            label,
            image_extension,
        )

        self.visual_attributes = [
            "asymmetry_asymmetric",
            "asymmetry_symmetric_1_axis",
            "pigment_network",
            "dots_globules",
            "streaks",
            "regression_areas",
            "blue_whitish_veil",
            "color_white",
            "color_red",
            "color_light_brown",
            "color_dark_brown",
            "color_blue_gray_brown",
            "color_black",
        ]
        self.labels = self.data[self.label]

        if augment:
            _, self.metadata_path = augment_dataset(self)
            self.load_metadata()  # Force metadata reload
            self.labels = self.data[self.label]

        print("[PH2] Loaded dataset with", len(self.data), "rows")

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
            image_id,
            image_id + "_Dermoscopic_Image",
            image_id + "." + self.image_extension,
        )
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.int)
        visual_features = record[self.visual_attributes].values.astype(float)
        visual_features = torch.tensor(visual_features, dtype=torch.float)

        return image, label, visual_features
