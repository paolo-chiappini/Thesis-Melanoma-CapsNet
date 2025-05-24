from ..base_dataset import BaseDataset
import os
import torch
from PIL import Image


class PH2Dataset(BaseDataset):
    def __init__(
        self,
        root,
        transform=None,
        image_id="image_name",
        label="diagnosis_melanoma",
        image_extension="bmp",
    ):
        super().__init__(
            root,
            "PH2_dataset_preprocessed.csv",
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

    def __getitem__(self, index):
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
