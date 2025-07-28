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
        load_segmentations=True,
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

        # Feature importance Random Forest (available on Kaggle)
        # TRBL	0.200910
        # GP	0.168374
        # MS	0.113095
        # BDG	0.110860
        # ESA	0.102919
        # WLSA	0.090970
        # APC	0.064269
        # -------------------- threshold
        # SPC	0.030327
        # PV	0.029661
        # PRL	0.027858
        # OPC	0.018847
        # None	0.016453
        # PIF	0.012541
        # PRLC	0.004654
        # PLR	0.003681
        # PLF	0.002315
        # PES	0.000885
        # MVP	0.000823
        # PDES	0.000556

        self.visual_attributes = [
            "APC",
            "BDG",
            "ESA",
            "GP",
            "MS",
            # "MVP",
            # "None",
            # "OPC",
            # "PDES",
            # "PES",
            # "PIF",
            # "PLF",
            # "PLR",
            # "PRL",
            # "PRLC",
            # "PV",
            # "SPC",
            "TRBL",
            "WLSA",
        ]
        self.labels = self.data[self.label]

        self.visual_features = torch.tensor(
            self.data[self.visual_attributes].values, dtype=torch.float
        )

        if augment:
            _, self.metadata_path = augment_dataset(self)
            self.load_metadata()  # Force metadata reload
            self.labels = self.data[self.label]

        self.load_segmentations = load_segmentations
        self.segmentations_path = "segmentations"
        self.segmentation_extension = "png"

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

        segmentation_path = os.path.join(
            self.root,
            self.segmentations_path,
            image_id + "_segmentation." + self.segmentation_extension,
        )

        image_path = os.path.normpath(image_path)
        segmentation_path = os.path.normpath(segmentation_path)

        image = Image.open(image_path).convert("RGB")
        segmentation = (
            Image.open(segmentation_path).convert("L")
            if self.load_segmentations
            else None
        )

        if self.transform:
            image = self.transform(image)
            segmentation = (
                self.transform(segmentation) if segmentation is not None else None
            )

        label = torch.tensor(label, dtype=torch.int)
        visual_features = record[self.visual_attributes].values.astype(float)
        visual_features = torch.tensor(visual_features, dtype=torch.float)

        return (image, label, visual_features, segmentation)

    def check_missing_files(self):
        full_image_path = lambda _: self.image_path

        super().check_missing_files(full_image_path, "image_id")
