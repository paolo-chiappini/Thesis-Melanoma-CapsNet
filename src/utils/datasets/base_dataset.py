from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image


class BaseDataset(Dataset):
    def __init__(
        self,
        root,
        metadata_path,
        image_path,
        transform=None,
        image_id="image",
        label="target",
        image_extension="jpg",
    ):
        self.root = root
        self.metadata_path = metadata_path
        self.image_path = image_path
        self.transform = transform
        self.image_id = image_id
        self.label = label
        self.image_extension = image_extension

        self.data = self.load_metadata()

    def load_metadata(self):
        df = pd.read_csv(os.path.join(self.root, self.metadata_path))
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data.iloc[index]
        image_id = record[self.image_id]
        label = record[self.label]

        image_path = os.path.join(
            self.root, self.image_path, image_id, image_id + "." + self.image_extension
        )
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
