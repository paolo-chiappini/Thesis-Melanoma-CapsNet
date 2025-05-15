from ..base_dataset import BaseDataset
import os
from PIL import Image

class PH2Dataset(BaseDataset): 
    def __init__(self, root, transform=None, image_id='image_name', label='diagnosis_melanoma', image_extension='bmp'):
        super().__init__(root, 'PH2_dataset_preprocessed.csv', 'PH2_Dataset', transform, image_id, label, image_extension)
        
    def __getitem__(self, index):
        record = self.data.iloc(index)
        image_id = record[self.image_id]
        label = record(self.label)
        
        image_path = os.path.join(self.root, self.image_path, image_id, image_id + '_Dermoscopic_Image', image_id + self.image_extension)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label