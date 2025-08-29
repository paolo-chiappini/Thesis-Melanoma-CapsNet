import pydicom as dicom
import yaml
import os
from PIL import Image
from tqdm import tqdm
import argparse


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["preprocess"]["img_size"]


def preprocess_images(
    input_dir, output_dir="./data/ISICDataset/train_preprocessed", size=(256, 256)
):

    pbar = tqdm(os.listdir(input_dir))
    for file in pbar:
        input_path = os.path.normpath(os.path.join(input_dir, file))
        output_path = os.path.normpath(
            os.path.join(output_dir, file.replace(".dcm", ".png"))
        )

        try:
            dicom_image = dicom.dcmread(input_path).pixel_array
            image = Image.fromarray(dicom_image).convert("RGB")
            image = image.resize(size=size)
            pbar.display(f"Processing file: {file}")
            image.save(output_path)
            pbar.update()
        except Exception as e:
            print(f"Failed to process {file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=False)
    args = parser.parse_args()

    size = load_config(args.config)
    preprocess_images(args.input, args.output, (size, size))
