import pandas as pd
import os
from os import path
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def preprocess_metadata(root, raw_csv, output_csv):
    df = pd.read_excel(path.join(root, raw_csv))

    df.columns = df.columns.str.replace("\n", " ", regex=True).str.strip()
    colname_mapping = {
        "Image Name": "image_name",
        "Histological Diagnosis": "histological_diagnosis",
        "Common Nevus": "diagnosis_common_nevus",
        "Atypical Nevus": "diagnosis_atypical_nevus",
        "Melanoma": "diagnosis_melanoma",
        "Asymmetry (0/1/2)": "asymmetry",
        "Pigment Network (AT/T)": "pigment_network",
        "Dots/Globules (A/AT/T)": "dots_globules",
        "Streaks (A/P)": "streaks",
        "Regression Areas (A/P)": "regression_areas",
        "Blue-Whitish Veil (A/P)": "blue_whitish_veil",
        "White": "color_white",
        "Red": "color_red",
        "Light-Brown": "color_light_brown",
        "Dark-Brown": "color_dark_brown",
        "Blue-Gray": "color_blue_gray_brown",
        "Black": "color_black",
    }
    df.rename(columns=colname_mapping, inplace=True)

    assert (
        "histological_diagnosis" in df.columns
    ), "Error during preprocessing: could not find histological_diagnosis"

    cols_to_fill = df.columns[df.columns.get_loc("histological_diagnosis") + 1 :]
    df[cols_to_fill] = df[cols_to_fill].fillna(0)
    df[cols_to_fill] = df[cols_to_fill].replace(
        {"X": 1, "A": 0, "P": 1, "AT": 0, "T": 1}
    )

    assert (
        "asymmetry" in df.columns
    ), "Error during preprocessing: could not find asymmetry"

    df["asymmetry"] = df["asymmetry"].map(
        {0: "fully_symmetric", 1: "symmetric_1_axis", 2: "asymmetric"}
    )

    df_hist_diag_encoded = pd.get_dummies(
        df["histological_diagnosis"], dummy_na=True, drop_first=False, dtype=int
    ).add_prefix("hist_")
    df_asymm_encoded = pd.get_dummies(
        df["asymmetry"], dummy_na=False, drop_first=False, dtype=int
    ).add_prefix("asymmetry_")

    df_new = pd.concat(
        [
            df["image_name"],
            df_hist_diag_encoded,
            df_asymm_encoded,
            df.loc[
                :,
                ~df.columns.str.contains("|".join(["image_name", "asymmetry", "hist"])),
            ],
        ],
        axis=1,
    )
    df_new.columns = (
        df_new.columns.str.replace(" ", "_", regex=True).str.strip().str.lower()
    )

    df_new.to_csv(path.join(root, output_csv), index=False)
    print(
        f"> Finished preprocessing dataset: results available at {path.join(root, output_csv)}"
    )


def create_masks_from_polygons(root, gt_folder, output_folder):
    width, height = 600, 450

    output_root = path.normpath(path.join(root, output_folder))
    os.makedirs(output_root, exist_ok=True)

    input_root = path.normpath(path.join(root, gt_folder))

    json_files = []
    for annotator_folder in os.listdir(input_root):
        annotator_path = path.join(input_root, annotator_folder)

        if not os.path.isdir(annotator_path):
            continue

        for filename in os.listdir(annotator_path):
            if filename.endswith(".json"):
                json_files.append(path.join(annotator_path, filename))

    def is_json_string(content):
        try:
            obj = json.loads(content)
            return isinstance(obj, dict) or isinstance(obj, str)
        except json.JSONDecodeError:
            return False

    for json_path in tqdm(json_files, desc="Mask generation", unit="file"):
        patient_id = path.splitext(path.basename(json_path))[0]

        with open(json_path, "r") as f:
            raw = f.read()

        try:
            data = json.loads(raw)
            # If the content itself is a JSON string, decode again
            if isinstance(data, str) and is_json_string(data):
                data = json.loads(data)

            # At this point, data should be a proper dictionary
            if isinstance(data, dict):
                # Overwrite file with cleaned version
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to clean {json_path}: {e}")

        for label, polygons in data.items():
            mask = np.zeros((height, width), dtype=np.uint8)

            for polygon in polygons:
                points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [points], 1)

                out_filename = f"{patient_id}_{label}.png"
                out_path = path.join(output_root, out_filename)
                Image.fromarray(mask * 255).save(out_path)

    print(f"> Finished generating masks: results available at {output_root}")


if __name__ == "__main__":
    import argparse

    pd.set_option("future.no_silent_downcasting", True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/EXHAMDataset")
    parser.add_argument("--raw")
    parser.add_argument("--out")
    parser.add_argument("--gt_folder", default="Datasets/ground_truth_annotations")
    parser.add_argument("--masks_out", default="va_masks")
    args = parser.parse_args()

    preprocess_metadata(args.root, args.raw, args.out)
    create_masks_from_polygons(args.root, args.gt_folder, args.masks_out)
