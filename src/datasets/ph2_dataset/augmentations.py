import os
from PIL import Image
from torchvision import transforms
import pandas as pd
from collections import Counter

safe_augmentations = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
)


def augment_dataset(
    dataset,
    augmentations_per_sample=5,
    output_dir="./data/PH2Dataset/PH2_Dataset",
    output_filename="PH2_dataset_augmented.csv",
    augmentations=safe_augmentations,
    try_balance=True,
):
    os.makedirs(output_dir, exist_ok=True)

    class_count = Counter(dataset.labels)
    data_count = len(dataset)
    augmented_rows = []

    for _, row in dataset.data.iterrows():
        image_id = row[dataset.image_id]
        original_path = os.path.join(
            dataset.root,
            dataset.image_path,
            image_id,
            f"{image_id}_Dermoscopic_Image",
            f"{image_id}.{dataset.image_extension}",
        )
        image = Image.open(original_path).convert("RGB")

        augmentation_count = (
            int(
                augmentations_per_sample
                * (1 + (1 - class_count[row[dataset.label]] / data_count))
            )
            if try_balance
            else augmentations_per_sample
        )

        print(f"Augmentations for class {row[dataset.label]}: {augmentation_count}")

        for i in range(augmentation_count):
            augmented_image = augmentations(image)
            new_id = f"{image_id}_aug_{i}"

            # Save augmented image
            save_folder = os.path.join(
                output_dir, new_id, f"{new_id}_Dermoscopic_Image"
            )
            save_folder = os.path.normpath(save_folder)
            print(f"Saving to {save_folder}")
            os.makedirs(save_folder, exist_ok=True)
            transforms.ToPILImage()(augmented_image).save(
                os.path.join(save_folder, f"{new_id}.{dataset.image_extension}")
            )

            # Create new metadata entry
            new_row = row.copy()
            new_row[dataset.image_id] = new_id
            augmented_rows.append(new_row)

    df_augmented = pd.DataFrame(augmented_rows)
    df_combined = pd.concat([dataset.data, df_augmented], ignore_index=True)
    output_csv = os.path.normpath(os.path.join(dataset.root, output_filename))
    df_combined.to_csv(output_csv, index=False)

    return df_combined, output_csv
