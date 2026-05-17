import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = Path("dataset/tomato")
OUTPUT_DIR = Path("dataset/tomato_split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

random.seed(42)


def is_image(file_path):
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def create_folder(path):
    path.mkdir(parents=True, exist_ok=True)


def split_dataset():

    if not SOURCE_DIR.exists():
        print("Source folder not found!")
        return

    if OUTPUT_DIR.exists():
        print("Dataset already split.")
        return

    class_folders = [f for f in SOURCE_DIR.iterdir() if f.is_dir()]

    print(f"Found {len(class_folders)} classes")

    for class_folder in class_folders:

        class_name = class_folder.name

        images = [img for img in class_folder.iterdir() if is_image(img)]

        random.shuffle(images)

        total = len(images)

        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        for split_name, split_images in splits.items():

            split_dir = OUTPUT_DIR / split_name / class_name

            create_folder(split_dir)

            for image in split_images:

                shutil.copy2(
                    image,
                    split_dir / image.name
                )

        print(
            f"{class_name} -> "
            f"train:{len(train_images)} "
            f"val:{len(val_images)} "
            f"test:{len(test_images)}"
        )

    print("\nDataset split completed successfully!")


if __name__ == "__main__":
    split_dataset()