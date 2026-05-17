from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_DIR = Path("dataset/tomato_split")
IMG_SIZE = 224
BATCH_SIZE = 32


train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_dataloaders():
    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val",
        transform=val_test_transforms
    )

    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=val_test_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print("Classes:")
    for idx, class_name in enumerate(classes):
        print(idx, class_name)

    print("\nNumber of classes:", len(classes))
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(train_loader))
    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)