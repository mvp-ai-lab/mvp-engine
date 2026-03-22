"""Dataset builders for the ViT image classification recipe."""

from pathlib import Path

from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset

from ..configs.schema import ViTClassificationConfig

IMAGENET_DEFAULT_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD: list[float] = [0.229, 0.224, 0.225]


def build_transforms(
    image_size: int,
    is_train: bool,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    """Build standard ImageNet transforms for ViT classification."""
    mean = mean or IMAGENET_DEFAULT_MEAN
    std = std or IMAGENET_DEFAULT_STD

    if is_train:
        transform_list = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        resize_size = int(image_size / 0.875)
        transform_list = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
        ]

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transform_list)


def build_dataset(config: ViTClassificationConfig, workflow: str) -> VisionDataset:
    """Build either a fake ImageNet-like dataset or a real ImageFolder dataset."""
    is_train = workflow == "train"
    image_size = int(config.data.image_size)
    transform = build_transforms(
        image_size=image_size,
        is_train=is_train,
        mean=list(config.data.mean),
        std=list(config.data.std),
    )

    if config.data.use_fake_data:
        size = int(config.data.fake_train_size if is_train else config.data.fake_val_size)
        return datasets.FakeData(
            size=size,
            image_size=(3, image_size, image_size),
            num_classes=int(config.data.num_classes),
            transform=transform,
        )

    data_root = config.data.train_path if is_train else config.data.val_path
    dataset_path = Path(data_root)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{workflow} dataset path does not exist: {dataset_path}. "
            "Set data.use_fake_data=true for the template smoke run, or point train_path/val_path to ImageNet folders."
        )

    return datasets.ImageFolder(root=str(dataset_path), transform=transform)
