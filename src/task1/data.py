from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    num_classes: int


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.65, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.15),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, eval_tfms


def build_flowers102_loaders(
    *,
    root: str | Path,
    batch_size: int,
    image_size: int,
    num_workers: int,
    download: bool = True,
) -> DataLoaders:
    train_tfms, eval_tfms = build_transforms(image_size)
    root = Path(root)

    train_set = datasets.Flowers102(root=root, split="train", transform=train_tfms, download=download)
    val_set = datasets.Flowers102(root=root, split="val", transform=eval_tfms, download=download)
    test_set = datasets.Flowers102(root=root, split="test", transform=eval_tfms, download=download)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    return DataLoaders(
        train=DataLoader(train_set, shuffle=True, drop_last=False, **loader_kwargs),
        val=DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs),
        test=DataLoader(test_set, shuffle=False, drop_last=False, **loader_kwargs),
        num_classes=102,
    )

