#!/usr/bin/env python3
"""
Smoke test for the DataLoader path — runs on CPU, no CUDA or DDP needed.
Iterates a few batches and verifies shapes/dtypes are correct before deploying to EC2.

Usage:
    python3 smoke_test_dataloader.py /path/to/BUPT-CBFace-12/images
"""
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

BATCH_SIZE = 8
NUM_WORKERS = 2
NUM_BATCHES = 3

def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return imgs, labels

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/dataset/images")
        sys.exit(1)

    root = sys.argv[1]
    print(f"Dataset: {root}")

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    print(f"Classes: {len(dataset.classes)}  Samples: {len(dataset)}")

    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # CPU-only test
        drop_last=True,
        collate_fn=collate_fn,
    )

    print(f"\nIterating {NUM_BATCHES} batches (batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS})...\n")

    for i, (imgs, labels) in enumerate(loader):
        if i >= NUM_BATCHES:
            break
        print(f"  Batch {i+1}:")
        print(f"    imgs  shape={imgs.shape}  dtype={imgs.dtype}  "
              f"min={imgs.min():.3f}  max={imgs.max():.3f}")
        print(f"    labels shape={labels.shape}  dtype={labels.dtype}  "
              f"values={labels.tolist()}")

        assert imgs.shape == (BATCH_SIZE, 3, 112, 112), f"Unexpected img shape: {imgs.shape}"
        assert labels.shape == (BATCH_SIZE,), f"Unexpected label shape: {labels.shape}"
        assert labels.dtype == torch.long, f"Unexpected label dtype: {labels.dtype}"

    print(f"\nPASS — DataLoader is healthy.")

if __name__ == "__main__":
    main()
