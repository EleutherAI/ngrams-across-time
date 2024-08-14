from argparse import ArgumentParser
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
import numpy as np

# Could put all the ngram dists here and have a custom get to get them

class MemoryMappedDataset(Dataset):
    def __init__(self, source_dataset, target_data):
        self.source_dataset = source_dataset
        self.target_data = target_data

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):
        item = self.source_dataset[idx]
        item['label'] = torch.from_numpy(self.target_data[idx:idx+1])
        return item

def main():
    source_dataset_path = '/mnt/ssd-1/lucia/features-across-time/data/pile-deduped/val_tokenized.hf'
    target_dataset_path = '/mnt/ssd-1/lucia/features-across-time/data/pile-deduped/smoothed-1-gram-pile-dists-16bit.npy'

    source_dataset: Dataset = load_from_disk(source_dataset_path) # type: ignore
    source_dataset = source_dataset.select(range(1024))
    target_data = np.memmap(target_dataset_path, mode='r', dtype=np.float16)

    new_dataset = MemoryMappedDataset(source_dataset, target_data)

    print(f"Dataset size: {len(new_dataset)}")
    print(f"Features: {list(new_dataset[0].keys())}")

if __name__ == "__main__":
    main()