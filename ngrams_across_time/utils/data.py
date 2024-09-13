from torch.utils.data import Dataset as TorchDataset
from typing import List



class ZippedDataset(TorchDataset):
    def __init__(self, target_dataset: TorchDataset, low_order_dataset: TorchDataset, 
                 high_order_dataset: TorchDataset, base_dataset: TorchDataset):
        self.target_dataset = target_dataset
        self.low_order_dataset = low_order_dataset
        self.high_order_dataset = high_order_dataset
        self.base_dataset = base_dataset
        assert len(self.target_dataset) == len(self.low_order_dataset) == len(self.high_order_dataset) == len(self.base_dataset)

    def select(self, idx: List[int]):
        return ZippedDataset(
            target_dataset=self.target_dataset.select(idx),
            low_order_dataset=self.low_order_dataset.select(idx),
            high_order_dataset=self.high_order_dataset.select(idx),
            base_dataset=self.base_dataset.select(idx),
        )

    def __getitem__(self, idx: int):
        return self.target_dataset[idx], self.low_order_dataset[idx], self.high_order_dataset[idx], self.base_dataset[idx]
    
    def __len__(self) -> int:
        return len(self.target_dataset)