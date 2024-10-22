from argparse import ArgumentParser
from typing import Callable, Sized
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision as tv
from concept_erasure import LeaceFitter, OracleFitter, QuadraticFitter, LeaceEraser, QuadraticEraser
from torch.utils.data import Subset
from torch import Tensor, nn
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm


torch.set_default_tensor_type(torch.DoubleTensor)

class LeacedDataset(Dataset):
    """Wrapper for a dataset of (X, Z) pairs that erases Z from X"""
    def __init__(
        self,
        inner: Dataset[tuple[Tensor, ...]],
        eraser: Callable,
        transform: Callable[[Tensor], Tensor] = lambda x: x,
    ):
        # Pylance actually keeps track of the intersection type
        assert isinstance(inner, Sized), "inner dataset must be sized"
        assert len(inner) > 0, "inner dataset must be non-empty"

        self.dataset = inner
        self.eraser = eraser
        self.transform = transform

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x, z = self.dataset[idx]

        # Erase BEFORE transforming
        if isinstance(self.eraser, LeaceEraser):
            x_erased = self.eraser(x.flatten())
        else:
            z_tensor = torch.tensor(data=z)
            if z_tensor.ndim == 0:
                z_tensor = z_tensor.unsqueeze(0)
            x_erased = self.eraser(x.unsqueeze(0), z_tensor)
        return self.transform(x_erased), z

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eraser", type=str, choices=("none", "leace", "oleace", "qleace"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nontest = CIFAR10(
        "/home/lucia/cifar10", download=True, transform=tv.transforms.ToTensor()
    )
    train, val = random_split(nontest, [0.9, 0.1])

    test = CIFAR10(
        "/home/lucia/cifar10-test",
        download=True,
        train=False,
        transform=tv.transforms.ToTensor(),
    )
    k = 10  # Number of classes
    final = nn.Flatten(0)
    train_trf = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomCrop(32, padding=4),
        final,
    ])
    if args.eraser != "none":
        cls = {
            "leace": LeaceFitter,
            "oleace": OracleFitter,
            "qleace": QuadraticFitter,
        }[args.eraser]

        fitter = cls(3 * 32 * 32, k, dtype=torch.float64, device=device, shrinkage=False)
        if args.debug:
            train = Subset(train, range(100))
            eraser = QuadraticEraser(
                torch.zeros(3 * 32 * 32, dtype=torch.float64), 
                global_mean=torch.zeros(3 * 32 * 32, dtype=torch.float64),
                ot_maps=torch.zeros(10, 3 * 32 * 32, 3 * 32 * 32, dtype=torch.float64),
            )
        else:
            for x, y in tqdm(train):
                y = torch.as_tensor(y).view(1)
                if args.eraser != "qleace":
                    y = F.one_hot(y, k)

                fitter.update(x.view(1, -1).to(device), y.to(device))

            eraser = fitter.eraser.to("cpu")
    else:
        eraser = lambda x, y: x
    leaced_train = LeacedDataset(train, eraser, transform=train_trf)
    non_augmented_leaced_train = LeacedDataset(train, eraser, transform=final)
    leaced_val = LeacedDataset(val, eraser, transform=final)
    leaced_test = LeacedDataset(test, eraser, transform=final)

    # Accumulate items from the vanilla train dataset
    class_data = defaultdict(list)
    for x, y in leaced_train:
        class_data[y].append(x.flatten())
    train_items = {
        key: torch.stack(class_data[key]) 
        for key in class_data
    }
    covariances = {
        key: torch.matmul(train_items[key].T, train_items[key]) / len(train_items[key])
        for key in train_items
    }

    non_augmented_class_data = defaultdict(list)
    for x, y in non_augmented_leaced_train:
        non_augmented_class_data[y].append(x.flatten())
    non_augmented_train_items = {
        key: torch.stack(non_augmented_class_data[key])
        for key in non_augmented_class_data
    }
    non_augmented_covariances = {
        key: torch.matmul(non_augmented_train_items[key].T, non_augmented_train_items[key]) / len(non_augmented_train_items[key])
        for key in non_augmented_train_items
    }

    vanilla_class_data = defaultdict(list)
    for x, y in train:
        vanilla_class_data[y].append(x.flatten())
    vanilla_train_items = {
        key: torch.stack(vanilla_class_data[key]) 
        for key in vanilla_class_data
    }
    vanilla_covariances = {
        key: torch.matmul(vanilla_train_items[key].T, vanilla_train_items[key]) / len(vanilla_train_items[key])
        for key in vanilla_train_items
    }

    # Get the covariance matrix between each pair of edited classes
    non_aug_edited_means = []
    edited_means = []
    vanilla_means = []
    for i in range(10):
        for j in range(i + 1, 10):
            non_aug_cov_diff = non_augmented_covariances[i] - non_augmented_covariances[j]
            non_aug_norm = non_aug_cov_diff.norm(p="fro")
            non_aug_edited_means.append(non_aug_norm)

            cov_diff = covariances[i] - covariances[j]
            norm = cov_diff.norm(p="fro")
            edited_means.append(norm)

            vanilla_cov_diff = vanilla_covariances[i] - vanilla_covariances[j]
            vanilla_norm = vanilla_cov_diff.norm(p="fro")
            vanilla_means.append(vanilla_norm)

    print("Edited means", torch.tensor(edited_means).mean().item())
    print("max edited max", torch.tensor(edited_means).max().item())
    print("Vanilla means", torch.tensor(vanilla_means).mean().item())
    print("Non-augmented edited means", torch.tensor(non_aug_edited_means).mean().item())
    print("Non-augmented edited max", torch.tensor(non_aug_edited_means).max().item())
