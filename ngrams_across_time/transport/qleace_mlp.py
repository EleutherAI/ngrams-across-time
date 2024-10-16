from argparse import ArgumentParser
from itertools import pairwise
from typing import Callable, Sized

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm
import torchvision as tv
from concept_erasure import LeaceFitter, OracleFitter, QuadraticFitter, LeaceEraser
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor, nn
from torch.utils.data import Dataset, random_split, Subset
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

torch.set_float32_matmul_precision('high')


class Mlp(pl.LightningModule):
    def __init__(self, k, h=128):
        super().__init__()
        self.save_hyperparameters()

        self.build_net()
        self.train_acc = tm.Accuracy("multiclass", num_classes=k)
        self.val_acc = tm.Accuracy("multiclass", num_classes=k)
        self.test_acc = tm.Accuracy("multiclass", num_classes=k)
    
    def build_net(self):
        k, h = self.hparams['k'], self.hparams['h']
        self.net = torch.nn.Sequential(
            torch.nn.Linear(32 * 32 * 3, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, k),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
    
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)

        self.train_acc(y_hat, y)
        self.log(
            "train_acc", self.train_acc, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        self.val_acc(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        self.test_acc(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


class ResMlp(Mlp):    
    def build_net(self):
        sizes = [3 * 32 * 32] + [self.hparams['h']] + [self.hparams['k']]

        self.net = nn.Sequential(
            *[
                MlpBlock(in_dim, out_dim, device=self.device, dtype=self.dtype)
                for in_dim, out_dim in pairwise(sizes)
            ]
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        # ResNet initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    
    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
        )


class ResNet(Mlp):
    def build_net(self):
        self.net = tv.models.resnet18(pretrained=False, num_classes=self.hparams['k'])

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
        )


class ViT(Mlp):
    def build_net(self):
        self.net = tv.models.resnet18(pretrained=False, num_classes=self.hparams['k'])
    
    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
        )


class MlpBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        self.linear1 = nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(
            out_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.bn1 = nn.BatchNorm1d(out_features, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(out_features, device=device, dtype=dtype)
        self.downsample = (
            nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
            if in_features != out_features
            else None
        )

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.linear2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = nn.functional.relu(out)

        return out


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
            x_erased = self.eraser(x.flatten(), torch.tensor(z).type_as(x).to(torch.int64))
        return self.transform(x_erased.view(x.shape)), z 

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--eraser", type=str, choices=("none", "leace", "oleace", "qleace"))
    parser.add_argument("--net", type=str, choices=("mlp", "resmlp", "resnet"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if args.eraser == "qleace":
    #     device = torch.device("cpu")
    # Split the "train" set into train and validation
    nontest = CIFAR10(
        "/home/lucia/cifar10", download=True, transform=tv.transforms.ToTensor()
    )
    train, val = random_split(nontest, [0.9, 0.1])

    # Test set is entirely separate
    test = CIFAR10(
        "/home/lucia/cifar10-test",
        download=True,
        train=False,
        transform=tv.transforms.ToTensor(),
    )
    k = 10  # Number of classes
    final = nn.Identity() if args.net == "resnet" else nn.Flatten(0)
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

        fitter = cls(3 * 32 * 32, k, dtype=torch.float32, device=device)
        # train_subset = Subset(train, range(20_000))
        for x, y in tqdm(train):
            y = torch.as_tensor(y).view(1)
            if args.eraser != "qleace":
                y = F.one_hot(y, k)

            fitter.update(x.view(1, -1).to(device), y.to(device))

        eraser = fitter.eraser.to("cpu")
    else:
        eraser = lambda x, y: x
    train = LeacedDataset(train, eraser, transform=train_trf)
    val = LeacedDataset(val, eraser, transform=final)
    test = LeacedDataset(test, eraser, transform=final)

    # Create the data module
    dm = pl.LightningDataModule.from_datasets(train, val, test, batch_size=128, num_workers=8)

    model_cls = {
        "mlp": Mlp,
        "resmlp": ResMlp,
        "resnet": ResNet,
    }[args.net]
    model = model_cls(k)

    trainer = pl.Trainer(
        callbacks=[
            # EarlyStopping(monitor="val_loss", patience=5),
        ],
        logger=WandbLogger(name=args.name, project="mdl", entity="eleutherai"),
        max_epochs=200,
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)