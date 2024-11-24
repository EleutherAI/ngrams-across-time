# Sequential MLP probe can't learn LEACE
# Sequential MLP without pytorch lightning can learn LEACE
# Something in probe.fit

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchmetrics as tm

from argparse import ArgumentParser
from typing import Callable, Sized
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
from torchmetrics import Accuracy

import torch
import torch.nn.functional as F
import torchvision as tv
from concept_erasure import LeaceFitter, OracleEraser, OracleFitter, QuadraticFitter, LeaceEraser
from torch.utils.data import Subset
from torch import Tensor, nn
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm
import lovely_tensors as lt

lt.monkey_patch()

torch.set_default_tensor_type(torch.DoubleTensor)

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


# class Mlp(pl.LightningModule):
#     def __init__(self, k, h=128):
#         super().__init__()
#         self.k = k
#         self.h = h
#         self.build_net()
#         self.train_acc = Accuracy(task="multiclass", num_classes=k)
#         self.val_acc = Accuracy(task="multiclass", num_classes=k)
#         self.test_acc = Accuracy(task="multiclass", num_classes=k)
    
#     def build_net(self):
#         self.net = nn.Sequential(
#             nn.Linear(32 * 32 * 3, self.h),
#             nn.ReLU(),
#             nn.Linear(self.h, self.h),
#             nn.ReLU(),
#             nn.Linear(self.h, self.k),
#         )

#     def forward(self, x):
#         return self.net(x)

#     def training_step()


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        model.train_acc(output, target)
        
        if wandb.run is not None:
            wandb.log({"train_loss": loss.item()})
    
    epoch_loss = total_loss / len(train_loader)
    train_acc = model.train_acc.compute()
    model.train_acc.reset()
    
    if wandb.run is not None:
        wandb.log({"train_epoch_loss": epoch_loss, "train_acc": train_acc})
    return epoch_loss, train_acc

def evaluate(model, loader, device, mode='val'):
    model.eval()
    total_loss = 0
    acc_metric = model.val_acc if mode == 'val' else model.test_acc
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            acc_metric(output, target)
    
    avg_loss = total_loss / len(loader)
    acc = acc_metric.compute()
    acc_metric.reset()
    
    if wandb.run is not None:
        wandb.log({
            f"{mode}_loss": avg_loss,
            f"{mode}_acc": acc
        })
    return avg_loss, acc

def train_model(model, train_dataset, val_dataset, test_dataset, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if not args.debug:
        wandb.init(name=args.name, project="mdl", entity="eleutherai")
    
    batch_size = 128
    num_workers = 8
    train_loader, val_loader, test_loader = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )
    
    optimizer = torch.optim.AdamW(model.parameters())
    
    for epoch in range(200):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        val_loss, val_acc = evaluate(model, val_loader, device, mode='val')
        
        print(f'Epoch {epoch+1}/200')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    test_loss, test_acc = evaluate(model, test_loader, device, mode='test')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')


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
            x = x_erased.reshape(x.shape)
        elif isinstance(self.eraser, OracleEraser):
            x_erased = self.eraser(x.flatten(), torch.tensor(z).unsqueeze(0))
            x = x_erased.reshape(x.shape)
        else:
            z_tensor = torch.tensor(z)
            if z_tensor.ndim == 0:
                z_tensor = z_tensor.unsqueeze(0)
            x = self.eraser(x.unsqueeze(0), z_tensor)
        return self.transform(x), z

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--eraser", type=str, choices=("none", "leace", "oleace", "qleace"))
    parser.add_argument("--net", type=str, choices=("mlp", "resmlp", "resnet"))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

        fitter = cls(3 * 32 * 32, k, dtype=torch.float64, device=device, shrinkage=False)
        state_path = Path("erasers_cache") / f"cifar10_state.pth"
        state_path.parent.mkdir(exist_ok=True)
        state = {} if not state_path.exists() else torch.load(state_path)
        
        if args.eraser in state:
            eraser = state[args.eraser]
        else:
            for x, y in tqdm(train):
                y = torch.as_tensor(y).view(1)
                if args.eraser != "qleace":
                    y = F.one_hot(y, k)

                fitter.update(x.view(1, -1).to(device), y.to(device))

            eraser = fitter.eraser.to("cpu")

            state[args.eraser] = eraser
            torch.save(state, state_path)
    else:
        eraser = lambda x, y: x
    train = LeacedDataset(train, eraser, transform=train_trf)
    val = LeacedDataset(val, eraser, transform=final)
    test = LeacedDataset(test, eraser, transform=final)

    if args.debug:
        train = Subset(train, range(10_000))
    
    # Create the data module
    dm = pl.LightningDataModule.from_datasets(train, val, test, batch_size=128, num_workers=8)
    
    model_cls = {
        "mlp": Mlp,
        # "resmlp": ResMlp,
        # "resnet": ResNet,
    }[args.net]
    model = model_cls(k, h=args.width)
    train_model(model, train, val, test, args)

    # trainer = pl.Trainer(
    #     logger=WandbLogger(name=args.name, project="mdl", entity="eleutherai") if not args.debug else None,
    #     max_epochs=200,
    #     precision=64,
    #     deterministic=True,
    #     gradient_clip_val=None,
    # )
    # trainer.fit(model, dm)
    # trainer.test(model, dm)