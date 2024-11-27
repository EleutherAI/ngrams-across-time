from argparse import ArgumentParser
from pathlib import Path

import torch    
import torchvision as tv
from torch.utils.data import random_split
from torch import Tensor
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import (
    ConvNextV2Config, ConvNextV2ForImageClassification,
    Swinv2Config, Swinv2ForImageClassification,
    ViTConfig, ViTForImageClassification
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import lovely_tensors as lt
import plotly.io as pio
from datasets import Dataset as HfDataset
from sae.config import SaeConfig, TrainConfig
from sae.trainer import SaeTrainer
# from sae import SaeConfig, SaeTrainer, TrainConfig

from ngrams_across_time.utils.utils import set_seeds
from ngrams_across_time.clearnets.train.lightning_wrapper import ScheduleFreeLightningWrapper


torch.set_float32_matmul_precision('high')
pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469
lt.monkey_patch()
# set_seeds(598)


def vit_hookpoints(model):
    hookpoints = []
    for i in range(len(model.vit.encoder.layer)):
        hookpoints.append(f"model.vit.encoder.layer.{i}.attention.output")
        hookpoints.append(f"model.vit.encoder.layer.{i}.output")

    return hookpoints

def swin_hookpoints(model):
    hookpoints = []
    for i, stage in enumerate(model.swinv2.encoder.layers):
        for j in range(len(stage.blocks)):
            hookpoints.append(f"model.swinv2.encoder.layers.{i}.blocks.{j}.attention.output")

    return hookpoints

def convnext_hookpoints(model):
    hookpoints = []
    for i, stage in enumerate(model.convnextv2.encoder.stages):
        for j in range(len(stage.layers)):
            hookpoints.append(f"model.convnextv2.encoder.stages.{i}.layers.{j}.pwconv2")

    return hookpoints


def create_matched_models(image_size=32, num_labels=10):
    """
    Creates ConvNeXtV2, Swin, and ViT models with roughly matched parameter counts
    using Hugging Face Transformers library.
    Returns models and their parameter counts.
    """
    # ConvNeXtV2-Tiny configuration (~28M parameters)
    convnext_config = ConvNextV2Config(
        num_channels=3,
        num_labels=num_labels,
        depths=[3, 3, 9, 3],
        hidden_sizes=[96, 192, 384, 768],
        layer_norm_eps=1e-6,
        image_size=image_size
    )
    convnext_model = ConvNextV2ForImageClassification(convnext_config)
    
    # Swin-Tiny configuration (~28M parameters)
    swin_config = Swinv2Config(
        image_size=image_size,
        patch_size=4,
        num_channels=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        num_labels=num_labels
    )
    swin_model = Swinv2ForImageClassification(swin_config)
    
    # ~27M parameters
    vit_config = ViTConfig(
        image_size=image_size,
        patch_size=16,
        num_channels=3,
        hidden_size=432,  # Increased from 384
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=1728,  # 4 * hidden_size
        num_labels=num_labels
    )
    vit_model = ViTForImageClassification(vit_config)

    learning_rate = 1e-5
    betas = (0.95, 0.999)
    warmup_steps = 10_000

    models = {
        # 'ConvNeXtV2': ScheduleFreeLightningWrapper(convnext_model, learning_rate, betas, warmup_steps),
        'Swin': ScheduleFreeLightningWrapper(swin_model, learning_rate, betas, warmup_steps),
        # 'ViT': ScheduleFreeLightningWrapper(vit_model, learning_rate, betas, warmup_steps)
    }
    param_counts = {
        name: sum(p.numel() for p in model.parameters())
        for name, model in models.items()
    }

    hookpoints = {
        'ViT': vit_hookpoints(vit_model),
        'Swin': swin_hookpoints(swin_model),
        'ConvNeXtV2': convnext_hookpoints(convnext_model)
    }
    return {
        'models': models,
        'param_counts': param_counts,
        'configs': {
            'ConvNeXtV2': convnext_config,
            'Swin': swin_config,
            'ViT': vit_config
        },
        'hookpoints': hookpoints
    }


def get_cifar10() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, Tensor, Tensor]:
    nontest = CIFAR10("/home/lucia/cifar10", download=True)

    images, labels = zip(*nontest)
    X: Tensor = torch.stack(list(map(to_tensor, images)))
    Y = torch.tensor(labels)

    # Shuffle deterministically
    rng = torch.Generator(device=X.device).manual_seed(42)
    perm = torch.randperm(len(X), generator=rng, device=X.device)
    X, Y = X[perm], Y[perm]

    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024

    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    # Test set is entirely separate
    test = CIFAR10(root="/home/lucia/cifar10-test", train=False, download=True)
    test_images, test_labels = zip(*test)
    X_test: Tensor = torch.stack(list(map(to_tensor, test_images)))
    Y_test = torch.tensor(test_labels)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, k, X, Y


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sae", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def train():
    args = parse_args()

    # 598, then 42, then incorporated into run name
    set_seeds(args.seed)

    image_size = 32
    results = create_matched_models(image_size)
    
    print("\nParameter counts:")
    for model_name, count in results['param_counts'].items():
        print(f"{model_name}: {count:,} parameters")
        
    # Load CIFAR-10
    # X_train, Y_train, X_val, Y_val, X_test, Y_test, k, X, Y = get_cifar10()
    nontest = CIFAR10(
        "/home/lucia/cifar10", download=True, transform=tv.transforms.ToTensor()
    )
    train, val = random_split(nontest, [0.9, 0.1])
    test = CIFAR10(
        "/home/lucia/cifar10-test", download=True, transform=tv.transforms.ToTensor()
    )

    # Convert train to a HF Dataset
    sae_train = HfDataset.from_dict({
        "input_ids": torch.stack([x for x, _ in train]),
        "label": torch.tensor([y for _, y in train]),
    })
    sae_train.set_format(type="torch", columns=["input_ids", "label"])

    dm: pl.LightningDataModule = pl.LightningDataModule.from_datasets(
        train,
        val,
        test,
        batch_size=128, 
        num_workers=47,
    )

    for name, model in results['models'].items():
        # seed=38 models trained in 32bit, didn't affect perf
        wandb_name = f"{name}_s={args.seed}"
        ckpts_path = Path('arch-evals') / wandb_name
        if not ckpts_path.exists() or args.overwrite:
            trainer = pl.Trainer(
                devices=[1, 3],
                logger=WandbLogger(
                    name=wandb_name, project="arch-evals", entity="eleutherai", reinit=True
                ) if not args.debug else None,
                max_epochs=300,
                precision='bf16-mixed',
                deterministic=True,
                gradient_clip_val=None,
                callbacks=[
                    ModelCheckpoint(dirpath=ckpts_path, save_last=True, every_n_epochs=5, save_top_k=-1), 
                    EarlyStopping(monitor='val_loss', patience=30)
                ],
            )

            trainer.fit(model, dm)
            trainer.test(model, dm)

            try:
                wandb.finish()
            except Exception as e:
                print(e)


        if args.sae:
            assert results['hookpoints'][name] != [], "Hookpoints not set for model"
            cfg = TrainConfig(
                SaeConfig(multi_topk=True, expansion_factor=8),
                batch_size=8,
                run_name=str(wandb_name),
                log_to_wandb=True,
                hookpoints=results['hookpoints'][name],
                vision=name == 'ConvNeXtV2',
                grad_acc_steps=2,
                micro_acc_steps=2,
            )
            trainer = SaeTrainer(
                cfg, sae_train, model.cuda(), 
                dummy_inputs={'pixel_values': torch.randn(1, 3, 32, 32)}
            )

            trainer.fit()


def inference():
    image_size = 32
    results = create_matched_models(image_size)

    pass




if __name__ == "__main__":
    train()

    