import os
import copy
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import torch
import numpy as np
import einops
import tqdm.auto as tqdm
import plotly.express as px
import plotly.express as px
import plotly.io as pio
from ngrams_across_time.grok.transformers import CustomTransformer, TransformerConfig
import torch.nn.functional as F
import lovely_tensors as lt
lt.monkey_patch()

pio.templates['plotly'].layout.xaxis.title.font.size = 20 # type: ignore
pio.templates['plotly'].layout.yaxis.title.font.size = 20 # type: ignore
pio.templates['plotly'].layout.title.font.size = 30 # type: ignore

TRAIN_MODEL = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(MODEL_SEED = 999):
    # Define the location to save the model, using a relative path
    PTH_LOCATION = f"workspace/grok/{MODEL_SEED}.pth"
    os.makedirs(Path(PTH_LOCATION).parent, exist_ok=True)

    """# Model Training

    ## Config
    """
    p = 113
    frac_train = 0.3

    # Optimizer config
    lr = 1e-3
    wd = 1.
    betas = (0.9, 0.98)

    num_epochs = 25000
    checkpoint_every = 100

    DATA_SEED = 598

    """## Define Task
    * Define modular addition
    * Define the dataset & labels

    Input format:
    |a|b|=|
    """

    a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
    b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
    equals_vector = einops.repeat(torch.tensor(113), " -> (i j)", i=p, j=p)

    dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)

    labels = (dataset[:, 0] + dataset[:, 1]) % p

    """Convert this to a train + test set - 30% in the training set"""

    torch.manual_seed(DATA_SEED)
    indices = torch.randperm(p*p)
    cutoff = int(p*p*frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    train_data = dataset[train_indices]
    train_labels = labels[train_indices]
    test_data = dataset[test_indices]
    test_labels = labels[test_indices]

    """## Define Model"""
    # n_ctx
    # 5 yields drop around 13k
    # 4 yields fast drop around 8k
    # 3 yields fast drop around 5k
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(MODEL_SEED)
    import random; random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    config = TransformerConfig(
        d_vocab=p + 1,
        d_model=128,
        n_ctx=5, # dummy inputs hardcoded in huggingface transformers expect > 4
        num_layers=1,
        num_heads=4,
        d_mlp=512,
        act_type="ReLU",
        use_ln=False,
    )
    model = CustomTransformer(config)

    model.to(device)

    """Disable the biases, as we don't need them for this task and it makes things easier to interpret."""

    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    """## Define Optimizer + Loss"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    def loss_fn(logits, labels):
        if len(logits.shape) == 3:
            logits = logits[:, -1]
        logits = logits.to(torch.float64)
        return F.cross_entropy(logits, labels).mean()
    train_logits = model(train_data).logits[..., :-1]
    # train_logits = model(train_data)
    train_loss = loss_fn(train_logits, train_labels)
    print(train_loss)
    test_logits = model(test_data).logits[..., :-1]
    # test_logits = model(test_data)
    test_loss = loss_fn(test_logits, test_labels)
    print(test_loss)

    print("Uniform loss:")
    print(np.log(p))

    """## Actually Train

    **Weird Decision:** Training the model with full batch training rather than stochastic gradient descent. We do this so to make training smoother and reduce the number of slingshots.
    """

    train_losses = []
    test_losses = []
    model_checkpoints = []
    checkpoint_epochs = []

    ablation_loss_increases = []
    if TRAIN_MODEL:
        for epoch in tqdm.tqdm(range(num_epochs)):
            # train_logits = model(train_data)
            train_logits = model(train_data).logits[..., :-1]
            train_loss = loss_fn(train_logits, train_labels)
            train_loss.backward()
            train_losses.append(train_loss.item())

            optimizer.step()
            optimizer.zero_grad()

            with torch.inference_mode():
                # test_logits = model(test_data)
                test_logits = model(test_data).logits[..., :-1]
                test_loss = loss_fn(test_logits, test_labels)
                test_losses.append(test_loss.item())

            if ((epoch + 1) % checkpoint_every) == 0:
                checkpoint_epochs.append(epoch)
                model_checkpoints.append(copy.deepcopy(model.state_dict()))
                print(f"Epoch {epoch} Train Loss {train_loss.item()} Test Loss {test_loss.item()}")

        torch.save(
            {
                "model": model.state_dict(),
                "dataset": dataset,
                "labels": labels,
                "config": config,
                "checkpoints": model_checkpoints,
                "checkpoint_epochs": checkpoint_epochs,
                "test_losses": test_losses,
                "train_losses": train_losses,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "ablation_loss_increases": ablation_loss_increases
            },
            PTH_LOCATION)
    if not TRAIN_MODEL:
        cached_data = torch.load(PTH_LOCATION)
        model.load_state_dict(cached_data['model'])
        model_checkpoints = cached_data["checkpoints"]
        checkpoint_epochs = cached_data["checkpoint_epochs"]
        test_losses = cached_data['test_losses']
        train_losses = cached_data['train_losses']
        train_indices = cached_data["train_indices"]
        test_indices = cached_data["test_indices"]
        ablation_loss_increases = cached_data["ablation_loss_increases"]

    """## Show Model Training Statistics, Check that it groks!"""

    print(len(test_losses))
    print(len(train_losses))

    epochs = np.arange(0, len(train_losses), 100)
    df = pd.DataFrame({
        'Epoch': np.concatenate([epochs, epochs]),
        'Loss': np.array([train_losses[0 : len(train_losses) : 100] + test_losses[0 : len(test_losses) : 100]]).squeeze(),
        'Type': np.array(['Train'] * len(epochs) + ['Test'] * len(epochs))
        }
    )

    fig = px.line(
        df, 
        x='Epoch', y='Loss', color='Type', title='Training Curve for Modular Addition', 
        log_y=True
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend_title='',
        hovermode='x unified'
    )

    memorization_end_epoch = 1500
    circuit_formation_end_epoch = 12500
    cleanup_end_epoch = 18_500

    def add_lines(figure):
        figure.add_vline(memorization_end_epoch, line_dash="dash", opacity=0.7)
        figure.add_vline(circuit_formation_end_epoch, line_dash="dash", opacity=0.7)
        figure.add_vline(cleanup_end_epoch, line_dash="dash", opacity=0.7)
        return figure

    fig = add_lines(fig)
    fig.write_image(f"loss_curves-{MODEL_SEED}.pdf", format="pdf")

def parse_args():
    parser = ArgumentParser(description="Train the model")
    parser.add_argument("--model_seed", type=int, default=999, help="Seed for the model")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.model_seed)