import os
import copy
from pathlib import Path
from copy import deepcopy

import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import einops
import tqdm.auto as tqdm
import plotly.express as px
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, HookedTransformerConfig
import plotly.express as px
import plotly.io as pio

from auto_circuit.data import PromptDataset, PromptDataLoader
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores, AblationType
from auto_circuit.prune import patch_mode, run_circuits

from ngrams_across_time.utils.utils import assert_type

pio.templates['plotly'].layout.xaxis.title.font.size = 20 # type: ignore
pio.templates['plotly'].layout.yaxis.title.font.size = 20 # type: ignore
pio.templates['plotly'].layout.title.font.size = 30 # type: ignore

TRAIN_MODEL = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""Plotting helper functions:"""
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)


from plotly.graph_objects import Figure
def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs) -> Figure:
    fig = px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs)
    fig.show(renderer)
    return fig

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

def main():
    # Define the location to save the model, using a relative path
    PTH_LOCATION = "workspace/_scratch/grokking_demo.pth"
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

    cfg = HookedTransformerConfig(
        n_layers = 1,
        n_heads = 4,
        d_model = 128,
        d_head = 32,
        d_mlp = 512,
        act_fn = "relu",
        normalization_type=None,
        d_vocab=p+1,
        d_vocab_out=p,
        n_ctx=3,
        init_weights=True,
        device=device.type,
        seed = 999,
    )

    # breakpoint()
    model = HookedTransformer(cfg)

    """Disable the biases, as we don't need them for this task and it makes things easier to interpret."""

    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    """## Define Optimizer + Loss"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    def loss_fn(logits, labels):
        if len(logits.shape)==3:
            logits = logits[:, -1]
        logits = logits.to(torch.float64)
        log_probs = logits.log_softmax(dim=-1)
        correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
        return -correct_log_probs.mean()
    train_logits = model(train_data)
    train_loss = loss_fn(train_logits, train_labels)
    print(train_loss)
    test_logits = model(test_data)
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
    checkpoint_ablation_loss_increases = []
    if TRAIN_MODEL:
        for epoch in tqdm.tqdm(range(num_epochs)):
            train_logits = model(train_data)
            train_loss = loss_fn(train_logits, train_labels)
            train_loss.backward()
            train_losses.append(train_loss.item())

            optimizer.step()
            optimizer.zero_grad()

            with torch.inference_mode():
                test_logits = model(test_data)
                test_loss = loss_fn(test_logits, test_labels)
                test_losses.append(test_loss.item())

            if ((epoch + 1) % checkpoint_every) == 0:
                checkpoint_epochs.append(epoch)
                model_checkpoints.append(copy.deepcopy(model.state_dict()))
                print(f"Epoch {epoch} Train Loss {train_loss.item()} Test Loss {test_loss.item()}")

                if (epoch + 1) % (checkpoint_every * 10) == 0:
                    # Add patching configuration
                    model.set_use_attn_result(True)
                    model.set_use_hook_mlp_in(True)
                    model.set_use_split_qkv_input(True)

                    # Patch using earlier checkpoint or alternative prompt
                    # answers = [batch.unsqueeze(dim=0) for batch in list(train_labels.unbind())]

                    # patching_ds = PromptDataset(train_data[:-1], train_data[1:], answers[:-1], answers[1:])
                    # dataloader = PromptDataLoader(patching_ds, seq_len=train_data.shape[-1], diverge_idx=0)
                    print("Calculating EAP prune scores for patching corrupt n-grams into learned n-gram")
                    ablation_model = patchable_model(deepcopy(model), factorized=True, device=device, separate_qkv=True, seq_len=train_data.shape[-1], slice_output="last_seq")
                    
                    # Remove patching configuration
                    model.set_use_attn_result(False)
                    model.set_use_hook_mlp_in(False)
                    model.set_use_split_qkv_input(False)

                    # Count the loss from ablating the top n edges
                    # todo run this once for each batch
                    # edge_prune_scores: PruneScores = mask_gradient_prune_scores(model=ablation_model, dataloader=dataloader,official_edges=set(),grad_function="logit",answer_function="avg_diff",mask_val=0.0)
                    # num_edges = 10
                    # logits = run_circuits(ablation_model, dataloader, [num_edges], prune_scores=edge_prune_scores, ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT)
                    # batch_logits = list(logits.values())[0] # this key will be edges - num_edges
                    # batch_logits = assert_type(dict, batch_logits)

                    loss_increases = []
                    answers = [batch.unsqueeze(dim=0) for batch in list(train_labels.unbind())]

                    patching_ds = PromptDataset(train_data[:-1], train_data[1:], answers[:-1], answers[1:])
                    dataloader = PromptDataLoader(patching_ds, seq_len=train_data.shape[-1], diverge_idx=0)
                    for batch in dataloader: 
                        edge_prune_scores: PruneScores = mask_gradient_prune_scores(model=ablation_model, dataloader=dataloader,official_edges=set(),grad_function="logit",answer_function="avg_diff",mask_val=0.0)
                        num_edges = 10
                        logits = run_circuits(ablation_model, dataloader, [num_edges], prune_scores=edge_prune_scores, ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT)
                        batch_logits = list(logits.values())[0] # this key will be edges - num_edges
                        batch_logits = assert_type(dict, batch_logits)

                        patched_logits = batch_logits[batch.key]
                        unpatched_logits = model(batch.clean)[:, -1, :]
                        patched_loss = F.cross_entropy(patched_logits.to(device).squeeze(), batch.answers.to(device).squeeze())
                        unpatched_loss = F.cross_entropy(unpatched_logits.squeeze(), batch.answers.to(device).squeeze())
                        loss_increases.append((patched_loss - unpatched_loss).item())

                    checkpoint_ablation_loss_increases.append(np.mean(loss_increases))                
                    print(checkpoint_ablation_loss_increases[-1])


        torch.save(
            {
                "model":model.state_dict(),
                "config": model.cfg,
                "checkpoints": model_checkpoints,
                "checkpoint_epochs": checkpoint_epochs,
                "test_losses": test_losses,
                "train_losses": train_losses,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "ablation_loss_increases": checkpoint_ablation_loss_increases
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

    # Commented out IPython magic to ensure Python compatibility.
    # %pip install git+https://github.com/neelnanda-io/neel-plotly.git
    # TODO use a normal line plot
    # Create the line plot
    epochs = np.arange(0, len(train_losses), 100)
    df = pd.DataFrame({
        'Epoch': np.concatenate([epochs, epochs, epochs[0 : len(epochs) : 10]]),
        'Loss': np.array([train_losses[0 : len(train_losses) : 100] + test_losses[0 : len(test_losses) : 100] + ablation_loss_increases]).squeeze(),
        'Type': np.array(['Train'] * len(epochs) + ['Test'] * len(epochs) + ['TrainAblation'] * int(len(epochs) // 10)).squeeze()
    })

    df = pd.DataFrame({'Epoch': np.concatenate([epochs, epochs, epochs[0 : len(epochs) : 10]]),'Loss': np.array([train_losses[0 : len(train_losses) : 100] + test_losses[0 : len(test_losses) : 100] + ablation_loss_increases]).squeeze(),'Type': np.array(['Train'] * len(epochs) + ['Test'] * len(epochs) + ['TrainAblation'] * int(len(epochs) // 10)).squeeze()})

    fig = px.line(df, x='Epoch', y='Loss', color='Type', title='Training Curve for Modular Addition',log_y=True)

    # Customize the layout
    fig.update_layout(
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend_title='',
        hovermode='x unified'
    )

    memorization_end_epoch = 1500
    circuit_formation_end_epoch = 13300
    cleanup_end_epoch = 16600

    def add_lines(figure):
        figure.add_vline(memorization_end_epoch, line_dash="dash", opacity=0.7)
        figure.add_vline(circuit_formation_end_epoch, line_dash="dash", opacity=0.7)
        figure.add_vline(cleanup_end_epoch, line_dash="dash", opacity=0.7)
        return figure

    fig = add_lines(fig)
    fig.write_image("loss_curves.pdf", format="pdf")


    def test_logits(logits, bias_correction=False, original_logits=None, mode="all"):
        """Code to test a tensor of edited logits"""
        # Calculates cross entropy loss of logits representing a batch of all p^2
        # possible inputs
        # Batch dimension is assumed to be first
        if logits.shape[1] == p * p:
            logits = logits.T
        if logits.shape == torch.Size([p * p, p + 1]):
            logits = logits[:, :-1]
        logits = logits.reshape(p * p, p)
        if bias_correction:
            # Applies bias correction - we correct for any missing bias terms,
            # independent of the input, by centering the new logits along the batch
            # dimension, and then adding the average original logits across all inputs
            logits = (
                einops.reduce(original_logits - logits, "batch ... -> ...", "mean") + logits
            )
        if mode == "train":
            return loss_fn(logits[train_indices], labels[train_indices])
        elif mode == "test":
            return loss_fn(logits[test_indices], labels[test_indices])
        elif mode == "all":
            return loss_fn(logits, labels)

    """Code to run a metric over every checkpoint"""

    metric_cache = {}
    def get_metrics(model, metric_cache, metric_fn, name, reset=False):
        if reset or (name not in metric_cache) or (len(metric_cache[name]) == 0):
            metric_cache[name] = []
            for c, sd in enumerate(tqdm.tqdm((model_checkpoints))):
                model.reset_hooks()
                model.load_state_dict(sd)
                out = metric_fn(model)
                if type(out) == Tensor:
                    out = utils.to_numpy(out)
                metric_cache[name].append(out)
            model.load_state_dict(model_checkpoints[-1])
            try:
                metric_cache[name] = torch.tensor(metric_cache[name])
            except:
                metric_cache[name] = torch.tensor(np.array(metric_cache[name]))



if __name__ == '__main__':
    main()