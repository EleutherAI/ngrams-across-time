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
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, HookedTransformerConfig
import plotly.express as px
import plotly.io as pio

from auto_circuit.data import PromptDataset, PromptDataLoader
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores, AblationType #, OutputSlice
from auto_circuit.prune import patch_mode, run_circuits

from src.utils.utils import assert_type

pio.templates['plotly'].layout.xaxis.title.font.size = 20 # type: ignore
pio.templates['plotly'].layout.yaxis.title.font.size = 20 # type: ignore
pio.templates['plotly'].layout.title.font.size = 30 # type: ignore

TRAIN_MODEL = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""Plotting helper functions:"""
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

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
    print(dataset[:5])
    print(dataset.shape)

    labels = (dataset[:, 0] + dataset[:, 1]) % p
    print(labels.shape)
    print(labels[:5])

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
    print(train_data[:5])
    print(train_labels[:5])
    print(train_data.shape)
    print(test_data[:5])
    print(test_labels[:5])
    print(test_data.shape)

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
    # from neel_plotly.plot import line
    # line([train_losses[::100], test_losses[::100]], x=np.arange(0, len(train_losses), 100), xaxis="Epoch", yaxis="Loss", log_y=True, title="Training Curve for Modular Addition", line_labels=['train', 'test'], toggle_x=True, toggle_y=True)
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
    fig.write_image("loss_curves.pdf", format="pdf")


    """# Analysing the Model

    ## Standard Things to Try
    """

    original_logits, cache = model.run_with_cache(dataset)
    original_logits = assert_type(Tensor, original_logits)
    print(original_logits.numel())

    """Get key weight matrices:"""

    W_E = model.embed.W_E[:-1]
    print("W_E", W_E.shape)
    W_neur = W_E @ model.blocks[0].attn.W_V @ model.blocks[0].attn.W_O @ model.blocks[0].mlp.W_in
    print("W_neur", W_neur.shape)
    W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U
    print("W_logit", W_logit.shape)

    original_loss = loss_fn(original_logits, labels).item()
    print("Original Loss:", original_loss)

    """### Looking at Activations

    Helper variable:
    """

    pattern_a = cache["pattern", 0, "attn"][:, :, -1, 0]
    pattern_b = cache["pattern", 0, "attn"][:, :, -1, 1]
    neuron_acts = cache["post", 0, "mlp"][:, -1, :]
    neuron_pre_acts = cache["pre", 0, "mlp"][:, -1, :]

    """Get all shapes:"""

    for param_name, param in cache.items():
        print(param_name, param.shape)

    imshow(cache["pattern", 0].mean(dim=0)[:, -1, :], title="Average Attention Pattern per Head", xaxis="Source", yaxis="Head", x=['a', 'b', '='])

    imshow(cache["pattern", 0][5][:, -1, :], title="Average Attention Pattern per Head", xaxis="Source", yaxis="Head", x=['a', 'b', '='])

    dataset[:4]

    imshow(cache["pattern", 0][:, 0, -1, 0].reshape(p, p), title="Attention for Head 0 from a -> =", xaxis="b", yaxis="a")

    imshow(
        einops.rearrange(cache["pattern", 0][:, :, -1, 0], "(a b) head -> head a b", a=p, b=p),
        title="Attention for Head 0 from a -> =", xaxis="b", yaxis="a", facet_col=0)

    """Plotting neuron activations"""

    cache["post", 0, "mlp"].shape

    imshow(
        einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=p, b=p),
        title="First 5 neuron acts", xaxis="b", yaxis="a", facet_col=0)

    """### Singular Value Decomposition"""

    W_E.shape

    U, S, Vh = torch.svd(W_E)
    line(S, title="Singular Values")
    imshow(U, title="Principal Components on the Input")

    # Control - random Gaussian matrix
    U, S, Vh = torch.svd(torch.randn_like(W_E))
    line(S, title="Singular Values Random")
    imshow(U, title="Principal Components Random")

    """## Explaining Algorithm

    ### Analyse the Embedding - It's a Lookup Table!
    """

    U, S, Vh = torch.svd(W_E)
    line(U[:, :8].T, title="Principal Components of the embedding", xaxis="Input Vocabulary")

    fourier_basis = []
    fourier_basis_names = []
    fourier_basis.append(torch.ones(p))
    fourier_basis_names.append("Constant")
    for freq in range(1, p//2+1):
        fourier_basis.append(torch.sin(torch.arange(p)*2 * torch.pi * freq / p))
        fourier_basis_names.append(f"Sin {freq}")
        fourier_basis.append(torch.cos(torch.arange(p)*2 * torch.pi * freq / p))
        fourier_basis_names.append(f"Cos {freq}")
    fourier_basis = torch.stack(fourier_basis, dim=0).to(device)
    fourier_basis = fourier_basis/fourier_basis.norm(dim=-1, keepdim=True)
    imshow(fourier_basis, xaxis="Input", yaxis="Component", y=fourier_basis_names)

    line(fourier_basis[:8], xaxis="Input", line_labels=fourier_basis_names[:8], title="First 8 Fourier Components")
    line(fourier_basis[25:29], xaxis="Input", line_labels=fourier_basis_names[25:29], title="Middle Fourier Components")

    imshow(fourier_basis @ fourier_basis.T, title="All Fourier Vectors are Orthogonal")

    """### Analyse the Embedding"""

    imshow(fourier_basis @ W_E, yaxis="Fourier Component", xaxis="Residual Stream", y=fourier_basis_names, title="Embedding in Fourier Basis")

    line((fourier_basis @ W_E).norm(dim=-1), xaxis="Fourier Component", x=fourier_basis_names, title="Norms of Embedding in Fourier Basis")

    key_freqs = [17, 25, 32, 47]
    key_freq_indices = [33, 34, 49, 50, 63, 64, 93, 94]
    fourier_embed = fourier_basis @ W_E
    key_fourier_embed = fourier_embed[key_freq_indices]
    print("key_fourier_embed", key_fourier_embed.shape)
    imshow(key_fourier_embed @ key_fourier_embed.T, title="Dot Product of embedding of key Fourier Terms")

    """### Key Frequencies"""

    line(fourier_basis[[34, 50, 64, 94]], title="Cos of key freqs", line_labels=[34, 50, 64, 94])

    line(fourier_basis[[34, 50, 64, 94]].mean(0), title="Constructive Interference")

    """## Analyse Neurons"""

    imshow(
        einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=p, b=p),
        title="First 5 neuron acts", xaxis="b", yaxis="a", facet_col=0)

    imshow(
        einops.rearrange(neuron_acts[:, 0], "(a b) -> a b", a=p, b=p),
        title="First neuron act", xaxis="b", yaxis="a",)

    imshow(fourier_basis[94][None, :] * fourier_basis[94][:, None], title="Cos 47a * cos 47b")

    imshow(fourier_basis[94][None, :] * fourier_basis[0][:, None], title="Cos 47a * const")

    imshow(fourier_basis @ neuron_acts[:, 0].reshape(p, p) @ fourier_basis.T, title="2D Fourier Transformer of neuron 0", xaxis="b", yaxis="a", x=fourier_basis_names, y=fourier_basis_names)

    imshow(fourier_basis @ neuron_acts[:, 5].reshape(p, p) @ fourier_basis.T, title="2D Fourier Transformer of neuron 5", xaxis="b", yaxis="a", x=fourier_basis_names, y=fourier_basis_names)

    imshow(fourier_basis @ torch.randn_like(neuron_acts[:, 0]).reshape(p, p) @ fourier_basis.T, title="2D Fourier Transformer of RANDOM", xaxis="b", yaxis="a", x=fourier_basis_names, y=fourier_basis_names)

    """### Neuron Clusters"""

    fourier_neuron_acts = fourier_basis @ einops.rearrange(neuron_acts, "(a b) neuron -> neuron a b", a=p, b=p) @ fourier_basis.T
    # Center these by removing the mean - doesn't matter!
    fourier_neuron_acts[:, 0, 0] = 0.
    print("fourier_neuron_acts", fourier_neuron_acts.shape)

    neuron_freq_norm = torch.zeros(p//2, model.cfg.d_mlp).to(device)
    for freq in range(0, p//2):
        for x in [0, 2*(freq+1) - 1, 2*(freq+1)]:
            for y in [0, 2*(freq+1) - 1, 2*(freq+1)]:
                neuron_freq_norm[freq] += fourier_neuron_acts[:, x, y]**2
    neuron_freq_norm = neuron_freq_norm / fourier_neuron_acts.pow(2).sum(dim=[-1, -2])[None, :]
    imshow(neuron_freq_norm, xaxis="Neuron", yaxis="Freq", y=torch.arange(1, p//2+1), title="Neuron Frac Explained by Freq")

    line(neuron_freq_norm.max(dim=0).values.sort().values, xaxis="Neuron", title="Max Neuron Frac Explained over Freqs")

    """## Read Off the Neuron-Logit Weights to Interpret"""

    W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U
    print("W_logit", W_logit.shape)

    line((W_logit @ fourier_basis.T).norm(dim=0), x=fourier_basis_names, title="W_logit in the Fourier Basis")

    neurons_17 = neuron_freq_norm[17-1]>0.85
    neurons_17.shape

    neurons_17.sum()

    line((W_logit[neurons_17] @ fourier_basis.T).norm(dim=0), x=fourier_basis_names, title="W_logit for freq 17 neurons in the Fourier Basis")

    """Study sin 17"""

    freq = 17
    W_logit_fourier = W_logit @ fourier_basis
    neurons_sin_17 = W_logit_fourier[:, 2*freq-1]
    line(neurons_sin_17)

    neuron_acts.shape

    inputs_sin_17c = neuron_acts @ neurons_sin_17
    imshow(fourier_basis @ inputs_sin_17c.reshape(p, p) @ fourier_basis.T, title="Fourier Heatmap over inputs for sin17c", x=fourier_basis_names, y=fourier_basis_names)

    """# Black Box Methods + Progress Measures

    ## Setup Code

    Code to plot embedding freqs
    """

    def embed_to_cos_sin(fourier_embed):
        if len(fourier_embed.shape) == 1:
            return torch.stack([fourier_embed[1::2], fourier_embed[2::2]])
        else:
            return torch.stack([fourier_embed[:, 1::2], fourier_embed[:, 2::2]], dim=1)

    # from neel_plotly.plot import melt

    # def plot_embed_bars(
    #     fourier_embed,
    #     title="Norm of embedding of each Fourier Component",
    #     return_fig=False,
    #     **kwargs
    # ):
    #     cos_sin_embed = embed_to_cos_sin(fourier_embed)
    #     df = melt(cos_sin_embed)
    #     # display(df)
    #     group_labels = {0: "sin", 1: "cos"}
    #     df["Trig"] = df["0"].map(lambda x: group_labels[x])
    #     fig = px.bar(
    #         df,
    #         barmode="group",
    #         color="Trig",
    #         x="1",
    #         y="value",
    #         labels={"1": "$w_k$", "value": "Norm"},
    #         title=title,
    #         **kwargs
    #     )
    #     fig.update_layout(dict(legend_title=""))

    #     if return_fig:
    #         return fig
    #     else:
    #         fig.show()

    """Code to test a tensor of edited logits"""

    def test_logits(logits, bias_correction=False, original_logits=None, mode="all"):
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

    """## Defining Progress Measures

    ### Loss Curves
    """

    memorization_end_epoch = 1500
    circuit_formation_end_epoch = 13300
    cleanup_end_epoch = 16600

    def add_lines(figure):
        figure.add_vline(memorization_end_epoch, line_dash="dash", opacity=0.7)
        figure.add_vline(circuit_formation_end_epoch, line_dash="dash", opacity=0.7)
        figure.add_vline(cleanup_end_epoch, line_dash="dash", opacity=0.7)
        return figure

    fig = line([train_losses[::100], test_losses[::100]], x=np.arange(0, len(train_losses), 100), xaxis="Epoch", yaxis="Loss", log_y=True, title="Training Curve for Modular Addition", line_labels=['train', 'test'], toggle_x=True, toggle_y=True, return_fig=True)
    add_lines(fig)

    """### Logit Periodicity"""
    all_logits = original_logits[:, -1, :]
    print(all_logits.shape)
    all_logits = einops.rearrange(all_logits, "(a b) c -> a b c", a=p, b=p)
    print(all_logits.shape)

    coses = {}
    for freq in key_freqs:
        print("Freq:", freq)
        a = torch.arange(p)[:, None, None]
        b = torch.arange(p)[None, :, None]
        c = torch.arange(p)[None, None, :]
        cube_predicted_logits = torch.cos(freq * 2 * torch.pi / p * (a + b - c)).to(device)
        cube_predicted_logits /= cube_predicted_logits.norm()
        coses[freq] = cube_predicted_logits

    approximated_logits = torch.zeros_like(all_logits)
    for freq in key_freqs:
        print("Freq:", freq)
        coeff = (all_logits * coses[freq]).sum()
        print("Coeff:", coeff)
        cosine_sim = coeff / all_logits.norm()
        print("Cosine Sim:", cosine_sim)
        approximated_logits += coeff * coses[freq]
    residual = all_logits - approximated_logits
    print("Residual size:", residual.norm())
    print("Residual fraction of norm:", residual.norm()/all_logits.norm())

    random_logit_cube = torch.randn_like(all_logits)
    print((all_logits * random_logit_cube).sum()/random_logit_cube.norm()/all_logits.norm())

    test_logits(all_logits)

    test_logits(approximated_logits)

    """#### Look During Training"""

    cos_cube = []
    for freq in range(1, p//2 + 1):
        a = torch.arange(p)[:, None, None]
        b = torch.arange(p)[None, :, None]
        c = torch.arange(p)[None, None, :]
        cube_predicted_logits = torch.cos(freq * 2 * torch.pi / p * (a + b - c)).to(device)
        cube_predicted_logits /= cube_predicted_logits.norm()
        cos_cube.append(cube_predicted_logits)
    cos_cube = torch.stack(cos_cube, dim=0)
    print(cos_cube.shape)

    def get_cos_coeffs(model):
        logits = model(dataset)[:, -1]
        logits = einops.rearrange(logits, "(a b) c -> a b c", a=p, b=p)
        vals = (cos_cube * logits[None, :, :, :]).sum([-3, -2, -1])
        return vals


    get_metrics(model, metric_cache, get_cos_coeffs, "cos_coeffs")
    print(metric_cache["cos_coeffs"].shape)

    fig = line(metric_cache["cos_coeffs"].T, line_labels=[f"Freq {i}" for i in range(1, p//2+1)], title="Coefficients with Predicted Logits", xaxis="Epoch", x=checkpoint_epochs, yaxis="Coefficient", return_fig=True)
    add_lines(fig)

    def get_cos_sim(model):
        logits = model(dataset)[:, -1]
        logits = einops.rearrange(logits, "(a b) c -> a b c", a=p, b=p)
        vals = (cos_cube * logits[None, :, :, :]).sum([-3, -2, -1])
        return vals / logits.norm()

    get_metrics(model, metric_cache, get_cos_sim, "cos_sim") # You may need a big GPU. If you don't have one and can't work around this, raise an issue for help!
    print(metric_cache["cos_sim"].shape)

    fig = line(metric_cache["cos_sim"].T, line_labels=[f"Freq {i}" for i in range(1, p//2+1)], title="Cosine Sim with Predicted Logits", xaxis="Epoch", x=checkpoint_epochs, yaxis="Cosine Sim", return_fig=True)
    add_lines(fig)

    def get_residual_cos_sim(model):
        logits = model(dataset)[:, -1]
        logits = einops.rearrange(logits, "(a b) c -> a b c", a=p, b=p)
        vals = (cos_cube * logits[None, :, :, :]).sum([-3, -2, -1])
        residual = logits - (vals[:, None, None, None] * cos_cube).sum(dim=0)
        return residual.norm() / logits.norm()

    get_metrics(model, metric_cache, get_residual_cos_sim, "residual_cos_sim")
    print(metric_cache["residual_cos_sim"].shape)

    fig = line([metric_cache["cos_sim"][:, i] for i in range(p//2)]+[metric_cache["residual_cos_sim"]], line_labels=[f"Freq {i}" for i in range(1, p//2+1)]+["residual"], title="Cosine Sim with Predicted Logits + Residual", xaxis="Epoch", x=checkpoint_epochs, yaxis="Cosine Sim", return_fig=True)
    add_lines(fig)

    """## Restricted Loss"""

    neuron_acts.shape

    neuron_acts_square = einops.rearrange(neuron_acts, "(a b) neur -> a b neur", a=p, b=p).clone()
    # Center it
    neuron_acts_square -= einops.reduce(neuron_acts_square, "a b neur -> 1 1 neur", "mean")
    neuron_acts_square_fourier = einsum("a b neur, fa a, fb b -> fa fb neur", neuron_acts_square, fourier_basis, fourier_basis)
    imshow(neuron_acts_square_fourier.norm(dim=-1), xaxis="Fourier Component b", yaxis="Fourier Component a", title="Norms of neuron activations by Fourier Component", x=fourier_basis_names, y=fourier_basis_names)

    original_logits, cache = model.run_with_cache(dataset)
    original_logits = assert_type(Tensor, original_logits)
    print(original_logits.numel())
    neuron_acts = cache["post", 0, "mlp"][:, -1, :]

    approx_neuron_acts = torch.zeros_like(neuron_acts)
    approx_neuron_acts += neuron_acts.mean(dim=0)
    a = torch.arange(p)[:, None]
    b = torch.arange(p)[None, :]
    for freq in key_freqs:
        cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
        cos_apb_vec /= cos_apb_vec.norm()
        cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
        sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
        sin_apb_vec /= sin_apb_vec.norm()
        sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
    restricted_logits = approx_neuron_acts @ W_logit
    print(loss_fn(restricted_logits[test_indices], test_labels))

    print(loss_fn(all_logits, labels)) # This bugged on models not fully trained

    """### Look During Training"""

    def get_restricted_loss(model):
        logits, cache = model.run_with_cache(dataset)
        logits = logits[:, -1, :]
        neuron_acts = cache["post", 0, "mlp"][:, -1, :]
        approx_neuron_acts = torch.zeros_like(neuron_acts)
        approx_neuron_acts += neuron_acts.mean(dim=0)
        a = torch.arange(p)[:, None]
        b = torch.arange(p)[None, :]
        for freq in key_freqs:
            cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
            cos_apb_vec /= cos_apb_vec.norm()
            cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
            approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
            sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
            sin_apb_vec /= sin_apb_vec.norm()
            sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
            approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
        restricted_logits = approx_neuron_acts @ model.blocks[0].mlp.W_out @ model.unembed.W_U
        # Add bias term
        restricted_logits += logits.mean(dim=0, keepdim=True) - restricted_logits.mean(dim=0, keepdim=True)
        return loss_fn(restricted_logits[test_indices], test_labels)
    get_restricted_loss(model)

    get_metrics(model, metric_cache, get_restricted_loss, "restricted_loss", reset=True)
    print(metric_cache["restricted_loss"].shape)

    fig = line([train_losses[::100], test_losses[::100], metric_cache["restricted_loss"]], x=np.arange(0, len(train_losses), 100), xaxis="Epoch", yaxis="Loss", log_y=True, title="Restricted Loss Curve", line_labels=['train', 'test', "restricted_loss"], toggle_x=True, toggle_y=True, return_fig=True)
    add_lines(fig)

    fig = line([torch.tensor(test_losses[::100])/metric_cache["restricted_loss"]], x=np.arange(0, len(train_losses), 100), xaxis="Epoch", yaxis="Loss", log_y=True, title="Restricted Loss to Test Loss Ratio", toggle_x=True, toggle_y=True, return_fig=True)
    # WARNING: bugged when cancelling training half way thr ough
    add_lines(fig)

    """## Excluded Loss"""

    approx_neuron_acts = torch.zeros_like(neuron_acts)
    # approx_neuron_acts += neuron_acts.mean(dim=0)
    a = torch.arange(p)[:, None]
    b = torch.arange(p)[None, :]
    for freq in key_freqs:
        cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
        cos_apb_vec /= cos_apb_vec.norm()
        cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
        sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
        sin_apb_vec /= sin_apb_vec.norm()
        sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
    excluded_neuron_acts = neuron_acts - approx_neuron_acts
    excluded_logits = excluded_neuron_acts @ W_logit
    print(loss_fn(excluded_logits[train_indices], train_labels))

    def get_excluded_loss(model):
        logits, cache = model.run_with_cache(dataset)
        logits = logits[:, -1, :]
        neuron_acts = cache["post", 0, "mlp"][:, -1, :]
        approx_neuron_acts = torch.zeros_like(neuron_acts)
        # approx_neuron_acts += neuron_acts.mean(dim=0)
        a = torch.arange(p)[:, None]
        b = torch.arange(p)[None, :]
        for freq in key_freqs:
            cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
            cos_apb_vec /= cos_apb_vec.norm()
            cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
            approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
            sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
            sin_apb_vec /= sin_apb_vec.norm()
            sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
            approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
        excluded_neuron_acts = neuron_acts - approx_neuron_acts
        residual_stream_final = excluded_neuron_acts @ model.blocks[0].mlp.W_out + cache["resid_mid", 0][:, -1, :]
        excluded_logits = residual_stream_final @ model.unembed.W_U
        return loss_fn(excluded_logits[train_indices], train_labels)
    get_excluded_loss(model)

    get_metrics(model, metric_cache, get_excluded_loss, "excluded_loss", reset=True)
    print(metric_cache["excluded_loss"].shape)

    fig = line([train_losses[::100], test_losses[::100], metric_cache["excluded_loss"], metric_cache["restricted_loss"]], x=np.arange(0, len(train_losses), 100), xaxis="Epoch", yaxis="Loss", log_y=True, title="Excluded and Restricted Loss Curve", line_labels=['train', 'test', "excluded_loss", "restricted_loss"], toggle_x=True, toggle_y=True, return_fig=True)
    add_lines(fig)


   

if __name__ == '__main__':
    main()