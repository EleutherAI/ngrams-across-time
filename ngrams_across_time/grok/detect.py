import os
from typing import Any
import heapq
import math
from collections import defaultdict
from argparse import ArgumentParser
import random
from pathlib import Path

import torch.nn.functional as F
import torch
import numpy as np
from torch import Tensor
import einops
from tqdm import tqdm
from transformers import AutoTokenizer
from nnsight import LanguageModel
from sae import Sae, SaeConfig, TrainConfig, SaeTrainer
from datasets import Dataset as HfDataset
from ngrams_across_time.feature_circuits.circuit import get_circuit
from ngrams_across_time.utils.utils import assert_type
from ngrams_across_time.feature_circuits.patch_nodes import patch_nodes
from ngrams_across_time.grok.transformers import CustomTransformer, TransformerConfig
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import lovely_tensors as lt

lt.monkey_patch()

device = torch.device("cuda")

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--loss", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def set_seeds(seed = 598):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seeds()

def permute_train_data(train_data, p):
    new_train_data = []
    new_train_labels = []
    
    for i in range(len(train_data)):
        original_a, original_b, _ = train_data[i]
        original_sum = (original_a + original_b) % p
        
        while True:
            # Randomly select a new pair (a, b)
            new_a = random.randint(0, p-1)
            new_b = random.randint(0, p-1)
            new_sum = (new_a + new_b) % p
            
            # Check if the new sum is different from the original
            if new_sum != original_sum:
                new_train_data.append(torch.tensor([new_a, new_b, p]))
                new_train_labels.append(new_sum)
                break
    
    return torch.stack(new_train_data), torch.tensor(new_train_labels)


def get_top_nodes(nodes: dict[Any, Any], num_nodes: int = 10) -> dict[str, tuple[float, int]]:
    top_nodes = []
    for submodule_path, dense_act in nodes.items():
        if submodule_path == "y":
            continue
    
        # Check scores in the 'act' tensor
        for i, score in enumerate(dense_act.act):
            if len(top_nodes) < num_nodes:
                heapq.heappush(top_nodes, (score.item(), submodule_path, i))
            else:
                heapq.heappushpop(top_nodes, (score.item(), submodule_path, i))
        
        resc_score = dense_act.resc.item()
        if len(top_nodes) < num_nodes:
            heapq.heappush(top_nodes, (resc_score, submodule_path, -1))  # Use -1 to indicate resc
        else:
            heapq.heappushpop(top_nodes, (resc_score, submodule_path, -1))

    # Return the results in descending order of scores
    top_nodes = sorted(top_nodes, reverse=True)
    if any([node[0] < 0 for node in top_nodes]):
        print('Not all top nodes have positive scores')
    top_nodes_dict = defaultdict(list)
    for score, submodule_path, idx in top_nodes:
        top_nodes_dict[submodule_path].append((score, idx))
    return dict(top_nodes_dict) # type: ignore

def all_node_scores(checkpoint_scores):
        scores = []
        for node, values in checkpoint_scores.items():
            if node == 'y':
                continue
            if isinstance(values, Tensor):
                scores.extend(values.flatten().tolist())
            else:
                scores.extend(values.act.flatten().tolist())
        return scores

def all_edge_scores(checkpoint_scores):
        scores = []
        for upstream_node, downstream_modes in checkpoint_scores.items():
            if upstream_node == 'y':
                continue
            for downstream_node, value in downstream_modes.items():
                value = assert_type(Tensor, value) # torch.sparse_coo_tensor
                # ignore edge index, just grab its value
                scores.extend(value.values().tolist())
        return scores

def plot_histograms(combined_scores):
    threshold = 2
    fig = make_subplots(rows=len(combined_scores), cols=4, column_widths=[1, 0.3, 1, 0.3])

    for i, scores in enumerate(combined_scores, 1):
        positive_small = scores[scores > 0]
        positive_small = positive_small[positive_small < threshold]
        positive_large = scores[scores >= threshold]
        negative = scores[scores < 0]
        zero_count = np.sum(scores == 0)

        fig.add_trace(go.Histogram(x=positive_small, name="Positive values", nbinsx=50), 
                        row=i, col=1)

        fig.add_trace(go.Histogram(x=positive_large, name="Positive values", nbinsx=50,),
                        row=i, col=2)

        # Histogram for negative values
        fig.add_trace(
            go.Histogram(x=negative, name="Negative values", nbinsx=50),
            row=i, col=3
        )

        # Bar for zero count
        fig.add_trace(
            go.Bar(x=['Zero values'], y=[zero_count], name="Zero values"),
            row=i, col=4
        )

        # fig.update_xaxes(title="Value", type="log", row=i, col=1)
        fig.update_xaxes(title="Value", row=i, col=2)
        fig.update_yaxes(title="Frequency", row=i, col=3)
        # fig.update_yaxes(title="Frequency", row=i, col=1)
        fig.update_yaxes(title="Count", row=i, col=1)

    # fig.update_layout(
    #     title="Distribution of Values",
    #     showlegend=False,
    #     height=900
    # )

    fig.update_layout(
        height=300 * len(combined_scores),  # Adjust height based on number of plots
        width=1200,
        title_text="Distribution of EAP Scores over Checkpoints",
        showlegend=False
    )
    fig.show()

def main():
    args = get_args()
    PATCH = args.patch
    LOSS = args.loss
    OVERWRITE = args.overwrite
    debug = args.debug
    
    # Generate dataset
    p = 113
    frac_train = 0.3
    a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
    b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
    equals_vector = einops.repeat(torch.tensor(113), " -> (i j)", i=p, j=p)
    dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
    labels = (dataset[:, 0] + dataset[:, 1]) % p

    indices = torch.randperm(p*p)
    cutoff = int(p*p*frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    train_data = dataset[train_indices]
    train_labels = labels[train_indices]

    test_data = dataset[test_indices]
    test_labels = labels[test_indices]
    
    # SAEs like multiple epochs
    num_epochs = 32
    sae_train_dataset = HfDataset.from_dict({
        "input_ids": train_data.repeat(num_epochs, 1),
        "labels": train_labels.repeat(num_epochs),
    })
    sae_train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    # Define model with existing on-disk checkpoints
    workspace = Path("workspace")
    workspace.mkdir(exist_ok=True)
    PTH_LOCATION = Path("workspace/grokking_demo.pth")

    cached_data = torch.load(PTH_LOCATION)
    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data["checkpoint_epochs"]

    model = CustomTransformer(TransformerConfig(
        vocab_size=p + 1,
        hidden_size=128,
        n_ctx=5,
        num_layers=1,
        num_heads=4,
        intermediate_size=512,
        act_type = "ReLU",
        use_ln = False
    ))

    data_path = Path("workspace/checkpoint_eap_data.pth")
    if data_path.exists():
        checkpoint_eap_data = torch.load(data_path)
    else:
        checkpoint_eap_data = {}

    model.load_state_dict(model_checkpoints[-1])
    model.cuda() # type: ignore
    logits = model(train_data[:2].cuda()).logits # type: ignore
    loss = F.cross_entropy(
        logits[:, -1, :], train_labels[:2].cuda(), reduction="none"
    )

    # NNSight requires a tokenizer, we are passing in tensors so any tokenizer will do
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if PATCH or LOSS:
        # every tenth epoch is recorded starting from 9
        for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))):
        # for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))[-1:]):
            model.remove_all_hooks()
            model.load_state_dict(state_dict)
            model.cuda() # type: ignore
            language_model: LanguageModel = LanguageModel(model, tokenizer=tokenizer) # type: ignore 

            # Load SAEs for epoch
            sae_path = Path(f"sae/grok.{epoch}")
            if not os.path.exists(sae_path):
                cfg = TrainConfig(
                    SaeConfig(multi_topk=True), 
                    batch_size=16,
                    run_name=str(sae_path),
                    log_to_wandb=False,
                    hookpoints=[
                        "blocks.*.hook_resid_pre", # start of block
                        "blocks.*.hook_resid_mid", # after ln / attention
                        "blocks.*.hook_resid_post", # after ln / mlp
                        "blocks.*.hook_mlp_out" # basically the same as resid_post but without the resid added
                    ],
                )
                trainer = SaeTrainer(cfg, sae_train_dataset, model)
                trainer.fit()
            
            # These correspondences are fairly borked but it's fine for our purposes
            embed = language_model.blocks[(0)].hook_resid_pre
            attns = [block.hook_resid_mid for block in language_model.blocks]
            mlps = [block.hook_mlp_out for block in language_model.blocks]
            resids = [block.hook_resid_post for block in language_model.blocks]

            dictionaries = {}
            dictionaries[embed] = Sae.load_from_disk(
                os.path.join(sae_path, 'blocks.0.hook_resid_pre'),
                device=device
            )

            for i in range(len(language_model.blocks)): # type: ignore
                dictionaries[attns[i]] = Sae.load_from_disk(
                    os.path.join(sae_path, f'blocks.{i}.hook_resid_mid'),
                    device=device
                )
                dictionaries[mlps[i]] = Sae.load_from_disk(
                    os.path.join(sae_path, f'blocks.{i}.hook_mlp_out'),
                    device=device
                )
                dictionaries[resids[i]] = Sae.load_from_disk(
                    os.path.join(sae_path, f'blocks.{i}.hook_resid_post'),
                    device=device
                )

            # Run EAP with SAEs
            # Count the loss increase when ablating an arbitrary number of edges (10):
            node_threshold = 0.01
            num_nodes = 10
            len_data = 4 if debug else 40
            train_data = train_data[:len_data]
            train_data = train_data.cuda()
            train_labels = train_labels[:len_data]
            permuted_train_data, permuted_train_labels = permute_train_data(train_data, p)
            permuted_train_data = permuted_train_data.cuda()

            def metric_fn(logits, repeat: int = 1):
                labels = train_labels
                if repeat > 1:
                    labels = labels.repeat(repeat)
                return (
                    F.cross_entropy(logits[:, -1, :], labels, reduction="none")
                )

            # initial_loss = metric_fn(model(permuted_train_data.cuda()).logits).mean().item()
            
            if PATCH and (OVERWRITE or epoch not in checkpoint_eap_data):
                nodes, edges = get_circuit(train_data, permuted_train_data, 
                                        language_model, embed, attns, mlps, 
                                        resids, dictionaries, metric_fn, 
                                        aggregation="sum", node_threshold=node_threshold, edge_threshold=node_threshold, nodes_only=True)
                # check whether dict with the same epoch value already exists in data
                if epoch in checkpoint_eap_data:
                    prev_data = checkpoint_eap_data[epoch].copy()
                else:
                    prev_data = None
                checkpoint_eap_data[epoch] = {
                    'nodes': nodes,
                    # 'edges': dict(edges),
                    'prev': prev_data,
                    'y': nodes['y'].item()
                }
                torch.save(checkpoint_eap_data, data_path)

            if LOSS and (OVERWRITE or 'loss' not in checkpoint_eap_data[epoch]):
                # Save the loss increase when ablating all top nodes
                top_nodes = get_top_nodes(
                    checkpoint_eap_data[epoch]['nodes'], num_nodes)
                # print('top nodes for epoch', epoch, top_nodes)
                all_submods = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
                loss, patch_loss, clean_loss = patch_nodes(
                    train_data, permuted_train_data, language_model, 
                    all_submods, dictionaries, metric_fn, top_nodes)
                
                checkpoint_eap_data[epoch]['circuit_ablation_loss'] = loss.mean().item()
                checkpoint_eap_data[epoch]['clean_loss'] = clean_loss.mean().item()
                checkpoint_eap_data[epoch]['patch_loss'] = patch_loss.mean().item()
                checkpoint_eap_data[epoch]['ablation_loss_increase'] = (loss - clean_loss).mean().item()

                # Binary search to find the minimum number of nodes to ablate to cause a loss increase of 2
                # TODO use accuracy instead: min nodes to get ranfom accuracy
                start = 0
                end = 4000
                mid = (start + end) // 2
                # a random classifier's accuracy is 1/113 = 0.0088
                # this corresponds to a loss of 4.7330
                random_loss = -math.log(1/113)
                while start < end:
                    model.remove_all_hooks()
                    # language_model = LanguageModel(model, tokenizer=tokenizer) # type: ignore 
                    top_nodes = get_top_nodes(
                        checkpoint_eap_data[epoch]['nodes'], mid)
                    loss, patch_loss, clean_loss = patch_nodes(
                        train_data, permuted_train_data, language_model, 
                        all_submods, dictionaries, metric_fn, top_nodes)
                    if loss.mean().item() >= random_loss:
                        end = mid
                    else:
                        start = mid + 1
                    mid = (start + end) // 2

                print(f'patched loss at {mid} nodes and epoch {epoch} against vs. target loss ~4.7:\
                {loss.mean().item()}, clean loss: {clean_loss.mean().item()}, patch loss: {patch_loss.mean().item()}') 
                
                checkpoint_eap_data[epoch]['min_nodes_to_ablate'] = mid
                torch.save(checkpoint_eap_data, data_path)
        
    checkpoint_eap_data = torch.load(data_path)

    combined_scores: list[list[float]] = [all_node_scores(scores['nodes']) for scores in checkpoint_eap_data.values()]
    # combined_scores: list[list[float]] = [all_edge_scores(scores['edges']) for scores in checkpoint_eap_data.values()]
    # Rearrangement: more components are relevant during grokking because both circuits are present
    # Prediction: ablating the same number of components causes a higher loss increase after grokking 

    # produce scores normalized by the y for that epoch
    
    normalized_scores = [
        [score / checkpoint_eap_data[epoch]['y'] for score in scores] 
        for scores, epoch in zip(combined_scores, checkpoint_eap_data.keys())
    ]

    thresholded_scores = [np.array(scores)[np.abs(scores) > 0.5] for scores in combined_scores]
    thresholded_normalized_scores = [np.array(scores)[np.abs(scores) > 0.01] for scores in normalized_scores]

    # plot line plot with data
    epochs = sorted([epoch for epoch in checkpoint_eap_data.keys()])
    ab = [len(checkpoint_eap_data[epoch].keys()) for epoch in checkpoint_eap_data.keys()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=[scores['patch_loss'] for scores in checkpoint_eap_data.values()], mode='lines', name='Patch loss'))
    fig.add_trace(go.Scatter(x=epochs, y=[scores['clean_loss'] for scores in checkpoint_eap_data.values()], mode='lines', name='Clean loss'))
    fig.add_trace(go.Scatter(x=epochs, y=[scores['circuit_ablation_loss'] for scores in checkpoint_eap_data.values()], mode='lines', name='Circuit ablation loss @ 10 nodes'))
    fig.add_trace(go.Scatter(x=epochs, y=[scores['ablation_loss_increase'] for scores in checkpoint_eap_data.values()], mode='lines', name='Circuit ablation loss increase'))
    fig.add_trace(go.Scatter(x=epochs, y=[scores['min_nodes_to_ablate'] for scores in checkpoint_eap_data.values()], mode='lines', name='Number of nodes to ablate for random accuracy'))
    fig.add_trace(go.Scatter(x=epochs, y=[len(item) for item in thresholded_scores], mode='lines', name='Number of features above score threshold'))
    fig.add_trace(go.Scatter(x=epochs, y=[len(item) for item in thresholded_normalized_scores], mode='lines', name='Number of features above normalized score threshold'))

    fig.update_layout(title="Patched Loss and Loss Increase over Epochs", xaxis_title="Epoch", yaxis_title="Loss")

    memorization_end_epoch = 1500
    circuit_formation_end_epoch = 13300
    cleanup_end_epoch = 16600

    def add_lines(figure):
        figure.add_vline(memorization_end_epoch, line_dash="dash", opacity=0.7)
        figure.add_vline(circuit_formation_end_epoch, line_dash="dash", opacity=0.7)
        figure.add_vline(cleanup_end_epoch, line_dash="dash", opacity=0.7)
        return figure

    fig = add_lines(fig)

    fig.show()
    
    # Patch loss is very noisy. 
    # There is a bump in the clean loss during grokking, whereas in the paper it only falls. perhaps they don't log their loss in high
    # enough definition to see the bump
    breakpoint()
    plot_histograms(thresholded_scores)
    plot_histograms(thresholded_normalized_scores)

    breakpoint()

if __name__ == "__main__":
    main()