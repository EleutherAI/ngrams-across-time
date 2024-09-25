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
from tqdm import tqdm
from transformers import AutoTokenizer
from nnsight import LanguageModel
from sae import SaeConfig, Sae, TrainConfig, SaeTrainer
from datasets import Dataset as HfDataset
import plotly.graph_objects as go
import lovely_tensors as lt

from ngrams_across_time.feature_circuits.circuit import get_circuit, get_residual_node_scores
from ngrams_across_time.utils.utils import assert_type, set_seeds
from ngrams_across_time.feature_circuits.patch_nodes import patch_nodes
from ngrams_across_time.feature_circuits.sae_loss import sae_loss
from ngrams_across_time.grok.transformers import CustomTransformer, TransformerConfig


lt.monkey_patch()
set_seeds(598)
device = torch.device("cuda")


def create_patch_train_data(train_data, p):
    new_train_data = []
    new_train_labels = []
    
    for i in range(len(train_data)):
        original_a, original_b, _ = train_data[i]
        original_sum = (original_a + original_b) % p
        
        # Randomly select a new data point until the label is different
        while True:
            new_a = random.randint(0, p-1)
            new_b = random.randint(0, p-1)
            new_sum = (new_a + new_b) % p
            
            if new_sum != original_sum:
                new_train_data.append([new_a, new_b, p])
                new_train_labels.append(new_sum)
                break
    
    return torch.tensor(new_train_data), torch.tensor(new_train_labels)


def get_top_nodes(nodes: dict[Any, Any], num_nodes: int) -> dict[str, tuple[float, int]]:
    '''Dict of submodule path to list of num_nodes node scores in descending order'''
    top_nodes = []
    for submodule_path, dense_act in nodes.items():
        if submodule_path == "y":
            continue
    
        for i, score in enumerate(dense_act.act):
            if len(top_nodes) < num_nodes:
                heapq.heappush(top_nodes, (score.item(), submodule_path, i))
            else:
                heapq.heappushpop(top_nodes, (score.item(), submodule_path, i))
        
        resc_score = dense_act.resc.item()
        if len(top_nodes) < num_nodes:
            heapq.heappush(top_nodes, (resc_score, submodule_path, -1)) # Use -1 to indicate resc
        else:
            heapq.heappushpop(top_nodes, (resc_score, submodule_path, -1))

    top_nodes = sorted(top_nodes, reverse=True)
    top_nodes_dict = defaultdict(list)
    for score, submodule_path, idx in top_nodes:
        top_nodes_dict[submodule_path].append((score, idx))
    return dict(top_nodes_dict) # type: ignore


def all_node_scores(checkpoint_scores) -> list[float]:
        scores = []
        for node, values in checkpoint_scores.items():
            if node == 'y':
                continue
            if isinstance(values, Tensor):
                scores.extend(values.flatten().tolist())
            else:
                scores.extend(values.act.flatten().tolist())
        return scores


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--loss", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_seed", type=int, default=999, help="Model seed to load checkpoints for")
    parser.add_argument("--run_name", type=str, default='')
    return parser.parse_args()


def main():
    args = get_args()

    # Use existing on-disk model checkpoints
    run_identifier = f"{args.model_seed}{'-' + args.run_name if args.run_name else ''}"
    PTH_LOCATION = Path(f"workspace/grok/{run_identifier}.pth")

    data_path = Path(f"workspace/eap_data/{run_identifier}.pth")
    data_path.parent.mkdir(exist_ok=True, parents=True)

    checkpoint_eap_data = torch.load(data_path) if data_path.exists() else {}

    cached_data = torch.load(PTH_LOCATION)
    model_checkpoints = cached_data["checkpoints"][::5]
    checkpoint_epochs = cached_data["checkpoint_epochs"][::5]

    # Get minimal set of checkpoints for model seed 1
    # model_checkpoints 0, 2, 12, 17, 45
    # checkpoint_epochs 0, 2, 12, 17, 45
    
    config = cached_data['config']
    config = assert_type(TransformerConfig, config)
    p = config.d_vocab - 1

    dataset = cached_data['dataset']
    labels = cached_data['labels']
    train_indices = cached_data['train_indices']
    test_indices = cached_data['test_indices']

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
    
    # Create smaller patching dataset
    len_data = 4 if args.debug else 40
    train_data = train_data[:len_data]
    train_data = train_data.cuda()
    train_labels = train_labels[:len_data]
    train_labels = train_labels.cuda()
    patch_train_data, patch_train_labels = create_patch_train_data(train_data, p)
    patch_train_data = patch_train_data.cuda()

    def metric_fn(logits, repeat: int = 1):
        labels = train_labels.repeat(repeat)
        return (
            F.cross_entropy(logits.to(torch.float64)[:, -1, :], labels, reduction="none")
        )
    
    model = CustomTransformer(config)

    # NNSight requires a tokenizer, we are passing in tensors so any tokenizer will do
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if args.patch or args.loss or args.residual:
        for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))):
            model.remove_all_hooks()
            model.load_state_dict(state_dict)
            model.cuda() # type: ignore

            checkpoint_eap_data[epoch] = {}
            checkpoint_eap_data[epoch]['parameter_norm'] = torch.cat([parameters.flatten() for parameters in model.parameters()]).flatten().norm(p=2).item()
            checkpoint_eap_data[epoch]['train_loss'] = F.cross_entropy(model(train_data.cuda()).logits.to(torch.float64)[:, -1], train_labels.cuda()).mean().item()
            checkpoint_eap_data[epoch]['test_loss'] = F.cross_entropy(model(test_data.cuda()).logits.to(torch.float64)[:, -1], test_labels.cuda()).mean().item()

            language_model = LanguageModel(model, tokenizer=tokenizer)

            sae_path = Path(f"sae/grok.{epoch}")
            if not sae_path.exists():
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

            if args.residual:
                all_submods = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
                loss_all_saes, mean_res_norm = sae_loss(train_data, language_model, all_submods, dictionaries, metric_fn)

                checkpoint_eap_data[epoch]['sae_loss'] = loss_all_saes.mean().item()
                checkpoint_eap_data[epoch]['mean_res_norm'] = mean_res_norm
            
            if args.patch:
                model.remove_all_hooks()
                nodes = get_residual_node_scores(train_data, patch_train_data, 
                                        language_model, embed, attns, mlps, 
                                        resids, metric_fn, aggregation="sum")
                checkpoint_eap_data[epoch]['residual_nodes'] = nodes
                checkpoint_eap_data[epoch]['y'] =  nodes['y'].item()
                # checkpoint_eap_data[epoch]['edges'] = dict(edges)

                probs: np.ndarray = np.array([score / checkpoint_eap_data[epoch]['y'] for score in all_node_scores(nodes)])
                with np.errstate(divide='ignore', invalid='ignore'):
                    resid_entropy = -np.nansum(probs * np.log(probs)) # type: ignore
                checkpoint_eap_data[epoch]['residual_entropy'] = resid_entropy.item()

                # Get SAE entropy
                model.remove_all_hooks()
                nodes, edges = get_circuit(train_data, patch_train_data, 
                                        language_model, embed, attns, mlps, 
                                        resids, dictionaries, metric_fn, aggregation="sum",
                                        nodes_only=True, node_threshold=0.01)
                checkpoint_eap_data[epoch]['nodes'] = nodes
                checkpoint_eap_data[epoch]['y'] =  nodes['y'].item()
                with np.errstate(divide='ignore', invalid='ignore'):
                    checkpoint_eap_data[epoch]['entropy'] = (-np.nansum(probs * np.log(probs))).item() # type: ignore


            if args.loss:
                model.remove_all_hooks()
                # Save the loss increase when ablating top 10 nodes
                num_nodes = 10
                top_nodes = get_top_nodes(
                    checkpoint_eap_data[epoch]['nodes'], num_nodes)
                all_submods = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
                loss, patch_loss, clean_loss = patch_nodes(
                    train_data, None, language_model, 
                    all_submods, dictionaries, metric_fn, top_nodes)
                checkpoint_eap_data[epoch]['circuit_zero_ablation_loss'] = loss.mean().item()

                loss, patch_loss, clean_loss = patch_nodes(
                    train_data, patch_train_data, language_model, 
                    all_submods, dictionaries, metric_fn, top_nodes)
                
                checkpoint_eap_data[epoch]['circuit_ablation_loss'] = loss.mean().item()
                checkpoint_eap_data[epoch]['clean_loss'] = clean_loss.mean().item()
                checkpoint_eap_data[epoch]['patch_loss'] = patch_loss.mean().item() # type: ignore
                checkpoint_eap_data[epoch]['ablation_loss_increase'] = (loss - clean_loss).mean().item()

                # Binary search for the number of nodes to ablate to reach random accuracy
                # A random classifier's accuracy is 1/113 = 0.0088, with a loss of 4.7330
                random_loss = -math.log(1/113)

                start = 0
                end = 4000
                mid = (start + end) // 2
                while start < end:
                    model.remove_all_hooks()
                    top_nodes = get_top_nodes(
                        checkpoint_eap_data[epoch]['nodes'], mid)
                    loss, patch_loss, clean_loss = patch_nodes(
                        train_data, patch_train_data, language_model, 
                        all_submods, dictionaries, metric_fn, top_nodes)
                    if loss.mean().item() >= random_loss:
                        end = mid
                    else:
                        start = mid + 1
                    mid = (start + end) // 2

                print(f'patched loss at {mid} nodes and epoch {epoch} against vs. target loss ~4.7:\
                {loss.mean().item()}, clean loss: {clean_loss.mean().item()}, patch loss: {patch_loss.mean().item() if patch_loss is not None else 0}') 
                
                checkpoint_eap_data[epoch]['min_nodes_to_ablate'] = mid
        torch.save(checkpoint_eap_data, data_path)
        
    checkpoint_eap_data = torch.load(data_path)

    combined_scores: list[list[float]] = [all_node_scores(scores['nodes']) for scores in checkpoint_eap_data.values()]
    combined_residual_scores: list[list[float]] = [all_node_scores(scores['residual_nodes']) for scores in checkpoint_eap_data.values()]
    abs_sum_scores = [sum([abs(score) for score in scores]) for scores in combined_scores]
    abs_sum_residual_scores = [sum([abs(score) for score in scores]) for scores in combined_residual_scores]
    square_sum_scores = [sum([score ** 2 for score in scores]) for scores in combined_scores]

    normalized_scores = [
        [score / checkpoint_eap_data[epoch]['y'] for score in scores] 
        for scores, epoch in zip(combined_scores, checkpoint_eap_data.keys())
    ]
    thresholded_normalized_scores = [np.array(scores)[np.abs(scores) > 0.01] for scores in normalized_scores]


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['train_loss'] for scores in checkpoint_eap_data.values()], mode='lines', name='Train loss'))
    fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['test_loss'] for scores in checkpoint_eap_data.values()], mode='lines', name='Test loss'))
    fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['parameter_norm'] for scores in checkpoint_eap_data.values()], mode='lines', name='Parameter norm'))
    # fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['min_nodes_to_ablate'] for scores in checkpoint_eap_data.values()], mode='lines', name='Number of nodes to ablate for random accuracy'))
    fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['entropy'] for scores in checkpoint_eap_data.values()], mode='lines', name='SAE Node entropy'))
    fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['residual_entropy'] for scores in checkpoint_eap_data.values()], mode='lines', name='Residual node entropy'))
    # fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['sae_loss'] for scores in checkpoint_eap_data.values()], mode='lines', name='SAE loss'))
    # fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['mean_res_norm'] for scores in checkpoint_eap_data.values()], mode='lines', name='mean SAE residual norm'))
    fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[len(item) for item in thresholded_normalized_scores], mode='lines', name='Number of features above normalized score threshold'))
    fig.add_trace(go.Scatter(x=checkpoint_epochs, y=abs_sum_scores, mode='lines', name='Sum of absolute scores'))
    fig.add_trace(go.Scatter(x=checkpoint_epochs, y=abs_sum_residual_scores, mode='lines', name='Sum of absolute residual scores'))
    fig.add_trace(go.Scatter(x=checkpoint_epochs, y=square_sum_scores, mode='lines', name='Sum of squared scores'))
    fig.update_layout(title="Grokking Loss and Node Entropy over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
                                           
    fig.show()
    breakpoint()

    
if __name__ == "__main__":
    main()


# Unused metrics code

# fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['patch_loss'] for scores in checkpoint_eap_data.values()], mode='lines', name='Patch loss'))
# fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['clean_loss'] for scores in checkpoint_eap_data.values()], mode='lines', name='Clean loss'))
# fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['circuit_ablation_loss'] for scores in checkpoint_eap_data.values()], mode='lines', name='Circuit ablation loss @ 10 nodes'))
# fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['ablation_loss_increase'] for scores in checkpoint_eap_data.values()], mode='lines', name='Circuit ablation loss increase'))
# fig.add_trace(go.Scatter(x=checkpoint_epochs, y=[scores['circuit_zero_ablation_loss'] for scores in checkpoint_eap_data.values()], mode='lines', name='Circuit zero ablation loss'))

# def all_edge_scores(checkpoint_scores):
#     scores = []
#     for upstream_node, downstream_modes in checkpoint_scores.items():
#         if upstream_node == 'y':
#             continue
#         for downstream_node, value in downstream_modes.items():
#             value = assert_type(Tensor, value) # torch.sparse_coo_tensor
#             # ignore edge index, just grab its value
#             scores.extend(value.values().tolist())
#     return scores


# memorization_end_epoch = 1500
# circuit_formation_end_epoch = 12500
# cleanup_end_epoch = 18_500

# def add_lines(figure):
#     figure.add_vline(memorization_end_epoch, line_dash="dash", opacity=0.7)
#     figure.add_vline(circuit_formation_end_epoch, line_dash="dash", opacity=0.7)
#     figure.add_vline(cleanup_end_epoch, line_dash="dash", opacity=0.7)
#     return figure

# fig = add_lines(fig)