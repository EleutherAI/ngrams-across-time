import os
from typing import Any
import heapq
import math
from collections import defaultdict
from argparse import ArgumentParser
import random
from pathlib import Path

from scipy import stats
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


def get_top_sae_nodes(nodes: dict[Any, Any], num_nodes: int) -> dict[str, tuple[float, int]]:
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
        
        if dense_act.resc is not None:
            res_scores = [dense_act.resc.item()]
        else:
            res_scores = dense_act.res.tolist()

        for i, score in enumerate(res_scores, start=1):
            if len(top_nodes) < num_nodes:
                heapq.heappush(top_nodes, (score, submodule_path, -i))
            else:
                heapq.heappushpop(top_nodes, (score, submodule_path, -i))
    top_nodes = sorted(top_nodes, reverse=True)
    top_nodes_dict = defaultdict(list)
    for score, submodule_path, idx in top_nodes:
        top_nodes_dict[submodule_path].append((score, idx))
    return dict(top_nodes_dict) # type: ignore


def get_top_residual_nodes(nodes: dict[Any, Any], num_nodes: int) -> dict[str, tuple[float, int]]:
    '''Dict of submodule path to list of num_nodes node scores in descending order'''
    top_nodes = []
    for submodule_path, act in nodes.items():
        if submodule_path == "y":
            continue
    
        for i, score in enumerate(act):
            if len(top_nodes) < num_nodes:
                heapq.heappush(top_nodes, (score.item(), submodule_path, i))
            else:
                heapq.heappushpop(top_nodes, (score.item(), submodule_path, i))

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


def abs_score_entropy(nodes):
    scores = [abs(score) for score in all_node_scores(nodes)]
    sum_scores = sum(scores)

    probs = np.array([score / sum_scores for score in scores])
    return stats.entropy(probs)


def min_nodes_to_random(node_scores, node_type, train_data, patch_data, language_model, all_submods, metric_fn, dictionaries):
    random_loss = -math.log(1/113)

    start = 0
    end = len(all_node_scores(node_scores))
    mid = (start + end) // 2
    while start < end:
        if node_type == "residual":
            top_nodes = get_top_residual_nodes(node_scores, mid)
        else:
            top_nodes = get_top_sae_nodes(node_scores, mid)
        # Always hit max number of nodes with zero ablations because there is some information in every node
        loss, patch_loss, clean_loss = patch_nodes(train_data, patch_data, language_model, all_submods, dictionaries, metric_fn, top_nodes)
        if loss.mean().item() >= random_loss:
            end = mid
        else:
            start = mid + 1
        mid = (start + end) // 2
    return mid


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--nodes", action="store_true")
    parser.add_argument("--circuit_size", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_seed", type=int, default=999, help="Model seed to load checkpoints for")
    parser.add_argument("--run_name", type=str, default='')
    return parser.parse_args()


def main():
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)

    args = get_args()
    run_identifier = f"{args.model_seed}{'-' + args.run_name if args.run_name else ''}"

    # Use existing on-disk model checkpoints
    MODEL_PATH = Path(f"workspace/grok/{run_identifier}.pth")
    cached_data = torch.load(MODEL_PATH)
    
    OUT_PATH = Path(f"workspace/inference/{run_identifier}.pth")
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

    model_checkpoints = cached_data["checkpoints"][::5]
    checkpoint_epochs = cached_data["checkpoint_epochs"][::5]

    # Minimal set of checkpoints for model seed 1 to show non-monoticity
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
    num_epochs = 128
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

    checkpoint_data = torch.load(OUT_PATH) if OUT_PATH.exists() else {}
    if args.circuit_size or args.residual or args.nodes:
        for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))):
            model.load_state_dict(state_dict)
            model.cuda() # type: ignore

            if epoch not in checkpoint_data:
                checkpoint_data[epoch] = {}
            checkpoint_data[epoch]['parameter_norm'] = torch.cat([parameters.flatten() for parameters in model.parameters()]).flatten().norm(p=2).item()
            checkpoint_data[epoch]['train_loss'] = F.cross_entropy(model(train_data.cuda()).logits.to(torch.float64)[:, -1], train_labels.cuda()).mean().item()
            checkpoint_data[epoch]['test_loss'] = F.cross_entropy(model(test_data.cuda()).logits.to(torch.float64)[:, -1], test_labels.cuda()).mean().item()

            language_model = LanguageModel(model, tokenizer=tokenizer)

            sae_path = Path(f"sae/{run_identifier}/grok.{epoch}")
            if not sae_path.exists():
                cfg = TrainConfig(
                    SaeConfig(multi_topk=True), 
                    batch_size=64,
                    run_name=str(sae_path),
                    log_to_wandb=True,
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
            all_submods = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]

            if args.residual:
                loss_all_saes, mean_res_norm, mse = sae_loss(train_data, language_model, all_submods, dictionaries, metric_fn)
                checkpoint_data[epoch]['sae_loss'] = loss_all_saes.mean().item()
                checkpoint_data[epoch]['sae_mse'] = mse
                checkpoint_data[epoch]['mean_res_norm'] = mean_res_norm
            
            if args.nodes:
                for method in ["ig", "grad", "attrib"]:
                    # Residual node scores with zero ablation
                    nodes = get_residual_node_scores(train_data, None, 
                                        language_model, embed, attns, mlps, 
                                        resids, metric_fn, aggregation="sum", method=method)
                    checkpoint_data[epoch][f'residual_{method}_zero'] = {'nodes': nodes}

                    # SAE node scores with zero ablation
                    nodes, _ = get_circuit(train_data, None, 
                                            language_model, embed, attns, mlps, 
                                            resids, dictionaries, metric_fn, aggregation="sum",
                                            nodes_only=True, node_threshold=0.01, method=method)
                    checkpoint_data[epoch][f'sae_{method}_zero'] = {'nodes': nodes}

                    # Gradient scores don't use an activation delta
                    if method == "grad":
                        continue

                    # Residual node scores with patch ablation
                    nodes = get_residual_node_scores(train_data, patch_train_data, 
                                        language_model, embed, attns, mlps, 
                                        resids, metric_fn, aggregation="sum", method=method)
                    checkpoint_data[epoch][f'residual_{method}_patch'] = {'nodes': nodes}

                    # SAE node scores with patch ablation
                    nodes, _ = get_circuit(train_data, patch_train_data, 
                                            language_model, embed, attns, mlps, 
                                            resids, dictionaries, metric_fn, aggregation="sum",
                                            nodes_only=True, node_threshold=0.01, method=method)
                    checkpoint_data[epoch][f'sae_{method}_patch'] = {'nodes': nodes}                

            # A random classifier's accuracy is 1/113 = 0.0088, resulting in a loss of 4.7330
            random_loss = -math.log(1/113)
            average_loss = checkpoint_data[epoch]['train_loss']
            loss_increase_to_random = random_loss - average_loss

            for node_score_type in [
                    'residual_ig_zero', 'sae_ig_zero', 'sae_attrib_zero', 'residual_attrib_zero', 'residual_grad_zero', 
                    'sae_grad_zero', 'sae_ig_patch', 'residual_ig_patch', 'sae_attrib_patch', 'residual_attrib_patch'
                ]:
                nodes = checkpoint_data[epoch][node_score_type]['nodes']
                
                # 1. Entropy measure
                # 2. Linearized approximation of number of nodes to ablate to achieve random accuracy
                checkpoint_data[epoch][node_score_type]['entropy'] = abs_score_entropy(nodes)

                scores = all_node_scores(nodes)
                sorted_scores = np.sort(scores)[::-1]
                positive_scores = sorted_scores[sorted_scores > 0]
                proxy_nodes_to_ablate = np.searchsorted(np.cumsum(positive_scores), loss_increase_to_random)
                checkpoint_data[epoch][node_score_type][f'linearized_circuit_feature_count'] = proxy_nodes_to_ablate

                # Binary search for the number of nodes to ablate to reach random accuracy
                node_type = 'residual' if 'residual' in node_score_type else 'sae'
                if args.circuit_size:
                    num_features = min_nodes_to_random(
                        checkpoint_data[epoch][node_score_type]['nodes'], node_type, train_data, None,
                        language_model, all_submods, metric_fn, dictionaries
                    )
                    checkpoint_data[epoch][node_score_type][f'circuit_feature_count'] = num_features
                    # TODO lucia track residual score norm

        torch.save(checkpoint_data, OUT_PATH)
        
    checkpoint_data = torch.load(OUT_PATH)

    def add_scores_if_exists(fig, key: str, score_type: str | None, name: str, normalize: bool = True):
        if score_type and score_type not in checkpoint_data[checkpoint_epochs[0]].keys():
            print(f"Skipping {name} because {score_type} does not exist in the checkpoint")
            return

        if (score_type and key not in checkpoint_data[checkpoint_epochs[0]][score_type].keys()) or (
            not score_type and key not in checkpoint_data[checkpoint_epochs[0]].keys()
        ):
            print(f"Skipping {name} because it does not exist in the checkpoint")
            return
        

        scores = (
            [scores[score_type][key] for scores in checkpoint_data.values()]
            if score_type
            else [scores[key] for scores in checkpoint_data.values()]
        )

        if normalize:
            # Normalize scores to be between 0 and 1
            max_score = max(scores)
            scores = [score / max_score for score in scores]
        fig.add_trace(go.Scatter(x=checkpoint_epochs, y=scores, mode='lines', name=name))
            

    for node_score_type in [
        'residual_ig_zero', 'sae_ig_zero', 'sae_attrib_zero', 'residual_attrib_zero',
        'residual_grad_zero', 'sae_grad_zero', 'sae_ig_patch', 'residual_ig_patch', 'sae_attrib_patch', 'residual_attrib_patch'
    ]:
        fig = go.Figure()
        # Underlying model and task stats
        add_scores_if_exists(fig, 'train_loss', None, 'Train loss')
        add_scores_if_exists(fig, 'test_loss', None, 'Test loss')
        add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm')

        ablation = '0' if 'zero' in node_score_type else 'patch'
        type = 'IG' if 'ig' in node_score_type else 'AtP'
        node_type = 'Residual' if 'residual' in node_score_type else 'SAE'

        add_scores_if_exists(fig, 'entropy', node_score_type, f'H({node_type} Nodes) {ablation} ablation, {type}')
        # Nodes to ablate to reach random classifier performance
        add_scores_if_exists(fig, 'linearized_circuit_feature_count', node_score_type, f'Proxy num nodes in circuit, {ablation}, {type}, {node_type}')
        add_scores_if_exists(fig, 'circuit_feature_count', node_score_type, f'Num nodes in circuit, {ablation}, {type}, {node_type}')
    
        # SAE summary stats
        # add_scores_if_exists(fig, 'sae_loss', 'SAE loss') 
        # add_scores_if_exists(fig, 'sae_mse', 'SAE MSE')
        # add_scores_if_exists(fig, 'mean_res_norm', 'mean SAE residual norm')

        fig.update_layout(
            title=f"Grokking Loss and Node Entropy over Epochs for seed {run_identifier} and node score type {node_score_type.replace('_', ' ')}", 
            xaxis_title="Epoch", 
            yaxis_title="Loss",
            width=1000
        )                                           
        fig.write_image(images_path / f'detect_{node_score_type}_{run_identifier}.pdf', format='pdf')

    
if __name__ == "__main__":
    main()