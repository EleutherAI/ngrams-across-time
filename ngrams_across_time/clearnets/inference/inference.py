# Functions to collect SAE-based metrics

from typing import Callable, Any
from collections import defaultdict
from functools import partial

import torch
import numpy as np
from torch import Tensor
from sae.sae import Sae
from torch.utils.data import DataLoader
import nnsight

from ngrams_across_time.clearnets.metrics import gini, hoyer, hoyer_square, abs_entropy


def get_batch_metrics(
    acts: Tensor, 
    accumulator: defaultdict[str, list], 
    feature_dims = [2], 
    instance_dims = [0, 1]
):
    if acts.ndim != 2:
        permuted_dims = instance_dims + feature_dims
        acts = acts.permute(*permuted_dims)
        acts = acts.flatten(0, len(instance_dims) - 1)
        acts = acts.flatten(1, len(feature_dims))
    
    l2_norms = torch.norm(acts, dim=1)

    # L2 norm mean
    accumulator['mean_l2'].append(l2_norms.mean().item())
    
    # L2 norm variance
    accumulator['var_l2'].append(l2_norms.var().item())
    
    # L2 norm skewness: E[(X - μ)³] / σ³
    centered = l2_norms - l2_norms.mean()
    skew = (centered ** 3).mean() / (l2_norms.std() ** 3)
    accumulator['skew_l2'].append(skew.item())
    
    # L2 norm kurtosis: E[(X - μ)⁴] / σ⁴ - 3 (subtracting 3 gives excess kurtosis)
    kurt = (centered ** 4).mean() / (l2_norms.var() ** 2) - 3
    accumulator['kurt_l2'].append(kurt.item())

    # Element-wise variance
    accumulator['var_trace'].append(torch.trace(torch.cov(acts.T.float())).item())

    return accumulator
    

def to_dense(top_acts: Tensor, top_indices: Tensor, num_latents: int):
    # In-place scatter seemed to break nnsight
    dense_empty = torch.zeros(top_acts.shape[0], top_acts.shape[1], num_latents, device=top_acts.device, dtype=top_acts.dtype, requires_grad=True)
    return dense_empty.scatter(-1, top_indices.long(), top_acts)


def concatenate_values(dictionary: dict[Any, Tensor]) -> np.ndarray:
    scores = []
    for values in dictionary.values():
        scores.extend(values.flatten().tolist())
        
    return np.array(scores)


@torch.no_grad()
def get_sae_acts(
        model,
        ordered_submods: list,
        dictionaries: dict[Any, Sae],
        dataloader: DataLoader,
        # for VIT this is equal to the number of patches
        seq_len: int | None,
        aggregate: bool = True, # or 'none' for not aggregating across sequence position
        device: str | torch.device = torch.device("cuda"),
        input_key: str = 'input_ids',
        act_callback: Callable | None = None
):
    """Meaned over a large dataset"""
    # Collect metadata and preallocate tensors    
    is_tuple = {}
    first_batch = next(iter(dataloader))
    batch_size = first_batch[input_key].shape[0]
    accumulated_nodes = {}
    # with torch.amp.autocast(str(device)):
    with model.scan(first_batch[input_key].to(device)) as tracer:
        for submodule in ordered_submods:
            output = (
                submodule.nns_output 
                if hasattr(submodule, 'nns_output') 
                else submodule.output
            )
            is_tuple[submodule] = type(output.shape) == tuple 

    with model.trace(first_batch[input_key].to(device)) as tracer:
        for submodule in ordered_submods:
            num_latents = dictionaries[submodule].num_latents
            shape = (batch_size, num_latents) if aggregate or seq_len is None else (batch_size, seq_len, num_latents)
            accumulated_nodes[submodule.path] = torch.zeros(shape, dtype=torch.float64).save() # type: ignore

    # Do inference
    total_samples = 0
    num_batches = 0
    accumulated_fvu = {submodule.path: 0. for submodule in ordered_submods}
    accumulated_multi_topk_fvu = {submodule.path: 0. for submodule in ordered_submods}
    callback_accumulator = defaultdict(list)

    with torch.amp.autocast(str(device)):
        for batch in dataloader:
            batch_nodes = {}
            batch_fvus = {}
            batch_multi_topk_fvus = {}
            
            with model.trace(batch[input_key].to(device)) as tracer:
                for submodule in ordered_submods:
                    num_latents = dictionaries[submodule].num_latents

                    if hasattr(submodule, 'nns_output'):
                        input = submodule.nns_output if not is_tuple[submodule] else submodule.nns_output[0]
                    else:
                        input = submodule.output if not is_tuple[submodule] else submodule.output[0]

                    dictionary = dictionaries[submodule]

                    flat_f = nnsight.apply(dictionary.forward, input.flatten(0, 1))
                    batch_fvus[submodule.path] = flat_f.fvu.cpu().save()
                    batch_multi_topk_fvus[submodule.path] = flat_f.multi_topk_fvu.cpu().save()

                    latent_acts = flat_f.latent_acts.view(input.shape[0], input.shape[1], -1)
                    latent_indices = flat_f.latent_indices.view(input.shape[0], input.shape[1], -1)
                    dense = to_dense(latent_acts, latent_indices, num_latents) # type: ignore
                    batch_nodes[submodule.path] = dense.cpu().save() # type: ignore
            
            for submodule in ordered_submods:
                if aggregate: 
                    batch_nodes[submodule.path] = batch_nodes[submodule.path].mean(dim=1)

                accumulated_nodes[submodule.path] += batch_nodes[submodule.path].double()
                accumulated_fvu[submodule.path] += batch_fvus[submodule.path].item()
                accumulated_multi_topk_fvu[submodule.path] += batch_multi_topk_fvus[submodule.path].item()
                
            acts = torch.cat([batch_nodes[submodule.path] for submodule in ordered_submods], dim=0)
            if act_callback is not None:
                callback_accumulator = act_callback(acts, callback_accumulator)

            total_samples += batch_size
            num_batches += 1

    nodes = {k: v.sum(0) / total_samples for k, v in accumulated_nodes.items()}
    fvu = {k: v / num_batches for k, v in accumulated_fvu.items()}
    multi_topk_fvu = {k: v / num_batches for k, v in accumulated_multi_topk_fvu.items()}


    return nodes, fvu, multi_topk_fvu, callback_accumulator


def get_metrics(
    model, 
    nnsight_model,
    dictionaries: dict,
    all_submods: list,
    train_dl, 
    test_dl, 
    seq_len: int,
    feature_dims=[1],
    instance_dims=[0]

):
    metrics: dict[str, Any] = {}

    metrics['parameter_norm'] = torch.cat([parameters.flatten() for parameters in model.parameters()]).flatten().norm(p=2).item()

    mean_nodes, fvu, multi_topk_fvu, metrics = get_sae_acts(
        nnsight_model, all_submods, dictionaries, 
        train_dl, aggregate=True, input_key='input_ids',
        seq_len=seq_len, act_callback=partial(get_batch_metrics, feature_dims=feature_dims, instance_dims=instance_dims)

    )
    mean_node_scores = concatenate_values(
        mean_nodes)

    mean_fvu = np.mean([v for v in fvu.values()])
    mean_multi_topk_fvu = np.mean([v for v in multi_topk_fvu.values()])

    metrics['sae_entropy_nodes'] = {'nodes': mean_nodes}   

    metrics['sae_fvu'] = mean_fvu
    metrics['sae_multi_topk_fvu'] = mean_multi_topk_fvu

    metrics['sae_entropy'] = abs_entropy(mean_node_scores)
    metrics['hoyer'] = hoyer(mean_node_scores)
    metrics['hoyer_square'] = hoyer_square(mean_node_scores)
    metrics['gini'] = gini(mean_node_scores)
    metrics['mean_l2'] = metrics['mean_l2']
    metrics['var_l2'] = metrics['var_l2']
    metrics['skew_l2'] = metrics['skew_l2'] 
    metrics['kurt_l2'] = metrics['kurt_l2']
    metrics['var_trace'] = metrics['var_trace']

    del mean_node_scores, mean_nodes, fvu, multi_topk_fvu

    test_nodes, test_fvu, test_multi_topk_fvu, running_metrics = get_sae_acts(
        nnsight_model, all_submods, dictionaries, 
        test_dl, aggregate=True, seq_len=seq_len, act_callback=get_batch_metrics
    )
    metrics['test_sae_fvu'] = np.mean([v for v in test_fvu.values()])
    metrics['test_sae_multi_topk_fvu'] = np.mean([v for v in test_multi_topk_fvu.values()])
    metrics['test_sae_entropy_nodes'] = {'nodes': test_nodes}   

    mean_test_nodes = {k: v.mean(dim=0) for k, v in test_nodes.items()}
    mean_test_node_scores = concatenate_values(mean_test_nodes)

    metrics['test_sae_entropy'] = abs_entropy(mean_test_node_scores)
    metrics['test_hoyer'] = hoyer(mean_test_node_scores)
    metrics['test_hoyer_square'] = hoyer_square(mean_test_node_scores)
    metrics['test_gini'] = gini(mean_test_node_scores)

    metrics['test_mean_l2'] = running_metrics['mean_l2']
    metrics['test_var_l2'] = running_metrics['var_l2']
    metrics['test_skew_l2'] = running_metrics['skew_l2']
    metrics['test_kurt_l2'] = running_metrics['kurt_l2']
    metrics['test_var_trace'] = running_metrics['var_trace']

    return metrics