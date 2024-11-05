from typing import Any

from nnsight.intervention import InterventionProxy

import torch as t
import nnsight

from ngrams_across_time.feature_circuits.dense_act import to_dense
from ngrams_across_time.feature_circuits.attribution import patching_effect


###### utilities for dealing with sparse COO tensors ######
def flatten_index(idxs, shape):
    """
    index : a tensor of shape [n, len(shape)]
    shape : a shape
    return a tensor of shape [n] where each element is the flattened index
    """
    idxs = idxs.t()
    # get strides from shape
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = list(reversed(strides))
    strides = t.tensor(strides).to(idxs.device)
    # flatten index
    return (idxs * strides).sum(dim=1).unsqueeze(0)

def prod(l):
    out = 1
    for x in l: out *= x
    return out

def sparse_flatten(x):
    x = x.coalesce()
    return t.sparse_coo_tensor(
        flatten_index(x.indices(), x.shape),
        x.values(),
        (prod(x.shape),)
    )

def reshape_index(index, shape):
    """
    index : a tensor of shape [n]
    shape : a shape
    return a tensor of shape [n, len(shape)] where each element is the reshaped index
    """
    multi_index = []
    for dim in reversed(shape):
        multi_index.append(index % dim)
        index //= dim
    multi_index.reverse()
    return t.stack(multi_index, dim=-1)

def sparse_reshape(x, shape):
    """
    x : a sparse COO tensor
    shape : a shape
    return x reshaped to shape
    """
    # first flatten x
    x = sparse_flatten(x).coalesce()
    new_indices = reshape_index(x.indices()[0], shape)
    return t.sparse_coo_tensor(new_indices.t(), x.values(), shape)

def sparse_mean(x, dim):
    if isinstance(dim, int):
        return x.sum(dim=dim) / x.shape[dim]
    else:
        return x.sum(dim=dim) / prod(x.shape[d] for d in dim)

######## end sparse tensor utilities ########

def get_transformer_resid_node_scores(
        clean,
        patch,
        model,
        embed,
        attns,
        mlps,
        resids,
        metric_fn,
        metric_kwargs=dict(),
        aggregation='sum', # or 'none' for not aggregating across sequence position
        method='ig' # get better approximations for early layers by using ig
):
    all_submods = [embed] + [submod for layer_submods in zip(attns, mlps, resids) for submod in layer_submods]
    # first get the patching effect of everything on y
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        {},
        metric_fn,
        metric_kwargs=metric_kwargs,
        method=method,
        dummy_input=clean[:3]
    )

    nodes: dict[Any, Any] = {'y' : total_effect}
    nodes[embed.path] = effects[embed]
    for i in range(len(resids)):
        nodes[attns[i].path] = effects[attns[i]]
        nodes[mlps[i].path] = effects[mlps[i]]
        nodes[resids[i].path] = effects[resids[i]]

    if aggregation == 'sum':
        for k in nodes:
            if k != 'y':
                nodes[k] = nodes[k].sum(dim=1)
    nodes = {k : v.mean(dim=0) for k, v in nodes.items()}
    return nodes # each node corresponds to one SAE feature, so this corresponds to feature importance



def get_resid_node_scores(
        clean,
        patch,
        model,
        all_submods,
        metric_fn,
        metric_kwargs=dict(),
        aggregate=True,
        method='ig' # get better approximations for early layers by using ig
):

    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        {},
        metric_fn,
        metric_kwargs=metric_kwargs,
        method=method,
        dummy_input=clean[:3]
    )

    nodes = {
        submod.path: effects[submod]
        for submod in all_submods
    }

    if aggregate:
        nodes = {k : v.sum(dim=1) for k, v in nodes.items()}
        
    nodes['y'] = total_effect

    nodes = {k : v.mean(dim=0) for k, v in nodes.items()}
    return nodes

def get_circuit(
        clean,
        patch,
        model,
        ordered_submods: list[Any],
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
        aggregate: bool = True, # or 'none' for not aggregating across sequence position
        method='ig' # get better approximations for early layers by using ig
):
    """Get effect scores for each SAE feature using integrated gradients or AtP."""

    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        ordered_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method=method,
        dummy_input=clean[:3]
    )

    nodes = {
        submod.path: effects[submod]
        for submod in ordered_submods
    }

    if aggregate:
        nodes = {k : v.sum(dim=1) for k, v in nodes.items()}

    nodes['y'] = total_effect

    nodes = {k : v.mean(dim=0) for k, v in nodes.items()}
    
    return nodes 


def get_mean_sae_entropy(
        model,
        ordered_submods: list[Any],
        dictionaries,
        data,
        batch_size: int | None = None,
        aggregate: bool = True, # or 'none' for not aggregating across sequence position
        device: str | t.device = t.device("cuda")
):
    num_latents = next(iter(dictionaries.values())).num_latents

    if batch_size is None:
        batch_size = len(data)

    is_tuple = {}
    with model.scan(data[:2]):
        for submodule in ordered_submods:
            is_tuple[submodule] = type(submodule.output.shape) == tuple
    
    accumulated_nodes = {}
    accumulated_fvu = {}
    accumulated_multi_topk_fvu = {}
    total_samples = 0

    for batch_start in range(0, len(data), batch_size):
        batch_end = min(batch_start + batch_size, len(data))
        batch_data = data[batch_start:batch_end].to(device)
        current_batch_size = len(batch_data)
        if current_batch_size != batch_size:
            # print("Truncating data to multiple of batch size")
            break
        
        batch_nodes = {}
        batch_fvus = {}
        batch_multi_topk_fvus = {}
        with model.trace(batch_data, scan=True):
            for submodule in ordered_submods:
                input = submodule.output if not is_tuple[submodule] else submodule.output[0]

                dictionary = dictionaries[submodule]
                flat_f: InterventionProxy = nnsight.apply(dictionary.forward, input.flatten(0, 1))
                batch_fvus[submodule.path] = flat_f.fvu.cpu().detach().save()
                batch_multi_topk_fvus[submodule.path] = flat_f.multi_topk_fvu.cpu().detach().save()
                latent_acts = flat_f.latent_acts.reshape(input.shape[0], input.shape[1], -1)
                latent_indices = flat_f.latent_indices.reshape(input.shape[0], input.shape[1], -1)
                dense = to_dense(latent_acts, latent_indices, num_latents)
                batch_nodes[submodule.path] = dense.cpu().detach().save()
        
        if aggregate:
            batch_nodes = {k: v.sum(dim=1) for k, v in batch_nodes.items()}
        
        for k, v in batch_nodes.items():
            if k not in accumulated_nodes:
                accumulated_nodes[k] = t.zeros_like(v)
            accumulated_nodes[k] += v * current_batch_size

        for k, v in batch_fvus.items():
            if k not in accumulated_fvu:
                accumulated_fvu[k] = 0
            accumulated_fvu[k] += v * current_batch_size

        for k, v in batch_multi_topk_fvus.items():
            if k not in accumulated_multi_topk_fvu:
                accumulated_multi_topk_fvu[k] = 0
            accumulated_multi_topk_fvu[k] += v * current_batch_size
        
        total_samples += current_batch_size

    nodes = {k: v / total_samples for k, v in accumulated_nodes.items()}
    fvu = {k: v / total_samples for k, v in accumulated_fvu.items()}
    multi_topk_fvu = {k: v / total_samples for k, v in accumulated_multi_topk_fvu.items()}

    nodes = {k: v.mean(dim=0) for k, v in nodes.items()}
    
    return nodes, fvu, multi_topk_fvu