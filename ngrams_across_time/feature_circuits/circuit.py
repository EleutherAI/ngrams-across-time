from collections import defaultdict
from typing import Any

import torch as t
from einops import rearrange

from ngrams_across_time.feature_circuits.dense_act import DenseAct
from ngrams_across_time.feature_circuits.attribution import patching_effect, jvp


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

def get_residual_node_scores(
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
):
    all_submods = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
    # first get the patching effect of everything on y
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        {},
        metric_fn,
        metric_kwargs=metric_kwargs,
        method='ig' # get better approximations for early layers by using ig
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


def get_circuit(
        clean,
        patch,
        model,
        embed,
        attns,
        mlps,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
        aggregation='sum', # or 'none' for not aggregating across sequence position
        nodes_only=False,
        node_threshold=0.1,
        edge_threshold=0.01,
):
    all_submods = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
    # first get the patching effect of everything on y
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method='ig' # get better approximations for early layers by using ig
    )

    def unflatten(tensor): # will break if dictionaries vary in size between layers
        b, s, f = effects[resids[0]].act.shape
        unflattened = rearrange(tensor, '(b s x) -> b s x', b=b, s=s)
        return DenseAct(act=unflattened[...,:f], res=unflattened[...,f:])
    
    # indices where effects are above the threshold
    features_by_submod = {
        submod : (effects[submod].to_tensor().flatten().abs() > node_threshold).nonzero().flatten().tolist() for submod in all_submods
    }

    n_layers = len(resids)

    nodes: dict[Any, Any] = {'y' : total_effect}
    nodes[embed.path] = effects[embed]
    for i in range(n_layers):
        nodes[attns[i].path] = effects[attns[i]]
        nodes[mlps[i].path] = effects[mlps[i]]
        nodes[resids[i].path] = effects[resids[i]]

    if nodes_only:
        if aggregation == 'sum':
            for k in nodes:
                if k != 'y':
                    nodes[k] = nodes[k].sum(dim=1)
        nodes = {k : v.mean(dim=0) for k, v in nodes.items()}
        return nodes, None # each node corresponds to one SAE feature, so this corresponds to feature importance

    edges = defaultdict(lambda : {})

    edges[f'resid_{len(resids)-1}'] = { 'y' : effects[resids[-1]].to_tensor().flatten().to_sparse() }

    def N(upstream, downstream):
        return jvp(
            clean,
            model,
            dictionaries,
            downstream,
            features_by_submod[downstream],
            upstream,
            grads[downstream],
            deltas[upstream],
            return_without_right=True,
        )

    # now we work backward through the model to get the edges
    for layer in reversed(range(len(resids))):
        resid = resids[layer]
        mlp = mlps[layer]
        attn = attns[layer]

        mlp_resid_effect, mlp_resid_grad = N(mlp, resid)
        attn_resid_effect, attn_resid_grad = N(attn, resid)

        edges[f'mlp_{layer}'][f'resid_{layer}'] = mlp_resid_effect
        edges[f'attn_{layer}'][f'resid_{layer}'] = attn_resid_effect

        if layer > 0:
            prev_resid = resids[layer-1]
        else:
            prev_resid = embed

        resid_mlp_effect, _ = N(prev_resid, mlp)
        resid_attn_effect, _ = N(prev_resid, attn)

        mlp_resid_grad = mlp_resid_grad.coalesce()
        attn_resid_grad = attn_resid_grad.coalesce()

        RMR_effect = jvp(
            clean,
            model,
            dictionaries,
            mlp,
            features_by_submod[resid],
            prev_resid,
            {feat_idx : unflatten(mlp_resid_grad[feat_idx].to_dense()) for feat_idx in features_by_submod[resid]},
            deltas[prev_resid],
        )
        RAR_effect = jvp(
            clean,
            model,
            dictionaries,
            attn,
            features_by_submod[resid],
            prev_resid,
            {feat_idx : unflatten(attn_resid_grad[feat_idx].to_dense()) for feat_idx in features_by_submod[resid]},
            deltas[prev_resid],
        )
        RR_effect, _ = N(prev_resid, resid)

        if layer > 0: 
            edges[f'resid_{layer-1}'][f'mlp_{layer}'] = resid_mlp_effect
            edges[f'resid_{layer-1}'][f'attn_{layer}'] = resid_attn_effect
            edges[f'resid_{layer-1}'][f'resid_{layer}'] = RR_effect - RMR_effect - RAR_effect
        else:
            edges['embed'][f'mlp_{layer}'] = resid_mlp_effect
            edges['embed'][f'attn_{layer}'] = resid_attn_effect
            edges['embed'][f'resid_0'] = RR_effect - RMR_effect - RAR_effect

    # rearrange weight matrices
    for child in edges:
        # get shape for child
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            weight_matrix = edges[child][parent]
            if parent == 'y':
                weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
            else:
                bp, sp, fp = nodes[parent].act.shape
                assert bp == bc
                weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
            edges[child][parent] = weight_matrix
    
    if aggregation == 'sum':
        # aggregate across sequence position
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=1)
                else:
                    weight_matrix = weight_matrix.sum(dim=(1, 4))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].sum(dim=1)

        # aggregate across batch dimension
        for child in edges:
            bc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = weight_matrix.sum(dim=(0,2)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].mean(dim=0)
    
    elif aggregation == 'none':

        # aggregate across batch dimensions
        for child in edges:
            # get shape for child
            bc, sc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, sp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=(0, 3)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            nodes[node] = nodes[node].mean(dim=0)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return nodes, edges
