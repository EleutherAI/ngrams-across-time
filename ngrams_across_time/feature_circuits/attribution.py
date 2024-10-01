from collections import namedtuple

import torch as t
from tqdm import tqdm
from numpy import ndindex
from typing import Dict, Union, Any
from sae import Sae
from sae.sae import ForwardOutput
import nnsight
from ngrams_across_time.feature_circuits.dense_act import DenseAct, to_dense

DEBUGGING = False

if DEBUGGING:
    tracer_kwargs = {'validate' : True, 'scan': True}
else:
    tracer_kwargs = {}

EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])

def _pe_attrib(
        clean,
        patch,
        model,
        submodules,
        dictionaries: dict[Any, Sae],
        metric_fn,
        metric_kwargs=dict(),
):
    num_latents = next(iter(dictionaries.values())).num_latents if dictionaries else None
    
    # Figure out which hidden states are tuples using a test input
    is_tuple = {}
    with model.scan("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    hidden_states_clean = {}
    grads = {}
    with model.trace(clean, scan=True):
        for submodule in submodules:
            saved_x = submodule.output.save()
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]

            if not submodule in dictionaries:
                hidden_states_clean[submodule] = x.save()
                grads[submodule] = x.grad.save()
                continue

            dictionary = dictionaries[submodule]
            flat_f = nnsight.apply(dictionary.forward, x.flatten(0, 1))
            f = ForwardOutput(
                latent_acts=flat_f.latent_acts.reshape(x.shape[0], x.shape[1], -1),
                latent_indices=flat_f.latent_indices.reshape(x.shape[0], x.shape[1], -1),
                sae_out=flat_f.sae_out.reshape(x.shape[0], x.shape[1], -1),
                fvu=flat_f.fvu,
                auxk_loss=flat_f.auxk_loss,
                multi_topk_fvu=flat_f.multi_topk_fvu
            )

            dense = to_dense(f.latent_acts, f.latent_indices, num_latents)
            x_hat = dense @ dictionary.W_dec.mT.T
            residual = x - x_hat

            hidden_states_clean[submodule] = DenseAct(act=dense, res=residual).save()
            grads[submodule] = hidden_states_clean[submodule].grad.save()
            residual.grad = t.zeros_like(residual)
            x_recon = x_hat + residual
            if is_tuple[submodule]:
                submodule.output[0][:] = x_recon
            else:
                submodule.output = x_recon
            x.grad = x_recon.grad
        metric_clean = metric_fn(model.output.logits, **metric_kwargs).save()
        metric_clean.sum().backward()

    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}
    grads = {k : v.value for k, v in grads.items()}

    if patch is None:
        if dictionaries:
            hidden_states_patch = {
                k : DenseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
            }
        else:
            hidden_states_patch = {
                k : t.zeros_like(v) for k, v in hidden_states_clean.items()
            }
        total_effect = t.tensor(float('-inf'))
    else:
        hidden_states_patch = {}
        with model.trace(patch, scan=True), t.inference_mode():
            for submodule in submodules:
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]

                if not submodule in dictionaries:
                    hidden_states_patch[submodule] = x.save()
                    continue

                dictionary = dictionaries[submodule]
                flat_f = nnsight.apply(dictionary.forward, x.flatten(0, 1))
                f = ForwardOutput(
                    latent_acts=flat_f.latent_acts.reshape(x.shape[0], x.shape[1], -1),
                    latent_indices=flat_f.latent_indices.reshape(x.shape[0], x.shape[1], -1),
                    sae_out=flat_f.sae_out.reshape(x.shape[0], x.shape[1], -1),
                    fvu=flat_f.fvu,
                    auxk_loss=flat_f.auxk_loss,
                    multi_topk_fvu=flat_f.multi_topk_fvu
                )
                dense = to_dense(f.latent_acts, f.latent_indices, num_latents)
                x_hat = dense @ dictionary.W_dec.mT.T
                residual = x - x_hat
                dense = to_dense(f.latent_acts, f.latent_indices, num_latents)
                hidden_states_patch[submodule] = DenseAct(act=dense, res=residual).save()
            metric_patch = metric_fn(model.output.logits, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        patch_state, clean_state, grad = hidden_states_patch[submodule], hidden_states_clean[submodule], grads[submodule]
        delta = patch_state - clean_state.detach() if patch_state is not None else -clean_state.detach()
        if dictionaries:
            effect = delta @ grad
        else:
            effect = delta * grad
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    total_effect = total_effect if total_effect is not None else None
    
    return EffectOut(effects, deltas, grads, total_effect)


def _pe_ig(
        clean,
        patch,
        model,
        submodules,
        dictionaries: dict[Any, Sae],
        metric_fn,
        steps=10,
        metric_kwargs=dict(),
):
    if dictionaries:
        num_latents = next(iter(dictionaries.values())).num_latents
    # first run through the fake inputs to figure out which hidden states are tuples
    is_tuple = {}
    with model.scan("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    hidden_states_clean = {}
    # scan=True must be set; otherwise faketensor shapes aren't populated and 
    # the reshape code will produce incorrect results (FAIL SILENTLY)
    with model.trace(clean, scan=True) as tracer, t.no_grad():
        for submodule in submodules:
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]

            if not submodule in dictionaries:
                hidden_states_clean[submodule] = x.save()
                continue

            dictionary = dictionaries[submodule]
            flat_f = nnsight.apply(dictionary.forward, x.flatten(0, 1))
            f = ForwardOutput(
                latent_acts=flat_f.latent_acts.reshape(x.shape[0], x.shape[1], -1),
                latent_indices=flat_f.latent_indices.reshape(x.shape[0], x.shape[1], -1),
                sae_out=flat_f.sae_out.reshape(x.shape[0], x.shape[1], -1),
                fvu=flat_f.fvu,
                auxk_loss=flat_f.auxk_loss,
                multi_topk_fvu=flat_f.multi_topk_fvu
            )

            dense = to_dense(f.latent_acts, f.latent_indices, num_latents)
            x_hat = dense @ dictionary.W_dec.mT.T
            x_res = x - x_hat

            hidden_states_clean[submodule] = DenseAct(dense, x_res).save()
        
        metric_clean = metric_fn(model.output.logits, **metric_kwargs).save()

    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}
    if patch is None:
        if dictionaries:
            hidden_states_patch = {
                k : DenseAct( 
                    act=t.zeros_like(v.act), 
                    res=t.zeros_like(v.res)
                ) for k, v in hidden_states_clean.items()
            }
        else:
            hidden_states_patch = {
                k : t.zeros_like(v) for k, v in hidden_states_clean.items()
            }
        # TODO better way of handling zero ablation than this - do a zero run and record metric
        total_effect = t.tensor(float('-inf'))
    else:
        hidden_states_patch = {}
        with model.trace(patch, scan=True), t.no_grad():
            for submodule in submodules:
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]

                if not submodule in dictionaries:
                    hidden_states_patch[submodule] = x.save()
                    continue

                dictionary = dictionaries[submodule]
                flat_f = nnsight.apply(dictionary.forward, x.flatten(0, 1))
                f = ForwardOutput(
                    latent_acts=flat_f.latent_acts.reshape(x.shape[0], x.shape[1], -1),
                    latent_indices=flat_f.latent_indices.reshape(x.shape[0], x.shape[1], -1),
                    sae_out=flat_f.sae_out.reshape(x.shape[0], x.shape[1], -1),
                    fvu=flat_f.fvu,
                    auxk_loss=flat_f.auxk_loss,
                    multi_topk_fvu=flat_f.multi_topk_fvu
                )
                dense = to_dense(f.latent_acts, f.latent_indices, num_latents)
                x_hat = dense @ dictionary.W_dec.mT.T
                x_res = x - x_hat

                hidden_states_patch[submodule] = DenseAct(dense, x_res).save()

            metric_patch = metric_fn(model.output.logits, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule] if submodule in dictionaries else None
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        with model.trace() as tracer:
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state

                if dictionary:
                    f.act.retain_grad()
                    f.res.retain_grad()
                else:
                    f.retain_grad()
                fs.append(f)

                with tracer.invoke(clean):
                    if is_tuple[submodule]:
                        if dictionary is None:
                            submodule.output[0][:] = f
                        else:
                            submodule.output[0][:] = (f.act @ dictionary.W_dec.mT.T) + f.res
                    else:
                        if dictionary is None:
                            submodule.output = f
                        else:
                            submodule.output = (f.act @ dictionary.W_dec.mT.T) + f.res

            # Create gradients for the act and residual
            metric = metric_fn(model.output.logits, **metric_kwargs, repeat=10)
            metric.sum().backward(retain_graph=True)

        if dictionary:
            mean_grad = sum([f.act.grad for f in fs]) / steps
            mean_residual_grad = sum([f.res.grad for f in fs]) / steps
            grad = DenseAct(act=mean_grad, res=mean_residual_grad)
        else:
            grad = f.grad

        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        if not dictionary:
            effect = grad * delta
        else:
            effect = grad @ delta
           
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return EffectOut(effects, deltas, grads, total_effect)


def _pe_exact(
    clean,
    patch,
    model,
    submodules,
    dictionaries,
    metric_fn,
    ):
    num_latents = next(iter(dictionaries.values())).num_latents
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.scan("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.inference_mode():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = nnsight.apply(dictionary.encode, x)
            x_hat = nnsight.apply(dictionary.decode, f.top_acts, f.top_indices)
            residual = x - x_hat
            dense = to_dense(f.top_acts, f.top_indices, num_latents)
            hidden_states_clean[submodule] = DenseAct(act=dense, res=residual).save()
        metric_clean = metric_fn(model.output.logits).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k : DenseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = t.tensor(float('-inf'))
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.inference_mode():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                f = nnsight.apply(dictionary.encode, x)
                x_hat = nnsight.apply(dictionary.decode, f.top_acts, f.top_indices)
                residual = x - x_hat
                dense = to_dense(f.top_acts, f.top_indices, num_latents)
                hidden_states_patch[submodule] = DenseAct(act=dense, res=residual).save()
            metric_patch = metric_fn(model.output.logits).save()
        total_effect = metric_patch.value - metric_clean.value
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        effect = DenseAct(act=t.zeros_like(clean_state.act), resc=t.zeros(*clean_state.res.shape[:-1])).to(model.device)
        
        # iterate over positions and features for which clean and patch differ
        idxs = t.nonzero(patch_state.act - clean_state.act)
        for idx in tqdm(idxs):
            with t.inference_mode():
                with model.trace(clean, **tracer_kwargs):
                    f = clean_state.act.clone()
                    f[tuple(idx)] = patch_state.act[tuple(idx)]
                    x_hat = (f @ dictionary.W_dec.mT.T) + clean_state.res.clone()
                    # x_hat = nnsight.apply(dictionary.decode, f.top_acts, f.top_indices)
                    if is_tuple[submodule]:
                        submodule.output[0][:] = x_hat + clean_state.res
                    else:
                        submodule.output = x_hat + clean_state.res
                    metric = metric_fn(model.output.logits).save()
                effect.act[tuple(idx)] = (metric.value - metric_clean.value).sum()

        for idx in list(ndindex(effect.resc.shape)):
            with t.inference_mode():
                with model.trace(clean, **tracer_kwargs):
                    res = clean_state.res.clone()
                    res[tuple(idx)] = patch_state.res[tuple(idx)]
                    x_hat = dictionary.decode(clean_state.act)
                    if is_tuple[submodule]:
                        submodule.output[0][:] = x_hat + res
                    else:
                        submodule.output = x_hat + res
                    metric = metric_fn(model.output.logits).save()
                effect.resc[tuple(idx)] = (metric.value - metric_clean.value).sum()
        
        effects[submodule] = effect
        deltas[submodule] = patch_state - clean_state
    total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, None, total_effect)

def patching_effect(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        method='attrib',
        steps=10,
        metric_kwargs=dict()
):
    if method == 'attrib':
        return _pe_attrib(clean, patch, model, submodules, dictionaries, metric_fn, metric_kwargs=metric_kwargs)
    elif method == 'ig':
        return _pe_ig(clean, patch, model, submodules, dictionaries, metric_fn, steps=steps, metric_kwargs=metric_kwargs)
    elif method == 'exact':
        return _pe_exact(clean, patch, model, submodules, dictionaries, metric_fn)
    else:
        raise ValueError(f"Unknown method {method}")

def jvp(
        input,
        model,
        dictionaries: dict[Any, Sae],
        downstream_submod,
        downstream_features,
        upstream_submod,
        left_vec : Union[DenseAct, Dict[int, DenseAct]],
        right_vec : DenseAct,
        return_without_right = False,
):
    """
    Return a sparse shape [# downstream features + 1, # upstream features + 1] tensor of Jacobian-vector products.
    """
    if not downstream_features: # handle empty list
        if not return_without_right:
            return t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(model.device)
        else:
            return t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(model.device), t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(model.device)

    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.scan("_"):
        is_tuple[upstream_submod] = type(upstream_submod.output.shape) == tuple
        is_tuple[downstream_submod] = type(downstream_submod.output.shape) == tuple

    downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]

    vjv_indices = {}
    vjv_values = {}
    if return_without_right:
        jv_indices = {}
        jv_values = {}

    with model.trace(input, scan=True) as tracer:
        # first specify forward pass modifications
        x = upstream_submod.output
        if is_tuple[upstream_submod]:
            x = x[0]

        flat_f = nnsight.apply(upstream_dict.forward, x.flatten(0, 1))
        f = ForwardOutput(
            latent_acts=flat_f.latent_acts.reshape(x.shape[0], x.shape[1], -1),
            latent_indices=flat_f.latent_indices.reshape(x.shape[0], x.shape[1], -1),
            sae_out=flat_f.sae_out.reshape(x.shape[0], x.shape[1], -1),
            fvu=flat_f.fvu,
            auxk_loss=flat_f.auxk_loss,
            multi_topk_fvu=flat_f.multi_topk_fvu
        )

        dense = to_dense(f.latent_acts, f.latent_indices, upstream_dict.num_latents)
        x_hat = dense @ upstream_dict.W_dec.mT.T
        x_res = x - x_hat # grad present
        upstream_act = DenseAct(act=dense, res=x_res).save()
        if is_tuple[upstream_submod]:
            upstream_submod.output[0][:] = x_hat + x_res
        else:
            upstream_submod.output = x_hat + x_res
        y = downstream_submod.output
        if is_tuple[downstream_submod]:
            y = y[0]

        flat_f = nnsight.apply(downstream_dict.forward, y.flatten(0, 1))
        f = ForwardOutput(
            latent_acts=flat_f.latent_acts.reshape(x.shape[0], x.shape[1], -1),
            latent_indices=flat_f.latent_indices.reshape(x.shape[0], x.shape[1], -1),
            sae_out=flat_f.sae_out.reshape(x.shape[0], x.shape[1], -1),
            fvu=flat_f.fvu,
            auxk_loss=flat_f.auxk_loss,
            multi_topk_fvu=flat_f.multi_topk_fvu
        )
        g = to_dense(f.latent_acts, f.latent_indices, downstream_dict.num_latents).save()
        y_hat = g @ downstream_dict.W_dec.mT.T
        y_res = y - y_hat
        downstream_act = DenseAct(act=g, res=y_res).save()

        for downstream_feat in downstream_features:
            if isinstance(left_vec, DenseAct):
                to_backprop = (left_vec @ downstream_act).to_tensor().flatten()
            elif isinstance(left_vec, dict):
                to_backprop = (left_vec[downstream_feat] @ downstream_act).to_tensor().flatten()
            else:
                raise ValueError(f"Unknown type {type(left_vec)}")
        
            vjv = (upstream_act.grad @ right_vec).to_tensor().flatten()
            if return_without_right:
                jv = (upstream_act.grad @ right_vec).to_tensor().flatten()
            x_res.grad = t.zeros_like(x_res)
            to_backprop[downstream_feat].backward(retain_graph=True)

            vjv_indices[downstream_feat] = vjv.nonzero().squeeze(-1).save()
            vjv_values[downstream_feat] = vjv[vjv_indices[downstream_feat].save()].save()

            if return_without_right:
                jv_indices[downstream_feat] = jv.nonzero().squeeze(-1).save()
                jv_values[downstream_feat] = jv[vjv_indices[downstream_feat]].save()

    
    d_downstream_contracted = len((downstream_act.value @ downstream_act.value).to_tensor().flatten())
    d_upstream_contracted = len((upstream_act.value @ upstream_act.value).to_tensor().flatten())
    if return_without_right:
        d_upstream = len(upstream_act.value.to_tensor().flatten())
    
    vjv_indices = t.tensor(
        [
            [downstream_feat for downstream_feat in downstream_features for _ in vjv_indices[downstream_feat].value],
            t.cat([vjv_indices[downstream_feat].value for downstream_feat in downstream_features], dim=0)
        ]
    ).to(model.device)
    vjv_values = t.cat([vjv_values[downstream_feat].value for downstream_feat in downstream_features], dim=0)

    if not return_without_right:
        return t.sparse_coo_tensor(vjv_indices, vjv_values, (d_downstream_contracted, d_upstream_contracted))

    jv_indices = t.tensor(
        [[downstream_feat for downstream_feat in downstream_features for _ in jv_indices[downstream_feat].value],
         t.cat([jv_indices[downstream_feat].value for downstream_feat in downstream_features], dim=0)]
    ).to(model.device)
    jv_values = t.cat([jv_values[downstream_feat].value for downstream_feat in downstream_features], dim=0)

    return (
        t.sparse_coo_tensor(vjv_indices, vjv_values, (d_downstream_contracted, d_upstream_contracted)),
        t.sparse_coo_tensor(jv_indices, jv_values, (d_downstream_contracted, d_upstream))
    )