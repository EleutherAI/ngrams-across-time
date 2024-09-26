import torch as t
from typing import Any
from sae import Sae
from sae.sae import ForwardOutput
import nnsight
from ngrams_across_time.feature_circuits.dense_act import to_dense


def sae_loss(
        clean,
        model,
        submodules,
        dictionaries: dict[Any, Sae],
        metric_fn,
        metric_kwargs=dict(),
):
    num_latents = next(iter(dictionaries.values())).num_latents
    # first run through the fake inputs to figure out which hidden states are tuples
    is_tuple = {}
    with model.scan("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    # scan=True must be set; otherwise faketensor shapes aren't populated and 
    # the reshape will produce incorrect results (FAIL SILENTLY)
    
    # Get loss with all SAEs and no residuals
    with model.trace(clean, scan=True), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            
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
            submodule.output = dense @ dictionary.W_dec.mT.T

        metric = metric_fn(model.output.logits, **metric_kwargs).save()

    # Get residual norm of each SAE output without interactions from other SAEs
    res = []
    with model.trace(clean, scan=True), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            
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
            res.append(x_res.norm().save())

    # average loss without residuals and with all dictionaries
    # average residual size
    return metric, t.stack(res).mean().item()


