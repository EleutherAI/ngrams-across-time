import torch as t
from typing import Any, Callable
from sae import Sae
from sae.sae import ForwardOutput
import nnsight
from nnsight import LanguageModel, NNsight
from nnsight.envoy import Envoy
from ngrams_across_time.feature_circuits.dense_act import to_dense


def sae_loss(
        clean: t.Tensor,
        model: LanguageModel | NNsight | Envoy,
        submodules: list,
        dictionaries: dict[Any, Sae],
        metric_fn: Callable,
        metric_kwargs=dict(),
        dummy_inputs: Any = "_"
):
    num_latents = next(iter(dictionaries.values())).num_latents
    # first run through the fake inputs to figure out which hidden states are tuples
    is_tuple = {}
    with model.scan(dummy_inputs) as tracer:
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    # scan=True must be set; otherwise faketensor shapes aren't populated and 
    # the reshape will produce incorrect results (FAIL SILENTLY)
    
    # Get loss with all SAEs and no residuals
    mses = []
    # breakpoint()
    with model.trace(clean, scan=True), t.no_grad(): # IndexError: too many indices for tensor of dimension 2
        for submodule in submodules:
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]

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
            submodule.output = dense @ dictionary.W_dec.mT.T
            
            x_res = x - submodule.output
            mses.append(t.mean(x_res**2).save())

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
    return metric, t.stack(res).mean().item(), t.stack(mses).mean().item()


