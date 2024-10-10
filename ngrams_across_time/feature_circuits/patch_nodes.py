from collections import namedtuple

import torch;
import torch as t
from typing import Any
from sae import Sae
from sae.sae import ForwardOutput
import nnsight
from ngrams_across_time.feature_circuits.dense_act import DenseAct
from ngrams_across_time.feature_circuits.dense_act import DenseAct, to_dense


# Patch the specified nodes of the model SAEs and get metric values
def patch_nodes(
        clean,
        patch,
        model,
        submodules,
        dictionaries: dict[Any, Sae],
        metric_fn,
        nodes,
        metric_kwargs=dict(),
):
    num_latents = next(iter(dictionaries.values())).num_latents if dictionaries else None
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
        metric_patch = t.tensor(float('-inf'))
    else:
        hidden_states_patch = {}
        with model.trace(patch, scan=True), t.no_grad():
            for submodule in submodules:
                x = submodule.output
                if is_tuple[submodule]:
                    # MLP activations are the same at every position
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
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    with model.trace(clean, scan=True):
        for submodule in submodules:
            dictionary = dictionaries[submodule] if submodule in dictionaries else None
            clean_state = hidden_states_clean[submodule]
            patch_state = hidden_states_patch[submodule]
            
            alpha_complement = None
            if dictionaries:
                alpha_act = torch.zeros_like(clean_state.act)
                alpha_res = torch.zeros_like(clean_state.res)

                if submodule.path in nodes:
                    feature_indices = [idx for score, idx in nodes[submodule.path]]
                    alpha_act[:, :, [idx for idx in feature_indices if idx != -1]] = 1.
                    if -1 in feature_indices:
                        alpha_res[:] = 1.
                
                alpha = DenseAct(alpha_act, alpha_res)
                alpha_complement = DenseAct(torch.ones_like(alpha_act) - alpha_act, torch.ones_like(alpha_res) - alpha_res)
            else:
                alpha = torch.zeros_like(clean_state)
                if submodule.path in nodes:
                    feature_indices = [idx for score, idx in nodes[submodule.path]]
                    alpha[:, :, [idx for idx in feature_indices if idx != -1]] = 1.
                alpha_complement = torch.ones_like(alpha) - alpha

            masked_acts = (alpha_complement * clean_state + alpha * patch_state)

            if is_tuple[submodule]:
                if dictionary is None:
                    submodule.output[0][:] = masked_acts
                else:
                    submodule.output[0][:] = (masked_acts.act @ dictionary.W_dec.mT.T) + masked_acts.res
            else:
                if dictionary is None:
                    submodule.output = masked_acts
                else:
                    submodule.output = (masked_acts.act @ dictionary.W_dec.mT.T) + masked_acts.res

        metric_patched = metric_fn(model.output.logits, **metric_kwargs).save()

    return metric_patched, metric_patch, metric_clean
