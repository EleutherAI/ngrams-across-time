# python -m ngrams_across_time.clearnets.train.autointerp_cache --dataset_repo "roneneldan/TinyStories" --dataset_split "train[:1%]" --dataset_row "text" --n_tokens 10_000_000

# --train_type "quantiles" --n_examples_train 40 --n_quantiles 10 --width 24576 
import os

from nnsight import NNsight
from simple_parsing import ArgumentParser
import torch
from transformers import AutoTokenizer
from sae_auto_interp.autoencoders import load_eai_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data
from sae.sae import Sae
from sae.config import SaeConfig
from typing import Any, Tuple, Dict
from ngrams_across_time.clearnets.train.sparse_gptneox_tinystories import TinyStoriesModel

def load_sparse_mlp_transformer_latents(
    model: Any,
) -> Tuple[Dict[str, Any], Any]:
    """
    Load hidden activations for specified layers and module.

    Args:
        model (Any): The model to load autoencoders for.
        layers (List[int]): List of layer indices to load autoencoders for.
        module (str): Module name ('mlp' or 'res').
        randomize (bool, optional): Whether to randomize the autoencoder. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[Dict[str, Any], Any]: A tuple containing the submodules dictionary and the edited model.
    """
    hook_to_d_in = resolve_widths(model, get_gptneo_hookpoints(model), torch.randint(0, 10000, (1, 1024)))

    submodules = {}

    for layer in range(len(model.transformer.h)):
        def _forward(x):
            return x

        hookpoint = f"transformer.h.{layer}.mlp"
        submodule = model.transformer.h[layer].mlp
        submodule.ae = AutoencoderLatents(
            None, _forward, width=hook_to_d_in[hookpoint] # type: ignore
        )
        submodules[submodule.path] = submodule

        hookpoint = f"transformer.h.{layer}.attn.attention"
        submodule = model.transformer.h[layer].attn
        submodule.ae = AutoencoderLatents(
            None, _forward, width=hook_to_d_in[hookpoint] # type: ignore
        )
        submodules[submodule.path] = submodule


    with model.edit("") as edited:
        for path, submodule in submodules.items():
            if "embed" not in path and "mlp" not in path:
                acts = submodule.output[0]
            else:
                acts = submodule.output
            submodule.ae(acts, hook=True)

    return submodules, edited

@torch.inference_mode()
def resolve_widths(
    model, module_names: list[str], inputs: torch.Tensor, 
    dims: list[int] = [-1],
) -> dict[str, int]:
    """Find number of output dimensions for the specified modules."""
    module_to_name = {
        model.get_submodule(name): name for name in module_names
    }
    shapes: dict[str, int] = {}

    def hook(module, _, output):
        # Unpack tuples if needed
        if isinstance(output, tuple):
            output, *_ = output

        name = module_to_name[module]

        pos_dims = [d if d >= 0 else output.ndim + d for d in dims]
        assert all(i in pos_dims for i in range(min(pos_dims), max(pos_dims) + 1)) and max(pos_dims) == output.ndim - 1, \
            f"Feature dimensions {dims} must be contiguous and include the final dimension"

        from math import prod
        shapes[name] = prod(output.shape[d] for d in dims)

    handles = [
        mod.register_forward_hook(hook) for mod in module_to_name
    ]
    dummy = inputs.to(model.device)
    try:
        model(dummy)
    finally:
        for handle in handles:
            handle.remove()
    
    return shapes

def get_gptneo_hookpoints(model):
    hookpoints = []
    for i in range(len(model.transformer.h)):
        hookpoints.append(f"transformer.h.{i}.attn.attention")
        hookpoints.append(f"transformer.h.{i}.mlp")
    return hookpoints


def main(cfg: CacheConfig, args): 
    size = '8'
    tokenizer = AutoTokenizer.from_pretrained("data/tinystories/restricted_tokenizer_v2")
    cpts_path = f'data/tinystories-{size}/checkpoints-5-dec/last.ckpt'
    
    # config = SparseGPTNeoConfig.from_pretrained(cpts_path)
    # model = SparseGPTNeoForCausalLM(config)
    ptl_model = TinyStoriesModel.load_from_checkpoint(
        cpts_path,
        dense=False,
        tokenizer=tokenizer
    )
    ptl_model.to(device='cuda')

    model = ptl_model.model
    shapes = resolve_widths(model, get_gptneo_hookpoints(model), torch.randint(0, 10000, (1, 1024)))

    
    model = NNsight(model, device_map="auto", torch_dtype=torch.bfloat16, tokenizer=tokenizer)
    model.tokenizer = tokenizer

    
    # I believe dispatch won't work for tinystories models
    # model = NNsight(model, device_map="auto", dispatch=True,torch_dtype=torch.bfloat16)
     
    submodule_dict,model = load_sparse_mlp_transformer_latents(model)

    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
        cfg.dataset_name,
        cfg.dataset_row,
    )
    print(submodule_dict)
    cache = FeatureCache(
        model, 
        submodule_dict, 
        batch_size=cfg.batch_size,
    )
    name = f"No-SAE"

    cache.run(cfg.n_tokens, tokens)
    breakpoint()
    
    print(f"Saving splits to {f'raw_features/{args.dataset_repo}/{name}'}")
    os.makedirs(f"raw_features/{args.dataset_repo}/{name}", exist_ok=True)
    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir=f"raw_features/{args.dataset_repo}/{name}"
    )
    print(f"Saving config to {f'raw_features/{args.dataset_repo}/{name}'}")
    cache.save_config(
        save_dir=f"raw_features/{args.dataset_repo}/{name}",
        cfg=cfg,
        model_name="TinyStories-8M-Sparse"
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    #ctx len 256
    parser.add_arguments(CacheConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options
    
    main(cfg, args)