# python -m ngrams_across_time.clearnets.autointerp_cache --dataset_repo "roneneldan/TinyStories" --dataset_split "train[:1%]" --dataset_row "text" --n_tokens 1_000_000

# python -m ngrams_across_time.clearnets.autointerp_explain --model "data/tinystories/sparse-8m-max-e=200-esp=15-s=42" --modules "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.0.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.1.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.2.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.3.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.4.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.5.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.7.mlp" --n_random 50 --n_examples_test 50 --train_type "quantiles" --n_examples_train 40 --n_quantiles 10 --width 8192


# python -m ngrams_across_time.clearnets.autointerp_explain --model "data/tinystories/dense-8m-max-e=200-esp=15-s=42" --modules "roneneldan/TinyStories/SAE/.transformer.h.0.mlp" "roneneldan/TinyStories/SAE/.transformer.h.1.mlp" "roneneldan/TinyStories/SAE/.transformer.h.2.mlp" "roneneldan/TinyStories/SAE/.transformer.h.3.mlp" "roneneldan/TinyStories/SAE/.transformer.h.4.mlp" "roneneldan/TinyStories/SAE/.transformer.h.5.mlp" "roneneldan/TinyStories/SAE/.transformer.h.6.mlp" "roneneldan/TinyStories/SAE/.transformer.h.7.mlp" --n_random 50 --n_examples_test 50 --train_type "quantiles" --n_examples_train 40 --n_quantiles 10 --width 8192
import os

from nnsight import NNsight
from simple_parsing import ArgumentParser
import torch
from transformers import AutoTokenizer
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data
from sae.sae import Sae
from sae.config import SaeConfig
from typing import Any, Tuple, Dict

from ngrams_across_time.clearnets.sparse_mlp.sparse_gptneox_tinystories import TinyStoriesModel
from ngrams_across_time.clearnets.inference.inference import to_dense


def load_dense_mlp_transformer_saes(model):
    pass

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
    # This doesn't matter because we don't use the width
    hook_to_d_in = resolve_widths(model, get_gptneo_hookpoints(model), torch.randint(0, 10000, (1, 1024)))

    submodules = {}

    for layer in range(len(model.transformer.h)):
        def _forward(x):
            return to_dense(x['top_acts'], x['top_indices'], num_latents=256 * 4 * 8) # hook_to_d_in[hookpoint])
            # return (x['top_indices'], x['top_acts'])

        hookpoint = f"transformer.h.{layer}.mlp"
        submodule = model.transformer.h[layer].mlp
        submodule.ae = AutoencoderLatents(
            None, _forward, width=256 * 4 * 8 # hook_to_d_in[hookpoint] # type: ignore
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
    model, module_names: list[str], inputs: torch.Tensor, dim = -1
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

        if isinstance(output, dict):
            output = output['hidden_states']

        name = module_to_name[module]

        shapes[name] = output.shape[dim]

    handles = [
        mod.register_forward_hook(hook) for mod in module_to_name
    ]
    dummy = inputs.to(model.device)
    try:
        model._model(dummy)
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
    tokenizer = AutoTokenizer.from_pretrained("data/tinystories/restricted_tokenizer_v2")
    ckpt_path = 'data/tinystories/sparse-8m-max-e=200-esp=15-s=42/checkpoints/last.ckpt'
    ptl_model = TinyStoriesModel.load_from_checkpoint(
        ckpt_path,
        dense=False,
        tokenizer=tokenizer
    )
    model = ptl_model.model
    model.to(device='cuda')
    # I believe dispatch won't work for tinystories models
    model = NNsight(ptl_model.model, device_map="auto", torch_dtype=torch.bfloat16, tokenizer=tokenizer) # dispatch=False
    model.tokenizer = tokenizer
     
    submodule_dict, model = load_sparse_mlp_transformer_latents(model)

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

    name = "sparse-feedfwd-transformer"

    cache.run(cfg.n_tokens, tokens)
    
    print(f"Saving splits to {f'raw_features/{cfg.dataset_repo}/{name}'}")
    os.makedirs(f"raw_features/{cfg.dataset_repo}/{name}", exist_ok=True)
    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir=f"raw_features/{cfg.dataset_repo}/{name}"
    )
    print(f"Saving config to {f'raw_features/{cfg.dataset_repo}/{name}'}")
    cache.save_config(
        save_dir=f"raw_features/{cfg.dataset_repo}/{name}",
        cfg=cfg,
        model_name="sparse-8m-max-e=200-esp=15-s=42"
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options
    cfg.tokenizer_or_model_name = "data/tinystories/restricted_tokenizer_v2"
    
    main(cfg, args)