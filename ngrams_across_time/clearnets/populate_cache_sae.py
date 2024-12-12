# python -m ngrams_across_time.clearnets.populate_cache_sae --dataset_repo "roneneldan/TinyStories" --dataset_split "train[:2%]" --dataset_row "text" --n_tokens 10_000_000

from nnsight import NNsight
from simple_parsing import ArgumentParser
import torch
# from sae_auto_interp.autoencoders import load_eai_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data
from transformers import AutoTokenizer
from ngrams_across_time.clearnets.sparse_mlp.sparse_gptneox_tinystories import TinyStoriesModel
from ngrams_across_time.clearnets.sparse_mlp.load_saes import load_eai_autoencoders

import os

def main(
        cfg: CacheConfig, 
        args,
        model_name: str = "dense-8m-max-e=200-esp=15-s=42", # "sparse-8m-max-e=200-esp=15-s=42"
        sae_dir: str = "/mnt/ssd-1/lucia/ngrams-across-time/sae/Dense TinyStories8M s=42 epoch 38"
    ): 
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained("data/tinystories/restricted_tokenizer_v2")
    ckpt_path = f'data/tinystories/{model_name}/checkpoints/last.ckpt'
    ptl_model = TinyStoriesModel.load_from_checkpoint(
        ckpt_path,
        dense=True,
        tokenizer=tokenizer
    )
    model = ptl_model.model
    model.to(device='cuda')
    # I believe dispatch won't work for tinystories models
    model = NNsight(ptl_model.model, device_map="auto", torch_dtype=torch.bfloat16, tokenizer=tokenizer) # dispatch=False
    model.tokenizer = tokenizer
     
    # Hacked in two places:
    # submodule = f"transformer.h.{layer}.mlp"
    # submodule = model.transformer.h[layer].mlp
    submodule_dict,model = load_eai_autoencoders(
        model,
        [0,1,2,3,4,5,6,7],
        sae_dir,
        module="mlp",
    )

    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
        cfg.dataset_name,
        dataset_row=cfg.dataset_row,
    )
    print(submodule_dict)
    cache = FeatureCache(
        model, 
        submodule_dict, 
        batch_size=cfg.batch_size,
    )
    name = "SAE"
    name += f"-{args.size}"
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
        model_name=model_name
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    #ctx len 256
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--size", type=str, default="8M")
    args = parser.parse_args()
    cfg = args.options
    
    main(cfg, args)