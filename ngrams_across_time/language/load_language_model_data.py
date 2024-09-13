import os
from typing import Dict

from pathlib import Path

from ngrams_across_time.utils.utils import assert_type

import torch
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from transformer_lens import HookedTransformer

from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.data import PromptDataset

from ngrams_across_time.utils.data import ZippedDataset
from ngrams_across_time.language.language_data_types import NgramDataset
from ngrams_across_time.language.hf_client import get_model_checkpoints, get_pythia_model_size, load_with_retries, with_retries


def load_token_data(max_ds_len: int = 1024):
    source_dataset_path = '/mnt/ssd-1/lucia/ngrams-across-time/data/val_tokenized.hf'
    val: Dataset = load_from_disk(source_dataset_path) # type: ignore
    val = val.select(range(max_ds_len))
    val.set_format("torch", columns=["input_ids"])
    return val


def get_ngram_dataset(
        model_name: str, 
        start: int, 
        end: int, 
        order: int, 
        vocab_size: int, 
        max_ds_len: int = 1024, 
        patchable: bool = False
):
    if patchable:
        return get_patchable_ngram_dataset(model_name, start, end, order)
    else:
        orders = [order - 1, order, order + 1]
        ngram_dists = get_ngram_dists(vocab_size, orders, max_ds_len)
        return ZippedDataset(
            target_dataset=ngram_dists[order],
            low_order_dataset=ngram_dists[order - 1],
            high_order_dataset=ngram_dists[order + 1],
            base_dataset=load_token_data(max_ds_len),
        )

def get_ngram_dists(vocab_size: int, orders: list[int], max_ds_len: int = 1024):

    ngram_data = {
        i: NgramDataset(
            Path(f'/mnt/ssd-1/lucia/ngrams-across-time/data/smoothed-{i}-gram-pile-dists-bf16-2_shards.npy'), 
            vocab_size, 
            max_ds_len,
            i
        )
        for i in orders
    }
    
    return ngram_data

def get_patchable_ngram_dataset(model_name: str, start: int, end: int, order: int) -> Dict[str, PromptDataset]:
    dataset_name = f"{order}-grams-{start}-{end}-{model_name.replace('/', '--')}"
    if not Path(dataset_name).exists():
        f"Prompt dataset not found - to generate use run_collect_and_filter.py --modality language --start {start} --end {end} --order {order} --model_name {model_name}"
    dataset = load_from_disk(dataset_name) # type: ignore
    dataset = assert_type(Dataset, dataset)

    patch_dict = {}

    for row in dataset:
        alternative_ngram_prefixes = torch.tensor(row['low_order'])[:, :-1] # type: ignore

        learned_ngram = torch.tensor(row['target']) # type: ignore
        learned_ngram_prefix_repeated = [learned_ngram[:-1] for _ in range(len(alternative_ngram_prefixes))]
        learned_ngram_suffix_repeated = [learned_ngram[-1:] for _ in range(len(alternative_ngram_prefixes))]
        
        patch_dict[learned_ngram] = PromptDataset(
            learned_ngram_prefix_repeated,
            alternative_ngram_prefixes,
            learned_ngram_suffix_repeated,
            learned_ngram_suffix_repeated
        )

    return patch_dict

def get_models(
        model_name: str,
        start: int = 0,
        end: int = 16384,
        patchable: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        order: int = 2,
):
    # Get checkpoints
    checkpoints = get_model_checkpoints(model_name)
    if not checkpoints:
        raise ValueError(f'No checkpoints found for {model_name}')
    
    ranged_checkpoints = {k: v for k, v in checkpoints.items() if (start is None or k >= start) and (end is None or k <= end)}
    ranged_checkpoints = dict(sorted(ranged_checkpoints.items()))
    
    # Load model
    model_size = get_pythia_model_size(model_name)
    models = {}
    for step, revision in ranged_checkpoints.items():
        model = load_with_retries(model_name, revision, model_size)
        if model:
            if patchable:
                def load():
                    return HookedTransformer.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        revision=revision,
                        cache_dir=".cache"
                    ).cuda()

                model: HookedTransformer = with_retries(load) # type: ignore
                model.set_use_attn_result(True)
                model.set_use_hook_mlp_in(True)
                model.set_use_split_qkv_input(True)

                models[step] = patchable_model(model, factorized=True, device=device, separate_qkv=True, seq_len=order - 1, slice_output="last_seq")
            else:
                models[step] = model


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)

    return models, vocab_size