from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from transformer_lens import HookedTransformer

from auto_circuit.utils.graph_utils import patchable_model

from ngrams_across_time.utils.data import MultiOrderDataset
from ngrams_across_time.language.language_data_types import NgramDataset
from ngrams_across_time.language.hf_client import get_model_checkpoints, get_pythia_model_size, load_with_retries, with_retries
from ngrams_across_time.language.ngram_datasets import get_ngram_datasets


def load_token_data(max_ds_len: int = 1024):
    source_dataset_path = '/mnt/ssd-1/lucia/ngrams-across-time/data/val_tokenized.hf'
    val: Dataset = load_from_disk(source_dataset_path) # type: ignore
    val = val.select(range(max_ds_len))
    val.set_format("torch", columns=["input_ids"])
    return val

def get_ngram_examples(model_name: str, order: int, max_ds_len: int = 1024):
    prompts_dataset_name = f"/mnt/ssd-1/lucia/ngrams-across-time/filtered-{order}-gram-data-{model_name.replace('/', '--')}.csv"
    prompts_dataset = pd.read_csv(prompts_dataset_name)

    dataset = load_token_data(max_ds_len)

    prompts_original = [
        torch.tensor(dataset[int(row['sample_idx'])]['input_ids'][:int(row['end_token_idx']) + 1])
        for _, row in prompts_dataset.iterrows()
    ]
    prompts_shorter = [prompt[1:] for prompt in prompts_original]
    
    prompts_original = prompts_original[:1]
    prompts_shorter = prompts_shorter[:1]
    
    print(f"{len(prompts_original)} prompts loaded for each set")
    return prompts_original, prompts_shorter

def get_ngram_dataset(vocab_size: int, order: int, max_ds_len: int = 1024):
    ngrams = [order - 1, order, order + 1]
    val = load_token_data(max_ds_len)

    ngram_data = {
        i: NgramDataset(
            Path(f'/mnt/ssd-1/lucia/ngrams-across-time/data/smoothed-{i}-gram-pile-dists-bf16-2_shards.npy'), 
            vocab_size, 
            max_ds_len,
            i
        )
        for i in ngrams
    }
    
    return MultiOrderDataset(
        target_dataset=ngram_data[order],
        low_order_dataset=ngram_data[order - 1],
        high_order_dataset=ngram_data[order + 1],
        base_dataset=val,
    )

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    datasets = get_ngram_datasets(50_254, ngrams=[2, 3], max_ds_len=48)
    dataloader = DataLoader(datasets[1], batch_size=32, shuffle=False)

    for batch in dataloader:
        pass

    print("All operations completed successfully.")

def get_models(
        model_name: str,
        start: int = 0,
        end: int = 16384,
        patchable: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_seq_len: int = 10,
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

                models[step] = patchable_model(model, factorized=True, device=device, separate_qkv=True, seq_len=max_seq_len, slice_output="last_seq")
            else:
                models[step] = model


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)

    return models, vocab_size