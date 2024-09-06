from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset

from ngrams_across_time.utils.data import MultiOrderDataset
from ngrams_across_time.language.language_data_types import NgramDataset
from ngrams_across_time.language.hf_client import get_model_checkpoints, get_pythia_model_size, load_with_retries
from ngrams_across_time.language.ngram_datasets import get_ngram_datasets


def get_ngram_dataset(vocab_size: int, target_ngram: int, max_ds_len: int = 1024):
    ngrams = [target_ngram - 1, target_ngram, target_ngram + 1]
    source_dataset_path = '/mnt/ssd-1/lucia/ngrams-across-time/data/val_tokenized.hf'
    val: Dataset = load_from_disk(source_dataset_path) # type: ignore
    val = val.select(range(max_ds_len))
    val.set_format("torch", columns=["input_ids"])

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
        target_dataset=ngram_data[target_ngram],
        low_order_dataset=ngram_data[target_ngram - 1],
        high_order_dataset=ngram_data[target_ngram + 1],
        base_dataset=val,
    )

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    datasets = get_ngram_datasets(50_254, ngrams=[2, 3], max_ds_len=48)
    dataloader = DataLoader(datasets[1], batch_size=32, shuffle=False)

    for batch in dataloader:
        pass

    print("All operations completed successfully.")

def load_models_and_ngrams(
    model_name: str,
    target_ngram: int = 2,
    dataset_name: str = None,
    start: int = 0,
    end: int = 16384,
    max_ds_len: int = 1024,
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)
    
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
            models[step] = (tokenizer, model)
        
    dataset = get_ngram_dataset(vocab_size, target_ngram=target_ngram, max_ds_len=max_ds_len)

    return models, dataset
