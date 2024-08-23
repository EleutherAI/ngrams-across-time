from argparse import ArgumentParser
import math
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from ngrams_across_time.language.ngram_datasets import get_ngram_datasets
from ngrams_across_time.language.hf_client import get_model_checkpoints, get_basic_pythia_model_names, get_pythia_model_size, load_with_retries

from ngrams_across_time.utils.divergences import kl_divergence_log_space, js_divergence_log_space
from ngrams_across_time.utils.tensor_db import TensorDatabase

# TODO pass val and ngrams separately
def process_datasets(
        revisions: dict[int, str], 
        model_name: str, 
        data: dict[str | int, DataLoader], 
        vocab_size: int,
        db_path: Path, 
        key='input_ids',
    ):

    # Initialise results structure
    result = {
        ngram: {
            step: {
                'kl': torch.zeros(1024, 2049),
                'js': torch.zeros(1024, 2049),
            }
            for step in revisions.keys()
        }
        for ngram in data.keys() if isinstance(ngram, int)
    }
    loss = torch.zeros(1024, 2048)

    # Collect data
    for step, revision in revisions.items():
        model_size = get_pythia_model_size(model_name)
        model = load_with_retries(model_name, revision, model_size)
        if not model:
            continue

        ngram_iters = {k: iter(data[k]) for k in result.keys()}     
        for j, batch in tqdm(enumerate(data['val']), total=1024):
            tokens = batch[key].cuda()
            batch_size = len(tokens)
            logits = model(tokens).logits[:, :, :vocab_size]

            if loss is not None: 
                batch_loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), tokens[:, 1:].reshape(-1), reduction='none')
                loss[
                    j * batch_size : (j * batch_size) + batch_size
                ] = batch_loss.detach().cpu().reshape(batch_size, -1)
                
            del batch, tokens
            
            for n, loader in ngram_iters.items():
                try:
                    ngram_logprobs = next(loader).cuda()
                    result[n][step]['kl'][
                        j * batch_size : (j * batch_size) + batch_size
                    ] = kl_divergence_log_space(logits, ngram_logprobs, dim=-1).detach().cpu()
                    result[n][step]['js'][
                        j * batch_size : (j * batch_size) + batch_size
                    ] = js_divergence_log_space(logits, ngram_logprobs, dim=-1).detach().cpu()
                    
                    del ngram_logprobs
                except StopIteration:
                    print(f"Loader {loader} is exhausted")
    
    # Save results
    db = TensorDatabase(str(db_path / "tensor_db"), str(db_path / "tensors"))
    for ngram, ngram_data in result.items():
        for step, metrics in ngram_data.items():
            for metric, tensor in metrics.items():
                tags = {
                    'model': model_name,
                    'revision': revisions[step],
                    'step': step,
                    'metric': metric,
                    'ngram': ngram,
                    'num-shards': 2,
                }
                db.add_tensor(tensor, tags)
    tags = {
        'model': model_name,
        'revision': revisions[step],
        'step': step,
        'metric': 'loss',
    }
    db.add_tensor(loss, tags)
    db.close()


def parse_args():
    parser = ArgumentParser(description='Search for interesting data points between contiguous checkpoints')
    
    group = parser.add_argument_group("Model arguments")
    group.add_argument('--model', type=str, nargs='+', help='Model names')
    group.add_argument('--start', type=int, required=True, help='Start checkpoint')
    group.add_argument('--end', type=int, required=True, help='End checkpoint')
    group.add_argument('--include_intermediate', action='store_true', help='Include intermediate checkpoints')
    group.add_argument('--batch_size', type=int, default=1)
    group.add_argument('--n', type=int,nargs='+', default=[2], help='n-grams to gather data for')
    
    group = parser.add_argument_group("Data arguments")
    group.add_argument('--out', type=str, default="/mnt/ssd-1/tensor_db", help='Location to save data in SQLite database')
    
    return parser.parse_args()

def collect_model_data(
        model_name: str,
        start: int,
        end: int,
        batch_size: int,
        n: list[int],
        db_path: Path,
        include_intermediate: bool = False,
        max_ds_len: int = 1024, # debug parameter
):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # This is correct; tokenizer.vocab_size is incorrect.
    vocab_size = len(tokenizer)

    # Load n-gram model distributions
    data: dict[str | int, DataLoader] = {
        k: DataLoader(v, batch_size=batch_size, shuffle=False)
        for k, v in get_ngram_datasets(vocab_size, ngrams=n, max_ds_len=max_ds_len).items()
    }

    checkpoints = get_model_checkpoints(model_name)
    if not checkpoints:
        raise ValueError('No checkpoints found')
    ranged_checkpoints = {k: v for k, v in checkpoints.items() if start <= k <= end}
    ranged_checkpoints = dict(sorted(ranged_checkpoints.items()))
    if not include_intermediate:
        keys = list(ranged_checkpoints.keys())
        ranged_checkpoints = {start: ranged_checkpoints[keys[0]], end: ranged_checkpoints[keys[-1]]}
    print(ranged_checkpoints)
    
    process_datasets(ranged_checkpoints, model_name, data, vocab_size=vocab_size, db_path=db_path)

    
def main():
    max_ds_len = 1024 # debugging parameter
    mp.set_start_method('spawn', force=True)

    args = parse_args()
    db_path = Path(args.out)

    model_names = args.model or get_basic_pythia_model_names()
    for model_name in model_names:
        collect_model_data(
            model_name,
            args.start,
            args.end,
            args.batch_size,
            args.n,
            db_path,
            args.include_intermediate,
            max_ds_len,
        )

if __name__ == '__main__':
    main()
