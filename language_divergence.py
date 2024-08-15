from argparse import ArgumentParser
import math
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from src.language.ngram_data import get_ngram_datasets
from src.language.get_revisions import get_model_checkpoints
from src.language.pythia import get_basic_pythia_model_names, get_model_size, load_with_retries

from src.utils.divergences import kl_divergence_log_space, js_divergence_log_space
from src.utils.tensor_db import TensorDatabase

# TODO pass val and ngrams separately
def process_datasets(
        revisions: dict[int, str], 
        model_name: str, 
        data: dict[str | int, DataLoader], 
        vocab_size: int,
        out: Path, 
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

    # Collect data
    for step, revision in revisions.items():
        model_size = get_model_size(model_name)
        model = load_with_retries(model_name, revision, model_size)
        if not model:
            continue

        ngram_iters = {k: iter(data[k]) for k in result.keys()}     
        for j, batch in tqdm(enumerate(data['val']), total=1024):
            batch_size = len(batch[key])
            logits = model(batch[key].cuda()).logits[:, :, :vocab_size]
            
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
                
            del batch
    
    # Save results
    db = TensorDatabase(str(out / "tensor_db.sqlite"), str(out / "tensors"))
    for ngram, ngram_data in result.items():
        for step, metrics in ngram_data.items():
            for metric, tensor in metrics.items():
                tags = {
                    'model': model_name,
                    'revision': revisions[step],
                    'step': step,
                    'metric': metric,
                    'ngram': ngram,
                }
                db.add_tensor(tensor, tags)
    db.close()

def parse_args():
    parser = ArgumentParser(description='Search for interesting data points between contiguous checkpoints')
    
    group = parser.add_argument_group("Model arguments")
    group.add_argument('--model', type=str, nargs='+', help='Model names')
    group.add_argument('--tokenizer', type=str, help='Tokenizer name')
    group.add_argument('--start', type=int, required=True, help='Start checkpoint')
    group.add_argument('--end', type=int, required=True, help='End checkpoint')
    group.add_argument('--batch_size', type=int, default=1)
    group.add_argument('--n', type=int,nargs='+', default=1, help='n-grams to gather data for')
    
    group = parser.add_argument_group("Data arguments")
    group.add_argument('--out', type=str, default="/mnt/ssd-1/tensor_db", help='Location to save data in SQLite database')
    
    return parser.parse_args()

def main():
    mp.set_start_method('spawn', force=True)

    args = parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load n-gram model distributions
    # Commented out: Disgusting hack to get around data lock
    data: dict[str | int, DataLoader] = {
        k: DataLoader(v, batch_size=args.batch_size, shuffle=False) # type: ignore
        for k, v in get_ngram_datasets(tokenizer.vocab_size, ngrams=args.n).items()
    }
    # } for _ in range(torch.cuda.device_count())]

    # Inference over checkpoints
    model_names = args.model or get_basic_pythia_model_names()
    for model_name in model_names:
        checkpoints = get_model_checkpoints(model_name)
        if not checkpoints:
            raise ValueError('No checkpoints found')
        ranged_checkpoints = {k: v for k, v in checkpoints.items() if args.start <= k <= args.end}
        ranged_checkpoints = dict(sorted(ranged_checkpoints.items()))
        print(ranged_checkpoints)
        
        process_datasets(ranged_checkpoints, model_name, data, vocab_size=tokenizer.vocab_size, out=Path(args.out))

def run_workers(worker, revisions: list[int] | list[str], **kwargs) -> list[dict]:
    """Parallelise inference over model checkpoints."""
    max_revisions_per_chunk = math.ceil(len(revisions) / torch.cuda.device_count())
    step_indices = [
        revisions[i : i + max_revisions_per_chunk]
        for i in range(0, len(revisions), max_revisions_per_chunk)
    ]

    args = [
        (i, step_indices[i], *kwargs.values())
        for i in range(len(step_indices))
    ]
    print(f"Inference on revisions {step_indices}, GPU count: {len(step_indices)}")
    with mp.Pool(len(step_indices)) as pool:
        return pool.starmap(worker, args)


if __name__ == '__main__':
    main()
