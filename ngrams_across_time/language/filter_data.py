from pathlib import Path
from argparse import ArgumentParser

import torch
from torch import Tensor
import pandas as pd
from datasets import load_from_disk, Dataset
from ngrams_across_time.utils.tensor_db import TensorDatabase


def top_k(matrix: Tensor, k=100):
    values, indices = torch.topk(matrix.flatten(), k)
    
    rows = indices // matrix.shape[1]
    cols = indices % matrix.shape[1]
    
    return list(zip(rows.tolist(), cols.tolist(), values.tolist()))


def filter_data(
        model: str,
        db_path: Path,
        start: int,
        end: int,
        n: int,
        ds_len: int = 1024,
): 
    # Arbitrarily chosen hyperparameter
    k = 2048

    db = TensorDatabase(str(db_path / "tensor_db"), str(db_path / "tensors"))

    def kl_div_delta(start: int, end: int, n: int):
        kl_start = db.query_last(model=model, step=start, metric='kl', ngram=n)
        kl_end = db.query_last(model=model, step=end, metric='kl', ngram=n)
        if kl_start is None or kl_end is None:
            raise ValueError(f"Could not find KL divergence data for model {model} at steps {start} and {end}")
        
        return kl_start['tensor'] - kl_end['tensor']

    kl_delta = kl_div_delta(start, end, n=n)[:ds_len]
    higher_kl_delta = kl_div_delta(start, end, n + 1)[:ds_len]

    start_loss = db.query_last(model=model, step=start, metric='loss')
    end_loss = db.query_last(model=model, step=end, metric='loss')
    val_set: Dataset = load_from_disk("data/val_tokenized.hf").select(range(ds_len)) # type: ignore

    filtered = []
    # Select the sequences with the largest drops in KL divergence
    for (row, col, kl_div_reduction) in top_k(kl_delta, k):
        higher_kl_div_reduction = higher_kl_delta[row, col].item()

        # Filter to items where the next highest order n-gram's KL divergence doesn't drop
        if higher_kl_div_reduction <= 0 and kl_div_reduction > 0:
            data = []
            # print(f"{row} {col}: n-gram KL div: {kl_div_reduction}, higher order n-gram KL div: {higher_kl_div_reduction}")
            ngram = val_set[row]['input_ids'][col - n + 1: col + 1].tolist()
            data.extend([row, (col - n + 1), col, kl_div_reduction, higher_kl_div_reduction, ngram])
            if start_loss: data.append(start_loss['tensor'][row, col].item())
            if end_loss: data.append(end_loss['tensor'][row, col].item())

            filtered.append(data)

    df = pd.DataFrame(filtered, columns=[
         "sample_idx", 
         "start_ngram_idx",
         "end_ngram_idx", 
         "kl_div_reduction", 
         "higher_order_kl_div_reduction",
         "ngram"
        ] + (
            ["start_loss"] if start_loss else []
        ) + (
            ["end_loss"] if end_loss else []
        )
    )
    df.to_csv(f"filtered-{n}-gram-data-{model.replace('/', '--')}-{start}-{end}.csv", index=False)


def parse_args():
    parser = ArgumentParser(description='Select data points that evolve between two checkpoints')
    
    group = parser.add_argument_group("Model arguments")
    group.add_argument('--model', type=str, default="EleutherAI/pythia-160m", help='Model name')
    group.add_argument('--start', type=int, required=True, help='Start checkpoint')
    group.add_argument('--end', type=int, required=True, help='End checkpoint')
    group.add_argument('--n', type=int, default=2, help='n-grams to gather data for')
    
    group = parser.add_argument_group("Data arguments")
    group.add_argument('--db', type=str, default="/mnt/ssd-1/tensor_db", help='Location to save data in SQLite database')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_data(args.model, Path(args.db), start=args.start, end=args.end, n=args.n)