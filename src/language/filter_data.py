from pathlib import Path
from argparse import ArgumentParser

import torch
from torch import Tensor
import pandas as pd

from src.utils.tensor_db import TensorDatabase


def bottom_k(matrix: Tensor, k=100):
    values, indices = torch.topk(matrix.flatten(), k, largest=False)
    
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

    db = TensorDatabase(str(db_path / "tensor_db.sqlite"), str(db_path / "tensors"))

    def kl_div_delta(start: int, end: int, n: int):
        kl_start = db.query_last(model=model, step=start, metric='kl', ngram=n)['tensor']
        kl_end = db.query_last(model=model, step=end, metric='kl', ngram=n)['tensor']
        return kl_start - kl_end

    kl_delta = kl_div_delta(start, end, n=n)[:ds_len]
    higher_kl_delta = kl_div_delta(start, end, n + 1)[:ds_len]

    filtered = []
    # Select the sequences with the largest drops in KL divergence
    for (row, col, kl_div_diff) in bottom_k(kl_delta, k):
        higher_kl_div_diff = higher_kl_delta[row, col].item()

        # Filter to items where the next highest order n-gram's KL divergence doesn't drop
        if higher_kl_div_diff >= 0 and kl_div_diff < 0:
            print(f"{row} {col}: n-gram KL div: {kl_div_diff}, higher order n-gram KL div: {higher_kl_div_diff}")
            filtered.append((row, col, kl_div_diff, higher_kl_div_diff))

    df = pd.DataFrame(filtered, columns=[
         "sample_idx", 
         "end_token_idx", 
         "kl_div_change", 
         "higher_order_kl_div_change"
        ])
    df.to_csv(f"filtered-{n}-gram-data-{model.replace('/', '--')}.csv", index=False)


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