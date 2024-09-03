from pathlib import Path
from argparse import ArgumentParser
from typing import Literal
import torch
import pandas as pd
import numpy as np

from ngrams_across_time.utils.tensor_db import TensorDatabase

def filter_data(
        model: str,
        db_path: Path,
        start: int,
        end: int,
        dataset: str,
        ce_type: Literal['got', 'qn'],
): 
    # Arbitrarily chosen percentile cutoff
    p = 0.2

    db = TensorDatabase(str(db_path / "tensor_db.sqlite"), str(db_path / "tensors"))

    metric = f"kl_{ce_type}"

    def kl_div_delta(start: int, end: int):
        kl_start = db.query_last(model=model, step=start, metric=metric, dataset=dataset)
        kl_end = db.query_last(model=model, step=end, metric=metric, dataset=dataset)
        loss_start = db.query_last(model=model, step=start, metric='ce', dataset=dataset)
        loss_end = db.query_last(model=model, step=end, metric='ce', dataset=dataset)
        if kl_start is None or kl_end is None:
            raise ValueError(f"Could not find KL divergence data for model {model} at steps {start} and {end}, dataset {dataset}, ce_type {ce_type}")
        return torch.stack([
            kl_start['tensor'][:, 0],
            kl_start['tensor'][:, 1] - kl_end['tensor'][:, 1], 
            kl_start['tensor'][:, 1], 
            kl_end['tensor'][:, 1],
            loss_start['tensor'][:, 1], 
            loss_end['tensor'][:, 1]
            ], dim=1)

    kl_delta = kl_div_delta(start, end)

    filtered = kl_delta[kl_delta[:, 4] > kl_delta[:, 5]] # End loss should be lower than start loss

    cut_point_diff = torch.quantile(filtered[:, 1], 1-p)
    end_cut_point = torch.quantile(filtered[:, 3], p)

    filtered = filtered[filtered[:, 1] > cut_point_diff]
    filtered = filtered[filtered[:, 3] < end_cut_point]

    print(f"Cut points: diff {cut_point_diff:.2f}, end {end_cut_point:.2f},\
           ranges: diff {kl_delta[:, 1].min():.2f} - {kl_delta[:, 1].max():.2f},\
              end {kl_delta[:, 3].min():.2f} - {kl_delta[:, 3].max():.2f}\
            remaining: {len(filtered)} of {len(kl_delta)}")

    sorted_indices = torch.argsort(filtered[:, 1], descending=True)
    filtered = filtered[sorted_indices]

    image_hashes = db.query_last(model=model, step=start, metric='image_hashes', dataset=dataset)['tensor']
    
    df = pd.DataFrame(filtered, columns=[
         "sample_idx", 
         "kl_div_change",
         "start_kl_div",
         "end_kl_div",
         'start_loss',
         'end_loss'
        ])
    df['sample_idx'] = df['sample_idx'].astype(int)
    df['image_hash'] = [image_hashes[int(idx)].item() for idx in df['sample_idx']]
    df.to_csv(f"filtered-{dataset}-data-{model.replace('/', '--')}-{start}-{end}-{ce_type}.csv", index=False)


def parse_args():
    parser = ArgumentParser(description='Select data points with large differences in concept edit impact between two checkpoints')
    
    group = parser.add_argument_group("Model arguments")
    group.add_argument('--model', type=str, default="convnext-nano", help='Model name')
    group.add_argument('--start', type=int, default=128, help='Start checkpoint')
    group.add_argument('--end', type=int, default=256, help='End checkpoint')
    group.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    group.add_argument('--ce_type', type=str, default='qn', help='Concept edit type')
    
    group = parser.add_argument_group("Data arguments")
    group.add_argument('--db', type=str, default="test_image", help='Location to save data in SQLite database')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_data(args.model, Path(args.db), start=args.start, end=args.end, dataset=args.dataset, ce_type=args.ce_type)