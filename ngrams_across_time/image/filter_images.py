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
    p_diff_change = 0.8
    p_end_clean_logit_diff = 0.3

    db = TensorDatabase(str(db_path / "tensor_db.sqlite"), str(db_path / "tensors"))

    metric = f"js_{ce_type}"

    def logit_diff_delta(start: int, end: int):
        logit_diff_clean_start = db.query_last(model=model, step=start, metric=f'logit_diff_clean_correct_{ce_type}', dataset=dataset)['tensor']
        logit_diff_clean_end = db.query_last(model=model, step=end, metric=f'logit_diff_clean_correct_{ce_type}', dataset=dataset)['tensor']
        logit_diff_edit_start = db.query_last(model=model, step=start, metric=f'logit_diff_{ce_type}_correct_{ce_type}', dataset=dataset)['tensor']
        logit_diff_edit_end = db.query_last(model=model, step=end, metric=f'logit_diff_{ce_type}_correct_{ce_type}', dataset=dataset)['tensor']
        logit_diff_delta_start = (logit_diff_clean_start - 
                            logit_diff_edit_start)
        logit_diff_delta_end = (logit_diff_clean_end - 
                            logit_diff_edit_end)
        if logit_diff_clean_start is None or logit_diff_clean_end is None:
            raise ValueError(f"Could not find divergence data for model {model} at steps {start} and {end}, dataset {dataset}, ce_type {ce_type}")
        return torch.stack([
            logit_diff_clean_start[:, 0],
            logit_diff_delta_start[:, 1] - logit_diff_delta_end[:, 1], 
            logit_diff_delta_start[:, 1],     
            logit_diff_delta_end[:, 1],
            logit_diff_clean_start[:, 1],
            logit_diff_clean_end[:, 1],
            logit_diff_edit_start[:, 1],
            logit_diff_edit_end[:, 1],
            ], dim=1)

    ld_delta = logit_diff_delta(start, end)

    filtered = ld_delta[ld_delta[:, 4] < ld_delta[:, 5]] # End clean logit diff should be higher than start clean logit diff

    cut_point_diff = torch.quantile(filtered[:, 1], p_diff_change)
    end_cut_point = torch.quantile(filtered[:, 3], p_end_clean_logit_diff)

    filtered = filtered[filtered[:, 1] > cut_point_diff]
    filtered = filtered[filtered[:, 3] < end_cut_point]

    print(f"Cut points: diff {cut_point_diff:.2f}, end {end_cut_point:.2f},\
           ranges: diff {ld_delta[:, 1].min():.2f} - {ld_delta[:, 1].max():.2f},\
              end {ld_delta[:, 3].min():.2f} - {ld_delta[:, 3].max():.2f}\
            remaining: {len(filtered)} of {len(ld_delta)}")

    sorted_indices = torch.argsort(filtered[:, 1], descending=True)
    filtered = filtered[sorted_indices]

    image_hashes = db.query_last(model=model, step=start, metric='image_hashes', dataset=dataset)['tensor']
    
    df = pd.DataFrame(filtered, columns=[
         "sample_idx", 
         "logit_diff_change",
         "start_logit_diff_delta",
         "end_logit_diff_delta",
         'start_clean_logit_diff',
         'end_clean_logit_diff',
         'start_edit_logit_diff',
         'end_edit_logit_diff',
        ])
    df['sample_idx'] = df['sample_idx'].astype(int)
    df['image_hash'] = [image_hashes[int(idx)].item() for idx in df['sample_idx']]
    df.to_csv(f"data/filtered-{dataset}-data-{model.replace('/', '--')}-{start}-{end}-{ce_type}.csv", index=False)


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