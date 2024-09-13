from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import Dataset

import torch
from torch import Tensor

from ngrams_across_time.utils.data import ZippedDataset
from ngrams_across_time.utils.tensor_db import TensorDatabase
from ngrams_across_time.image.image_data_types import image_hash
from ngrams_across_time.language.filter_ngrams import save_ngram_corruptions

def select_above_cutpoint(matrix: Tensor, cutpoint: float):
    mask = matrix > cutpoint
    if matrix.ndim == 2:
        rows, cols = torch.where(mask)
        values = matrix[mask]        
        return list(zip(zip(rows.tolist(), cols.tolist()), values.tolist()))
    elif matrix.ndim == 1:
        indices = torch.where(matrix > cutpoint)[0]
        values = matrix[mask]
        return list(zip(indices.tolist(), values.tolist()))
    else:
        raise ValueError(f"Invalid matrix dimension: {matrix.ndim}")


def filter_data(
        model: str,
        metric_db_path: Path,
        data: ZippedDataset,
        start: int,
        end: int,
        order: int,
        modality: Literal["language", "image"],
        quantile: float = 0.1,
        off_target_quantile: float = 0.3,
        metric: str = "kl",
        data_dir: Path = Path("data/"),
    ):
    metric_db = TensorDatabase(str(metric_db_path / "tensor_db"), str(metric_db_path / "tensors"))

    example_data = data.base_dataset

    def metric_delta(start: int, end: int, order: int):
        metric_start = metric_db.query_last(model=model, step=start, metric=metric, order=order)
        metric_end = metric_db.query_last(model=model, step=end, metric=metric, order=order)
        if metric_start is None or metric_end is None:
            raise ValueError(f"Could not find {metric} data for model {model} at steps {start} and {end}")
        
        return metric_start['tensor'] - metric_end['tensor']

    lower_metric_delta = metric_delta(start, end, order=order - 1)
    target_metric_delta = metric_delta(start, end, order=order)
    higher_metric_delta = metric_delta(start, end, order=order + 1)

    # We want lower and higher ngrams to change little with respect to the target order
    off_target_upper_bound = torch.quantile(target_metric_delta, off_target_quantile)

    # We want to get target order ngrams with large KL divergence changes with respect to any order
    higher_cutpoint_lower = torch.quantile(lower_metric_delta, 1 - quantile)
    higher_cutpoint_target = torch.quantile(target_metric_delta, 1 - quantile)
    higher_cutpoint_higher = torch.quantile(higher_metric_delta, 1 - quantile)
    # target_order_lower_bound = torch.max(torch.tensor([higher_cutpoint_lower, higher_cutpoint_target, higher_cutpoint_higher, 0]))
    target_order_lower_bound = higher_cutpoint_target

    loss_order = order if modality == 'image' else 'baseline'
    start_loss = metric_db.query_last(model=model, step=start, metric='loss', order=loss_order)
    end_loss = metric_db.query_last(model=model, step=end, metric='loss', order=loss_order)

    filtered = []
    # Select the sequences with the largest drops in KL divergence
    for (idx, metric_div_reduction) in select_above_cutpoint(target_metric_delta, target_order_lower_bound):
        higher_metric_div_reduction = higher_metric_delta[idx].item()
        lower_metric_div_reduction = lower_metric_delta[idx].item()
        # Filter to items where the next highest order n-gram's KL divergence doesn't drop
        if higher_metric_div_reduction <= off_target_upper_bound and lower_metric_div_reduction <= off_target_upper_bound:
            selected_data = []
            # print(f"{row} {col}: n-gram KL div: {kl_div_reduction}, higher order n-gram KL div: {higher_kl_div_reduction}")
            if modality == "language":
                example_idx = slice(idx[1] - order, idx[1])
                example = example_data[idx[0]]['input_ids'][example_idx].tolist()
                if start_loss: selected_data.append(start_loss['tensor'][idx[0], idx[1] - 1].item())
                if end_loss: selected_data.append(end_loss['tensor'][idx[0], idx[1] - 1].item())
            else:
                example = image_hash(example_data[idx]['pixel_values'])
                if start_loss: selected_data.append(start_loss['tensor'][idx].item())
                if end_loss: selected_data.append(end_loss['tensor'][idx].item())
            selected_data.extend([idx, metric_div_reduction, higher_metric_div_reduction, example])

            filtered.append(selected_data)

    if len(filtered) == 0:
        print(f"No data selected for {model} at steps {start} and {end}")

    df = pd.DataFrame(filtered, columns=
        (
            ["start_loss"] if start_loss else []
        ) + (
            ["end_loss"] if end_loss else []
        ) + [
         "sample_idx", 
         f"{metric}_reduction", 
         f"higher_order_{metric}_reduction",
         "example"
        ]
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / f"filtered-{order}-data-{model.replace('/', '--')}-{start}-{end}.csv", index=False)

    if modality == "language":
        save_ngram_corruptions(df, model, start, end, order)