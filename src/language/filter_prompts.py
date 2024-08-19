from pathlib import Path

import torch
import numpy as np

from src.utils.tensor_db import TensorDatabase


def calculate_diffs(
        model_name: str = "EleutherAI/pythia-160m",
        out: Path = Path("/mnt/ssd-1/tensor_db"),
):
    db = TensorDatabase(str(out / "tensor_db.sqlite"), str(out / "tensors"))

    first = db.query_last(
        model=model_name, 
        step=128, 
        metric='kl', 
        ngram=2
    )

    second = db.query_last(
        model=model_name, 
        step=256, 
        metric='kl', 
        ngram=2
    )['tensor']

    # Calculate the difference in KL divergence between the two checkpoints
    kl_diff = first - second
    
    flat_indices = torch.argsort(kl_diff.flatten())
    top_flat_indices = flat_indices[:30]
    rows, cols = np.unravel_index(top_flat_indices.cpu().numpy(), kl_diff.shape)
    top_coords_and_values = [(row, col, kl_diff[row, col].item()) for row, col in zip(rows, cols)]
    top_coords_and_values.sort(key=lambda x: x[2])
    for i, (row, col, value) in enumerate(top_coords_and_values, 1):
        print(f"#{i}: (row={row}, col={col}) = {value:.4f}")

    # Save rows and cols to csv with columns "sample_idx", "token_idx" and values as the KL divs
    import pandas as pd
    df = pd.DataFrame(top_coords_and_values, columns=["sample_idx", "token_idx", "kl_div"])
    df.to_csv("top_kl_divs.csv", index=False)



if __name__ == "__main__":
    calculate_diffs()