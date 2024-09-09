from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import argparse

import torch

from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.data import PromptDataLoader
from auto_circuit.types import AblationType
from auto_circuit.visualize import net_viz


from ngrams_across_time.experiments.load_models import load_models


def main(model_name: str, start: int, end: int, modality: str, order: int, dataset_name: str = ''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if modality == 'language':
        models, dataset_dict = load_models(
            modality,
            model_name,
            order,
            max_ds_len=1024,
            start=start,
            end=end,
            patchable=True,
        )
    else:
        models, dataset_dict = load_models(
            modality,
            model_name,
            order,
            dataset_name,
            start=start,
            end=end,
            patchable=True,
        )

    circuit_data = {
        "label": [],  # This will be a list of lists (token indices)
        "daat_edge_prune_scores": [],  # This will be a list of tensors
        "quantile_thresholds": [], # List of floats
    }
    quantiles = [0.1, 0.5, 0.9, 0.99, 0.999]
    base_model = models[end]
    ablation_model = models[start]

    for label, prompt_ds in dataset_dict.items():
        dataloader = PromptDataLoader(prompt_ds, seq_len=order - 1, diverge_idx=0)
        circuit_data["daat_edge_prune_scores"] = mask_gradient_prune_scores(
            model=base_model,
            dataloader=dataloader,
            official_edges=None,
            grad_function="logit",
            answer_function="avg_val",
            mask_val=0.0,
            ablation_type=AblationType.RESAMPLE,
            clean_corrupt="clean",
            alternate_model=ablation_model
        )
        circuit_data["label"].append(label)
        circuit_data["quantile_thresholds"].append([np.quantile(circuit_data["daat_edge_prune_scores"], q) for q in quantiles])

        # Viz
        for threshold in circuit_data["quantile_thresholds"]:
            sankey, included_layer_count = net_viz(base_model, base_model.edges, circuit_data["daat_edge_prune_scores"], score_threshold=threshold, vert_interval=(0, 1))
            
            if included_layer_count == 0:
                break
            print(f'{included_layer_count} layers included')
        
            fig = go.Figure(data=sankey)
            fig.write_image(f"{label}_viz_{threshold}.png", scale=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Circuit Discovery')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--start', type=int, required=True, help='Start index')
    parser.add_argument('--end', type=int, required=True, help='End index')
    parser.add_argument('--modality', type=str, required=True, choices=['language', 'vision'], help='Modality')
    parser.add_argument('--order', type=int, required=True, help='Order')
    parser.add_argument('--dataset_name', type=str, default='', help='Dataset name (optional)')

    args = parser.parse_args()

    main(args.model_name, args.start, args.end, args.modality, args.order, args.dataset_name)
