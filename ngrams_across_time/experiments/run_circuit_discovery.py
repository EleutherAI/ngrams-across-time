import numpy as np
import plotly.graph_objects as go
import argparse
from tqdm import tqdm
import pandas as pd
import torch
from einops import rearrange
import itertools

from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.data import PromptDataLoader
from auto_circuit.types import AblationType
from auto_circuit.visualize import net_viz
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.utils.graph_utils import patch_mode


from ngrams_across_time.experiments.load_models import load_models
from ngrams_across_time.utils.utils import get_logit_diff, get_loss



def score_circuits(quantiles, edge_scores, source_model, dest_model, dataloader, label, mode='multitarget'):
    device = next(dest_model.parameters()).device
    results = []

    corrupt_ablations = batch_src_ablations(source_model, dataloader, AblationType.RESAMPLE, 'corrupt')
    clean_ablations = batch_src_ablations(source_model, dataloader, AblationType.RESAMPLE, 'clean')
    for quantile, threshold in quantiles:
        keep_edges = []
        top_edges_with_scores = []
        for module, scores in tqdm(edge_scores.items(), 'compiling edge scores'):
            sequential = scores.ndim == 3
            if not sequential:
                scores = scores.unsqueeze(0)
            for seq_pos, score in enumerate(scores):
                score = rearrange(score, 'dest_idx src_idx -> (dest_idx src_idx)')
                high_score_indices = (score > threshold).nonzero().squeeze(1)
                if high_score_indices.numel() == 0:
                    continue

                seq_pos = seq_pos if sequential else None
                keep_edges.extend([dest_model.edge_dict[seq_pos][i] for i in high_score_indices])
                top_edges_with_scores.extend([(dest_model.edge_dict[seq_pos][i].name, score[i].item()) for i in high_score_indices])


        for batch in tqdm(dataloader, f'testing patch results on {quantile} quantile edges'):
            clean_input = batch.clean.to(device)
            corrupt_input = batch.corrupt.to(device)
            correct_class = batch.answers

            if mode == 'multitarget':
                target_class = batch.wrong_answers
                get_metrics = get_logit_diff
            else:
                target_class = None
                get_metrics = lambda model, input, correct_class, _: get_loss(model, input, correct_class)

            # Young model results
            source_clean_metrics = get_metrics(source_model, clean_input, correct_class, target_class)
            source_corrupt_metrics = get_metrics(source_model, corrupt_input, correct_class, target_class)
            
            # Old model results
            dest_clean_metrics = get_metrics(dest_model, clean_input, correct_class, target_class)
            dest_corrupt_metrics = get_metrics(dest_model, corrupt_input, correct_class, target_class)
            
            # Patched young model results
            with patch_mode(dest_model, corrupt_ablations[batch.key], keep_edges):
                patched_dest_corrupt_metrics = get_metrics(dest_model, corrupt_input, correct_class, target_class)
            with patch_mode(dest_model, clean_ablations[batch.key], keep_edges):
                patched_dest_clean_metrics = get_metrics(dest_model, clean_input, correct_class, target_class)

            batch_results = {
                'quantile': quantile,
                'label': ','.join(map(str, label.numpy())),
                'n_edges': len(keep_edges)
            }

            if mode == 'multitarget':
                batch_results.update({
                    'source_clean_diff': source_clean_metrics[0],
                    'source_corrupt_diff': source_corrupt_metrics[0],
                    'source_clean_loss': source_clean_metrics[1],
                    'source_corrupt_loss': source_corrupt_metrics[1],
                    'source_clean_off_loss': source_clean_metrics[2],
                    'source_corrupt_off_loss': source_corrupt_metrics[2],
                    'dest_clean_diff': dest_clean_metrics[0],
                    'dest_corrupt_diff': dest_corrupt_metrics[0],
                    'dest_clean_loss': dest_clean_metrics[1],
                    'dest_corrupt_loss': dest_corrupt_metrics[1],
                    'dest_clean_off_loss': dest_clean_metrics[2],
                    'dest_corrupt_off_loss': dest_corrupt_metrics[2],
                    'patched_dest_clean_diff': patched_dest_clean_metrics[0],
                    'patched_dest_corrupt_diff': patched_dest_corrupt_metrics[0],
                    'patched_dest_clean_loss': patched_dest_clean_metrics[1],
                    'patched_dest_corrupt_loss': patched_dest_corrupt_metrics[1],
                    'patched_dest_clean_off_loss': patched_dest_clean_metrics[2],
                    'patched_dest_corrupt_off_loss': patched_dest_corrupt_metrics[2],
                })
            else:
                batch_results.update({
                    'source_clean_loss': source_clean_metrics[0],
                    'source_corrupt_loss': source_corrupt_metrics[0],
                    'dest_clean_loss': dest_clean_metrics[0],
                    'dest_corrupt_loss': dest_corrupt_metrics[0],
                    'patched_dest_clean_loss': patched_dest_clean_metrics[0],
                    'patched_dest_corrupt_loss': patched_dest_corrupt_metrics[0],
                })

            results.append(pd.DataFrame(batch_results))

    return pd.concat(results)

def main(model_name: str, start: int, end: int, modality: str, order: int, dataset_name: str = '', viz: bool = False):

    if modality == 'language':
        (models, vocab_size), dataset_dict = load_models(
            modality,
            model_name,
            order,
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

    circuit_scores = []
    count = 0

    for label, prompt_ds in tqdm(dataset_dict.items(), desc='Circuit Discovery'):
        count += 1
        if count > 20:
            break
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
        all_scores = torch.cat([scores.flatten() for scores in circuit_data["daat_edge_prune_scores"].values()]).cpu().numpy()

        circuit_data["quantile_thresholds"].append([np.quantile(all_scores, q) for q in quantiles])

        # Viz
        if viz:
            for threshold in circuit_data["quantile_thresholds"]:
                sankey, included_layer_count = net_viz(
                    base_model, 
                    base_model.edges, 
                    circuit_data["daat_edge_prune_scores"], 
                    score_threshold=threshold[-1], 
                    vert_interval=(0, 1)
                )
                
                if included_layer_count == 0:
                    break
                print(f'{included_layer_count} layers included')
            
                fig = go.Figure(data=sankey)
                fig.write_image(f"{label}_viz_{threshold}.png", scale=4)
        circuit_scores.append(score_circuits(
            zip(quantiles, circuit_data["quantile_thresholds"][-1]), 
            circuit_data["daat_edge_prune_scores"], 
            ablation_model, 
            base_model, 
            dataloader, 
            label,
            mode='single'
        ))

    circuit_scores = pd.concat(circuit_scores)
    circuit_scores.to_csv(f"{model_name.replace('/', '--')}_circuit_scores.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Circuit Discovery')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--start', type=int, required=True, help='Start index')
    parser.add_argument('--end', type=int, required=True, help='End index')
    parser.add_argument('--modality', type=str, required=True, choices=['language', 'image'], help='Modality')
    parser.add_argument('--order', type=int, required=True, help='Order')
    parser.add_argument('--dataset_name', type=str, default='', help='Dataset name (optional)')
    parser.add_argument('--viz', action='store_true', help='Visualize the circuit')
    args = parser.parse_args()

    main(args.model_name, args.start, args.end, args.modality, args.order, args.dataset_name, args.viz)
