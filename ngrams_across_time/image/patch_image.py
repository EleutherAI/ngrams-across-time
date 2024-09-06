import torch
from torch.utils.data import Dataset
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import numpy as np

from einops import rearrange

from ngrams_across_time.image.load_image_model_data import load_models_and_dataset
from ngrams_across_time.image.image_data_types import QuantileNormalizedDataset, IndependentCoordinateSampler, GaussianMixture, ConceptEditedDataset
from ngrams_across_time.image.collect_image_metrics import image_hash
from auto_circuit.utils.graph_utils import patch_mode, patchable_model
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.data import PromptDataLoader, PromptDataset
from auto_circuit.types import AblationType
from auto_circuit.prune_algos.subnetwork_probing import subnetwork_probing_prune_scores

import plotly.graph_objects as go

from tqdm import tqdm

def get_logit_diff(model, input_tensor, correct_class, target_class):
    with torch.no_grad():
        output = model(input_tensor)
    logits = output.logits
    return ((logits[range(len(logits)), correct_class] - logits[range(len(logits)), target_class]).squeeze(1).cpu().numpy(), 
            torch.nn.functional.cross_entropy(logits, correct_class.squeeze(1).to(logits.device), reduction='none').cpu().numpy(), 
            torch.nn.functional.cross_entropy(logits, target_class.squeeze(1).to(logits.device), reduction='none').cpu().numpy())

def create_prompt_dataset_from_edited(
    edited_dataset: QuantileNormalizedDataset | IndependentCoordinateSampler | ConceptEditedDataset,
    original_dataset: Dataset | ConceptEditedDataset | GaussianMixture,
    img_col: str = "pixel_values",
    label_col: str = "label"
) -> PromptDataset:
    clean_prompts = []
    corrupt_prompts = []
    answers = []
    wrong_answers = []

    for i in range(len(edited_dataset)):
        original_sample = original_dataset[i]
        edited_sample = edited_dataset[i]

        clean_prompts.append(original_sample[img_col])
        corrupt_prompts.append(edited_sample[img_col])
        
        # Assuming the label is a single integer
        answers.append(torch.tensor([original_sample[label_col]]))
        wrong_answers.append(torch.tensor([edited_sample[label_col]]))

    return PromptDataset(
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
        answers=answers,
        wrong_answers=wrong_answers
    )

def run_experiment(model_name, dataset_name, ce_type, start_step, end_step, output_dir, patch_type, num_edges=1000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return_type = 'edited' if ce_type in ['qn', 'got'] else 'synthetic'

    model_loader, dataset = load_models_and_dataset(model_name, dataset_name, return_type)
    model_loader = list(model_loader)
    if patch_type == 'restore_ho':
        # Activations from more trained model restore less trained model
        source_model = patchable_model([m for ckpt, m in model_loader if ckpt == end_step][0], factorized=True, device=device)
        rec_model = patchable_model([m for ckpt, m in model_loader if ckpt == start_step][0], factorized=True, device=device)
    elif patch_type == 'ablate_ho':
        # Activations from less trained model ablate more trained model
        source_model = patchable_model([m for ckpt, m in model_loader if ckpt == start_step][0], factorized=True, device=device)
        rec_model = patchable_model([m for ckpt, m in model_loader if ckpt == end_step][0], factorized=True, device=device)

    if ce_type in ['qn', 'got']:
        data_index_path = Path(f"data/filtered-{dataset_name}-data-{model_name.replace('/', '--')}-{start_step}-{end_step}-{ce_type}.csv")
        data_indices = pd.read_csv(data_index_path)
        
        dataset = dataset.select(data_indices['sample_idx'].tolist())
        normal = dataset.normal_dataset
        edited = dataset.qn_dataset if ce_type == 'qn' else dataset.got_dataset

    elif ce_type == 'synthetic':
        normal = dataset.gauss_dataset
        edited = dataset.ics_dataset

    prompt_dataset = create_prompt_dataset_from_edited(edited, normal, img_col="pixel_values", label_col="label")
    ploader = PromptDataLoader(prompt_dataset, None, 0, batch_size=1, shuffle=False)
    edit_ablations = batch_src_ablations(source_model, ploader, AblationType.RESAMPLE, 'corrupt')
    clean_ablations = batch_src_ablations(source_model, ploader, AblationType.RESAMPLE, 'clean')

    edit_only_dataset = create_prompt_dataset_from_edited(edited, edited, img_col="pixel_values", label_col="label")
    edit_only_ploader = PromptDataLoader(edit_only_dataset, None, 0, batch_size=1, shuffle=False)
    
    for i, sample in enumerate(dataset):
        computed_hash = image_hash(sample[0]['pixel_values'])
        if computed_hash != data_indices['image_hash'][i]:
            print(f"Mismatch at index {i}: expected {data_indices['image_hash'][i]}, got {computed_hash}")


    edge_scores = subnetwork_probing_prune_scores(rec_model, edit_only_ploader, tree_optimisation=False, alternate_model=source_model, epochs=10)

    quantile = 0.9999

    all_scores = torch.cat([scores.flatten() for scores in edge_scores.values()])
    threshold = np.quantile(all_scores.cpu().numpy(), quantile)

    keep_edges = []
    top_edges_with_scores = []
    for module, scores in tqdm(edge_scores.items(), 'compiling edge scores'):
        scores = rearrange(scores, 'dest_idx src_idx -> (dest_idx src_idx)')
        high_score_indices = (scores > threshold).nonzero().squeeze(1)
        if high_score_indices.numel() == 0:
            continue
        keep_edges.extend([rec_model.edge_dict[None][i] for i in high_score_indices])
        top_edges_with_scores.extend([(rec_model.edge_dict[None][i].name, scores[i].item()) for i in high_score_indices])
    results = []

    for batch in tqdm(ploader, f'testing patch results on {quantile} quantile edges'):
        clean_input = batch.clean.to(device)
        corrupt_input = batch.corrupt.to(device)
        correct_class = batch.answers
        target_class = batch.wrong_answers

        # Young model results
        source_clean_diff, source_clean_loss, source_clean_off_loss = get_logit_diff(source_model, clean_input, correct_class, target_class)
        source_corrupt_diff, source_corrupt_loss, source_corrupt_off_loss = get_logit_diff(source_model, corrupt_input, correct_class, target_class)
        # Old model results
        rec_clean_diff, rec_clean_loss, rec_clean_off_loss = get_logit_diff(rec_model, clean_input, correct_class, target_class)
        rec_corrupt_diff, rec_corrupt_loss, rec_corrupt_off_loss = get_logit_diff(rec_model, corrupt_input, correct_class, target_class)
        # Patched young model results
        with patch_mode(rec_model, edit_ablations[batch.key], keep_edges):
            patched_rec_corrupt_diff, patched_rec_corrupt_loss, patched_rec_corrupt_off_loss = get_logit_diff(rec_model, corrupt_input, correct_class, target_class)
        with patch_mode(rec_model, clean_ablations[batch.key], keep_edges):
            patched_rec_clean_diff, patched_rec_clean_loss, patched_rec_clean_off_loss = get_logit_diff(rec_model, clean_input, correct_class, target_class)
        batch_results = {
            'source_clean_diff': source_clean_diff,
            'source_corrupt_diff': source_corrupt_diff,
            'source_clean_loss': source_clean_loss,
            'source_corrupt_loss': source_corrupt_loss,
            'source_clean_off_loss': source_clean_off_loss,
            'source_corrupt_off_loss': source_corrupt_off_loss,
            'rec_clean_diff': rec_clean_diff,
            'rec_corrupt_diff': rec_corrupt_diff,
            'rec_clean_loss': rec_clean_loss,
            'rec_corrupt_loss': rec_corrupt_loss,
            'rec_clean_off_loss': rec_clean_off_loss,
            'rec_corrupt_off_loss': rec_corrupt_off_loss,
            'patched_rec_clean_diff': patched_rec_clean_diff,
            'patched_rec_corrupt_diff': patched_rec_corrupt_diff,
            'patched_rec_clean_loss': patched_rec_clean_loss,
            'patched_rec_corrupt_loss': patched_rec_corrupt_loss,
            'patched_rec_clean_off_loss': patched_rec_clean_off_loss,
            'patched_rec_corrupt_off_loss': patched_rec_corrupt_off_loss,
            'patched_rec_clean_diff': patched_rec_clean_diff,
            'patched_rec_corrupt_diff': patched_rec_corrupt_diff,
        }

        results.append(pd.DataFrame(batch_results))

    results = pd.concat(results, ignore_index=True)
    # graph_quantile = 0.99999
    # threshold = np.quantile(all_scores.cpu().numpy(), graph_quantile)
    # sankey, included_layer_count = net_viz(rec_model, rec_model.edges, edge_scores, score_threshold=threshold, vert_interval=(0, 1))
    
    # print(f'{included_layer_count} layers included')

    # fig = go.Figure(data=sankey)
    # fig.write_image(Path(output_dir) / f"viz_{threshold}_{patch_type}.png", scale=4)

    top_edges_with_scores.sort(key=lambda x: x[1], reverse=True)

    return results, top_edges_with_scores

def main():
    parser = ArgumentParser(description='Run DAAT experiment')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--ce_type', type=str, choices=['got', 'qn', 'synthetic'], required=True, help='Concept edit type')
    parser.add_argument('--start', type=int, required=True, help='Start step for filtering')
    parser.add_argument('--end', type=int, required=True, help='End step for filtering')
    parser.add_argument('--output', type=str, required=True, help='Directory to save results and visualizations')
    parser.add_argument('--num_edges', type=int, default=1000, help='Number of edges to patch')
    parser.add_argument('--patch_type', type=str, choices=['ablate_ho', 'restore_ho'], default='ablate_ho', help='Type of patch to create')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, top_edges = run_experiment(
        args.model,
        args.dataset,
        args.ce_type,
        args.start,
        args.end,
        output_dir,
        args.patch_type,
        args.num_edges,
    )

    results.to_csv(output_dir / f"results-{args.model}-{args.dataset}-{args.ce_type}-{args.patch_type}-{args.start}-{args.end}-n{args.num_edges}.csv", index=False)
    
    # Save top edges with scores
    with open(output_dir / f"top_edges-{args.model}-{args.dataset}-{args.ce_type}-{args.patch_type}-{args.start}-{args.end}-n{args.num_edges}.txt", 'w') as f:
        for edge, score in top_edges:
            f.write(f"{edge}: {score}\n")

    print(f"Results, visualizations, and top edges saved to {output_dir}")

if __name__ == "__main__":
    main()
