import torch
from torch.utils.data import Dataset
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd

from ngrams_across_time.image.load_model_data import load_models_and_dataset
from ngrams_across_time.image.data_types import QuantileNormalizedDataset, IndependentCoordinateSampler, GaussianMixture, ConceptEditedDataset
from auto_circuit.utils.graph_utils import patch_mode, patchable_model
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.data import PromptDataLoader, PromptDataset
from auto_circuit.types import AblationType
from auto_circuit.prune_algos.subnetwork_probing import subnetwork_probing_prune_scores
from auto_circuit.visualize import net_viz

import plotly.graph_objects as go


def get_logit_diff(model, input_tensor, correct_class, target_class):
    with torch.no_grad():
        output = model(input_tensor)
    logits = output.logits
    return (logits[range(len(logits)), correct_class] - logits[range(len(logits)), target_class]).mean().item()

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

    model_loader, dataset = load_models_and_dataset(model_name, dataset_name, ce_type)
    
    # Load young and old models
    models = list(model_loader)
    if patch_type == 'restore_ho':
        # Activations from more trained model restore less trained model
        source_model = patchable_model(models[-1][1], factorized=False, device=device)
        rec_model = patchable_model(models[0][1], factorized=False, device=device)
    elif patch_type == 'ablate_ho':
        # Activations from less trained model ablate more trained model
        source_model = patchable_model(models[0][1], factorized=False, device=device)
        rec_model = patchable_model(models[-1][1], factorized=False, device=device)

    data_index_path = Path(f"data/filtered-{dataset_name}-data-{model_name.replace('/', '--')}-{start_step}-{end_step}.csv")
    data_indices = pd.read_csv(data_index_path)
    
    # Filter data
    dataset = dataset.select(data_indices['sample_idx'].tolist())

    if ce_type == 'qn':
        normal = dataset.normal_dataset
        edited = dataset.qn_dataset
    elif ce_type == 'got':
        normal = dataset.normal_dataset
        edited = dataset.got_dataset
    elif ce_type == 'synthetic':
        normal = dataset.gauss_dataset
        edited = dataset.ics_dataset

    prompt_dataset = create_prompt_dataset_from_edited(edited, normal, img_col="pixel_values", label_col="label")
    # Create PromptDataLoader
    ploader = PromptDataLoader(prompt_dataset, None, 0, batch_size=1, shuffle=False)
    edit_ablations = batch_src_ablations(source_model, ploader, AblationType.RESAMPLE, 'corrupt')
    clean_ablations = batch_src_ablations(source_model, ploader, AblationType.RESAMPLE, 'clean')

    edit_only_dataset = create_prompt_dataset_from_edited(edited, edited, img_col="pixel_values", label_col="label")
    edit_only_ploader = PromptDataLoader(edit_only_dataset, None, 0, batch_size=1, shuffle=False)
    # Discover circuit
    edge_scores = subnetwork_probing_prune_scores(rec_model, edit_only_ploader, tree_optimisation=False, alternate_model=source_model)

    keep_edges = []
    for edge in rec_model.edges:
        if edge.dest.module_name in edge_scores:
            if edge_scores[edge.dest.module_name][edge.patch_idx] > 0.9:
                keep_edges.append(edge)
    results = []

    for batch in ploader:
        clean_input = batch.clean.to(device)
        corrupt_input = batch.corrupt.to(device)
        correct_class = batch.answers
        target_class = batch.wrong_answers

        # Young model results
        source_clean_diff = get_logit_diff(source_model, clean_input, correct_class, target_class)
        source_corrupt_diff = get_logit_diff(source_model, corrupt_input, correct_class, target_class)

        # Old model results
        rec_clean_diff = get_logit_diff(rec_model, clean_input, correct_class, target_class)
        rec_corrupt_diff = get_logit_diff(rec_model, corrupt_input, correct_class, target_class)

        # Patched young model results
        with patch_mode(rec_model, next(iter(edit_ablations.values())), keep_edges):
            patched_rec_corrupt_diff = get_logit_diff(rec_model, corrupt_input, correct_class, target_class)

        with patch_mode(rec_model, next(iter(clean_ablations.values())), keep_edges):
            patched_rec_clean_diff = get_logit_diff(rec_model, clean_input, correct_class, target_class)

        results.append({
            'source_clean_diff': source_clean_diff,
            'source_corrupt_diff': source_corrupt_diff,
            'rec_clean_diff': rec_clean_diff,
            'rec_corrupt_diff': rec_corrupt_diff,
            'patched_rec_clean_diff': patched_rec_clean_diff,
            'patched_rec_corrupt_diff': patched_rec_corrupt_diff,
        })

    for threshold in [0.1, 0.2, 0.3, 0.5, 0.9, 1]:
        sankey, included_layer_count = net_viz(rec_model, rec_model.edges, edge_scores, score_threshold=threshold, vert_interval=(0, 1))
        
        if included_layer_count == 0:
            break
        print(f'{included_layer_count} layers included')
    
        fig = go.Figure(data=sankey)
        fig.write_image(Path(output_dir) / f"viz_{threshold}_{patch_type}.png", scale=4)


    return pd.DataFrame(results)

def main():
    parser = ArgumentParser(description='Run DAAT experiment')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--ce_type', type=str, choices=['got', 'qn', 'synthetic'], required=True, help='Concept edit type')
    parser.add_argument('--start_step', type=int, required=True, help='Start step for filtering')
    parser.add_argument('--end_step', type=int, required=True, help='End step for filtering')
    parser.add_argument('--output', type=str, required=True, help='Directory to save results and visualizations')
    parser.add_argument('--num_edges', type=int, default=1000, help='Number of edges to patch')
    parser.add_argument('--patch_type', type=str, choices=['ablate_ho', 'restore_ho'], default='ablate_ho', help='Type of patch to create')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiment(
        args.model,
        args.dataset,
        args.ce_type,
        args.start_step,
        args.end_step,
        output_dir,
        args.patch_type,
        args.num_edges,
    )

    results.to_csv(output_dir / f"results-{args.model}-{args.dataset}-{args.ce_type}-{args.patch_type}-{args.start_step}-{args.end_step}-n{args.num_edges}.csv", index=False)
    print(f"Results and visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()