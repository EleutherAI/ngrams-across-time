
import torch

from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.utils.graph_utils import patch_mode, patchable_model

from ngrams_across_time.language.patch_language import get_tl_model
from ngrams_across_time.language.load_language_model_data import load_token_data
from ngrams_across_time.experiments.load_models import load_models


def main(model_name: str, start: int, end: int, modality: str, order: int, dataset_name: str = ''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if modality == 'language':
        models, dataset = load_models(
            modality,
            model_name,
            order,
            dataset_name,
            True,
            max_ds_len=1024,
            start=start,
            end=end,
        )
    else:
        models, dataset = load_models(
            modality,
            model_name,
            order,
            dataset_name,
            start=start,
            end=end,
        )

    
    prompts_dataset_name = f"/mnt/ssd-1/lucia/ngrams-across-time/filtered-{n}-gram-data-{model_name.replace('/', '--')}.csv"
    if not Path(prompts_dataset_name).exists():
        # TODO generate prompts dataset here
        f"Prompt dataset not found - use select_language_prompts.py to generate."

    prompts_dataset = pd.read_csv(prompts_dataset_name)
    prompts: list[Tensor] = [
        dataset[int(row['sample_idx'])]['input_ids'][:int(row['end_token_idx']) + 1]
        for _, row in prompts_dataset.iterrows()    
    ] # type: ignore
    prompts = prompts[:1]
    print(len(prompts), " prompts loaded")

    ablation_model = get_patchable_model(model_name, revision_1, device, max_seq_len)
    ablation_ds = AblationDataset(prompts, ablation_model, max_seq_len)
    ablation_dl = DataLoader(ablation_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # TODO figure out how to narrow down ablations to individual tokens
    model = get_patchable_model(model_name, revision_2, device, max_seq_len)

    print("Calculating EAP prune scores for revision 1")
    edge_prune_scores: PruneScores = mask_gradient_prune_scores_daat(
        model=model,
        dataloader=ablation_dl,
        grad_function="logit",
        answer_function="avg_val",
        mask_val=0.0
    )

    # Viz
    for threshold in [0.1, 0.2, 0.3, 0.5, 1]: # , 3, 5, 10, 40]:
        sankey, included_layer_count = net_viz(model, model.edges, edge_prune_scores, score_threshold=threshold, vert_interval=(0, 1))
        
        if included_layer_count == 0:
            break
        print(f'{included_layer_count} layers included')
    
        fig = go.Figure(data=sankey)
        fig.write_image(f"viz_{threshold}.png", scale=4)
    
    # print(edge_prune_scores.keys(), [v[-1].shape for v in edge_prune_scores.values()])
    # scores = {key: value[-1].sum().item() for key, value in edge_prune_scores.items()}
    # sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    # breakpoint()