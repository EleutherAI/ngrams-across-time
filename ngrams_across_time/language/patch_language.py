from torch import Tensor
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

from transformer_lens import HookedTransformer
from auto_circuit.types import PruneScores, OutputSlice
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.visualize import net_viz

from ngrams_across_time.language.hf_client import with_retries
from ngrams_across_time.language.daat import AblationDataset, collate_fn, mask_gradient_prune_scores_daat


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "EleutherAI/pythia-160m"
    revision_1, revision_2 = "step128", "step256"
    n = 2
    max_seq_len = 10

    ngrams_dataset_name = "/mnt/ssd-1/lucia/ngrams-across-time/data/val_tokenized.hf"
    dataset = load_from_disk(ngrams_dataset_name)

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


def left_pad(tensors: list[Tensor], target_len: int) -> Tensor:
    padded_tensors = []
    for tensor in tensors:
        pad_length = target_len - tensor.size(0)
        padding = torch.zeros(pad_length, dtype=tensor.dtype, device=tensor.device)
        padded_tensor = torch.cat([padding, tensor])
        padded_tensors.append(padded_tensor)

    return torch.stack(padded_tensors)
    # padded_prompts: Tensor = left_pad(prompts, 2048).cuda()
    # answers = [prompt[-1] for prompt in padded_prompts]


def get_patchable_model(model_name: str, revision: str, device: torch.device, seq_len: int, slice_output: OutputSlice = "last_seq"):
    def load():
        return HookedTransformer.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            revision=revision,
            cache_dir=".cache"
        ).cuda()
    
    model: HookedTransformer = with_retries(load) # type: ignore
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    return patchable_model(model, factorized=True, device=device, separate_qkv=True, seq_len=seq_len, slice_output=slice_output)


if __name__ == "__main__":
    main()