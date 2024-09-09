import os
from pathlib import Path
from argparse import ArgumentParser

from torch import Tensor
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from transformer_lens import HookedTransformer
from auto_circuit.types import PruneScores, OutputSlice
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.data import PromptDataset, PromptDataLoader
from auto_circuit.visualize import net_viz, draw_seq_graph
from auto_circuit.prune import patch_mode, run_circuits

from ngrams_across_time.utils.utils import assert_type
from ngrams_across_time.language.hf_client import with_retries
from ngrams_across_time.language.daat import AblationDataset, collate_fn, mask_gradient_prune_scores_daat


def clean_bpe_decoding(token: str):
    """Clean up BPE decoding.
    From a random EleutherAI discord comment:
    # ĉ 0109 Horizontal Tab
    # ċ 010B Vertical Tab
    # Ē 0112 NL line feed, new line
    # Ĕ 0114 NP for feed, new page
    # ĕ 0115 carriage return
    # Ġ 0120 Space
    """
    
    return token.replace(
        'Ċ', '<hori_tab>').replace(
        'ċ', '<vert_tab>').replace(
        'Ē', '\\n').replace(
        'Ĕ', '\\np').replace(
        'ĕ', '\\r').replace(
        'Ġ', ' ')


def visualize(model, prune_scores, out: Path, ngram: list[str]):
    for threshold in [0.1, 0.2, 0.3, 0.5, 1, 3, 5, 10]:
        success, fig = draw_seq_graph(model, prune_scores, score_threshold=threshold, seq_labels=ngram, display_ipython=False)
        if success:
            fig.write_image(out / f"viz_{threshold}_{''.join(ngram)}.png", scale=4)
        else: 
            return

def get_prompt_loss_increase(model, circuit_data, row, num_edges, device, n):
    alternative_ngram_prefixes = torch.tensor(row['corrupt'], device=device)[:, :-1] # type: ignore

    learned_ngram = torch.tensor(row['clean'], device=device) # type: ignore
    learned_ngram_prefix_repeated = [learned_ngram[:-1] for _ in range(len(alternative_ngram_prefixes))]
    learned_ngram_suffix_repeated = [learned_ngram[-1:] for _ in range(len(alternative_ngram_prefixes))]
    
    patching_ds = PromptDataset(
        learned_ngram_prefix_repeated,
        alternative_ngram_prefixes,
        learned_ngram_suffix_repeated,
        learned_ngram_suffix_repeated
    )
    dataloader = PromptDataLoader(patching_ds, seq_len=n - 1, diverge_idx=0)

    print("Calculating EAP prune scores for patching corrupt n-grams into learned n-gram")
    edge_prune_scores: PruneScores = mask_gradient_prune_scores(
        model=model,
        dataloader=dataloader,
        official_edges=set(),
        grad_function="logit",
        answer_function="avg_val",
        mask_val=0.0
    )

    circuit_data["prompt_edge_prune_scores"].append(edge_prune_scores)

    # Get loss increase from ablating num_edges circuit edges
    batch_logits = run_circuits(model, dataloader, [num_edges], prune_scores=edge_prune_scores)[num_edges]
    batch_logits = assert_type(dict, batch_logits)

    patched_loss = F.cross_entropy(list(batch_logits.values())[0], learned_ngram[-1].unsqueeze(0).to(device))
    unpatched_loss = F.cross_entropy(
        model(learned_ngram.unsqueeze(0).to(device))[:, -1, :],
        learned_ngram[-1].unsqueeze(0).to(device)
    )
    loss_increase = patched_loss - unpatched_loss
    circuit_data['prompt_loss_increase'].append(loss_increase.item())


# TODO figure out how to narrow down ablations to individual tokens
def daat_patch(model_name: str, start: int, end: int, n: int, dataset: Dataset, out: Path, device: torch.device):
    output_path = out / 'daat'
    circuit_data_path = out / 'daat' / f'{start}-{end}-{model_name.replace("/", "--")}'
    os.makedirs(circuit_data_path, exist_ok=True)

    model = get_patchable_model(model_name, get_pythia_revision(end), device, n - 1)
    ablation_model = get_patchable_model(model_name, get_pythia_revision(start), device, n - 1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_prompt_dl(ngram):
        learned_ngram_prefix_repeated = [ngram[:-1]]
        learned_ngram_suffix_repeated = [ngram[-1:]]
        
        patching_ds = PromptDataset(
            learned_ngram_prefix_repeated,
            learned_ngram_prefix_repeated,
            learned_ngram_suffix_repeated,
            learned_ngram_suffix_repeated
        )
        return PromptDataLoader(patching_ds, seq_len=n - 1, diverge_idx=0)

    circuit_data = {
        "ngram": [],  # This will be a list of lists (token indices)
        'prompt_edge_prune_scores': [],
        "daat_edge_prune_scores": [],  # This will be a list of tensors
        'daat_loss_increase': [],
        'prompt_loss_increase': []
    }

    for row in dataset:
        ngram = torch.tensor(row['clean']) # type: ignore
        ablation_ds = AblationDataset([ngram], ablation_model)
        ablation_dl = DataLoader(ablation_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

        print("Calculating EAP prune scores for patching early model into late model")
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
        ).to(device)
    
    model: HookedTransformer = with_retries(load) # type: ignore
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    return patchable_model(model, factorized=True, device=device, separate_qkv=True, seq_len=seq_len, slice_output=slice_output)


def get_pythia_revision(step: int):
    return f"step{step}"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--out', type=str, default='output')
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-160m")
    parser.add_argument('--n', type=int, default=2)
        
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    os.makedirs(out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get precalculated dataset of n-grams learned between checkpoints
    dataset_name = f"{args.n}-grams-{args.start}-{args.end}-{args.model_name.replace('/', '--')}"
    if not Path(dataset_name).exists():
        f"Prompt dataset not found - use select_language_prompts.py to generate."
    dataset = load_from_disk(dataset_name) # type: ignore
    dataset = assert_type(Dataset, dataset)

    # Collect patching data
    daat_patch(args.model_name, args.start, args.end, args.n, dataset, out,  device)
    # prompt_patch(model_name, start, end, n, dataset, out, device)


if __name__ == "__main__":
    main()



def prompt_patch(model_name: str, start: int, end: int, n: int, dataset: Dataset, out: Path, device: torch.device):
    os.makedirs(out / 'prompt', exist_ok=True)

    model = get_patchable_model(model_name, get_pythia_revision(end), device, n - 1)

    circuit_data = {
        "ngram": [],  # This will be a list of lists (token indices)
        "prompt_edge_prune_scores": []  # This will be a list of tensors
    }
    for row in dataset:
        alternative_ngram_prefixes = torch.tensor(row['corrupt'], device=device)[:, :-1] # type: ignore

        learned_ngram = torch.tensor(row['clean'], device=device) # type: ignore
        learned_ngram_prefix_repeated = [learned_ngram[:-1] for _ in range(len(alternative_ngram_prefixes))]
        learned_ngram_suffix_repeated = [learned_ngram[-1:] for _ in range(len(alternative_ngram_prefixes))]
        
        patching_ds = PromptDataset(
            learned_ngram_prefix_repeated,
            alternative_ngram_prefixes,
            learned_ngram_suffix_repeated,
            learned_ngram_suffix_repeated
        )
        dataloader = PromptDataLoader(patching_ds, seq_len=n - 1, diverge_idx=0)

        print("Calculating EAP prune scores for patching corrupt n-grams into learned n-gram")
        edge_prune_scores: PruneScores = mask_gradient_prune_scores(
            model=model,
            dataloader=dataloader,
            official_edges=set(),
            grad_function="logit",
            answer_function="avg_val",
            mask_val=0.0
        )
        circuit_data["ngram"].append(learned_ngram)
        circuit_data["prompt_edge_prune_scores"].append(edge_prune_scores)


        # Vizualize circuits
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # visualize(model, edge_prune_scores, out / 'prompt', tokenizer.decode(learned_ngram))
