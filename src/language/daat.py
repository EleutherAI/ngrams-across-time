from typing import Literal, Optional, Any
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import DataLoader
from auto_circuit.types import AblationType, BatchKey, PruneScores
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    patch_mode,
    set_all_masks,
    train_mask_mode,
)
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff, batch_avg_answer_val


@dataclass(frozen=True)
class PromptAblationBatch:
    key: BatchKey
    batch_diverge_idx: int
    prompts: Tensor
    answers: list[Tensor] | Tensor
    ablations: dict[int, Tensor]
    wrong_answers: None


@dataclass(frozen=True)
class PromptAblationItem:
    prompt: Tensor
    ablations: dict[int, Tensor]
    answers: Tensor


def collate_fn(batch: list[Any]) -> PromptAblationBatch:
    prompts = torch.stack([p.prompt for p in batch]).to(batch[0].prompt.device)

    ablations: dict[int, Tensor] = {}
    for key in batch[0].ablations.keys():
        ablations[key] = torch.cat([item.ablations[key] for item in batch], dim=1).to(prompts.device)

    key = hash((str(prompts.tolist())))

    answers = torch.stack([p.answers for p in batch]).to(prompts.device)

    return PromptAblationBatch(
        key, 
        prompts.size(-1) - 1, 
        prompts, 
        answers,
        ablations,
        None
    )

class AblationDataset(Dataset):
    def __init__(self, prompts: list[Tensor], ablation_model: PatchableModel, max_len: int = 10):
        self.prompts = prompts
        self.ablation_model = ablation_model
        self.max_len = max_len

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx) -> PromptAblationItem:
        prompt = self.prompts[idx][-self.max_len:].cuda()
        return PromptAblationItem(
            prompt,
            src_ablations(self.ablation_model, prompt, ablation_type=AblationType.RESAMPLE),
            torch.tensor(len(prompt) - 1).cuda().unsqueeze(0)
        )


def mask_gradient_prune_scores_daat(
    model: PatchableModel,
    dataloader: DataLoader[PromptAblationBatch],
    grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
    answer_function: Literal["avg_diff", "avg_val", "mse"],
    mask_val: Optional[float] = None,
    integrated_grad_samples: Optional[int] = None,
) -> PruneScores:
    """
    Prune scores equal to the gradient of the mask values that interpolates the edges
    between the clean activations and the ablated activations.

    Args:
        model: The model to find the circuit for.
        dataloader: The dataloader to use for input.
        official_edges: Not used.
        grad_function: Function to apply to the logits before taking the gradient.
        answer_function: Loss function of the model output which the gradient is taken
            with respect to.
        mask_val: Value of the mask to use for the forward pass. Cannot be used if
            `integrated_grad_samples` is not `None`.
        integrated_grad_samples: If not `None`, we compute an approximation of the
            Integrated Gradients
            [(Sundararajan et al., 2017)](https://arxiv.org/abs/1703.01365) of the model
            output with respect to the mask values. This is computed by averaging the
            mask gradients over `integrated_grad_samples` samples of the mask values
            interpolated between 0 and 1. Cannot be used if `mask_val` is not `None`.
        ablation_type: The type of ablation to perform.
        clean_corrupt: Whether to use the clean or corrupt inputs to calculate the
            ablations.

    Returns:
        An ordering of the edges by importance to the task. Importance is equal to the
            absolute value of the score assigned to the edge.

    Note:
        When `grad_function="logit"` and `mask_val=0` this function is exactly
        equivalent to
        [`edge_attribution_patching_prune_scores`][auto_circuit.prune_algos.edge_attribution_patching.edge_attribution_patching_prune_scores].
    """
    assert (mask_val is not None) ^ (integrated_grad_samples is not None)  # ^ means XOR

    with train_mask_mode(model):
        for sample in (pbar := tqdm(range((integrated_grad_samples or 0) + 1))):
            pbar.set_description_str(f"Sample: {sample}")
            
            # Interpolate the mask value if integrating gradients
            val = mask_val if mask_val is not None else sample / integrated_grad_samples # type: ignore
            set_all_masks(model, val)

            for batch in dataloader:
                patch_src_outs = {
                    k: v.clone().detach() for k, v in batch.ablations.items()
                }
                with patch_mode(model, patch_src_outs):
                    logits = model(batch.prompts)[model.out_slice]
                    if grad_function == "logit":
                        token_vals = logits
                    elif grad_function == "prob":
                        token_vals = torch.softmax(logits, dim=-1)
                    elif grad_function == "logprob":
                        token_vals = log_softmax(logits, dim=-1)
                    elif grad_function == "logit_exp":
                        numerator = torch.exp(logits)
                        denominator = numerator.sum(dim=-1, keepdim=True)
                        token_vals = numerator / denominator.detach()
                    else:
                        raise ValueError(f"Unknown grad_function: {grad_function}")

                    if answer_function == "avg_diff":
                        loss = -batch_avg_answer_diff(token_vals, batch)
                    elif answer_function == "avg_val":
                        loss = -batch_avg_answer_val(token_vals, batch)
                    elif answer_function == "mse":
                        breakpoint()
                        loss = torch.nn.functional.mse_loss(token_vals, batch.answers)
                    else:
                        raise ValueError(f"Unknown answer_function: {answer_function}")

                    loss.backward()

    prune_scores: PruneScores = {}
    for dest_wrapper in model.dest_wrappers:
        grad = dest_wrapper.patch_mask.grad
        assert grad is not None
        prune_scores[dest_wrapper.module_name] = grad.detach().clone()
    return prune_scores
