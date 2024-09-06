# TODO train an SAE on each checkpoint
from functools import partial
from copy import deepcopy

import torch.nn.functional as F
import torch
from torch import Tensor
import numpy as np
from transformers import AutoModelForCausalLM
import einops
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
from sae import SaeConfig, SaeTrainer, TrainConfig, Sae
from datasets import Dataset as HfDataset
from transformers import GPT2Config

from ngrams_across_time.utils import assert_type

from auto_circuit.data import PromptDataset, PromptDataLoader
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores, AblationType
from auto_circuit.prune import patch_mode, run_circuits

device = torch.device("cuda")

from contextlib import contextmanager

@contextmanager
def patchable(model):
    def set_patchable(hooked_model, state: bool):
        hooked_model.set_use_attn_result(state)
        hooked_model.set_use_hook_mlp_in(state)
        hooked_model.set_use_split_qkv_input(state)
        return hooked_model
    
    try:
        set_patchable(model, True)
        yield model
    finally:
        set_patchable(model, False)


def hooked_to_hf(
        hooked_model: HookedTransformer, 
        hooked_cfg: HookedTransformerConfig
    ):

    hf_config = GPT2Config(
        vocab_size=hooked_cfg.d_vocab,
        n_positions=max(5, hooked_cfg.n_ctx), # minimum required by transformers
        n_embd=hooked_cfg.d_model,
        n_layer=hooked_cfg.n_layers,
        n_head=hooked_cfg.n_heads,
        activation_function="relu",
        resid_pdrop = 0.,
        embd_pdrop = 0.,
        attn_pdrop = 0.,    
    )

    hf_model = AutoModelForCausalLM.from_config(hf_config)

    # Map the state dict
    state_dict = hooked_model.state_dict()
    new_state_dict = {}
    
    # Embedding layers
    new_state_dict['transformer.wte.weight'] = state_dict['embed.W_E']
    wpe_weight = state_dict['pos_embed.W_pos']
    if wpe_weight.shape[0] < hf_config.n_positions:
        wpe_weight = torch.cat([wpe_weight, torch.zeros(hf_config.n_positions - wpe_weight.shape[0], 128, device=device)])
    new_state_dict['transformer.wpe.weight'] = wpe_weight
    
    for layer in range(hooked_cfg.n_layers):
        # Attention layers
        new_state_dict[f'transformer.h.{layer}.attn.c_attn.weight'] = torch.cat([
            state_dict[f'blocks.{layer}.attn.W_Q'],
            state_dict[f'blocks.{layer}.attn.W_K'],
            state_dict[f'blocks.{layer}.attn.W_V']
        ], dim=0).reshape(hooked_cfg.d_model, -1)

        # 12, 32 -> 384
        new_state_dict[f'transformer.h.{layer}.attn.c_attn.bias'] = torch.cat([
            state_dict[f'blocks.{layer}.attn.b_Q'],
            state_dict[f'blocks.{layer}.attn.b_K'],
            state_dict[f'blocks.{layer}.attn.b_V']
        ]).reshape(-1)
        # [4, 32, 128] -> [128, 128]
        new_state_dict[f'transformer.h.{layer}.attn.c_proj.weight'] = state_dict[f'blocks.{layer}.attn.W_O'].reshape(hooked_cfg.d_model, -1)
        new_state_dict[f'transformer.h.{layer}.attn.c_proj.bias'] = state_dict[f'blocks.{layer}.attn.b_O']
    
        # MLP layers
        new_state_dict[f'transformer.h.{layer}.mlp.c_fc.weight'] = state_dict[f'blocks.{layer}.mlp.W_in'] #.t()
        new_state_dict[f'transformer.h.{layer}.mlp.c_fc.bias'] = state_dict[f'blocks.{layer}.mlp.b_in']
        new_state_dict[f'transformer.h.{layer}.mlp.c_proj.weight'] = state_dict[f'blocks.{layer}.mlp.W_out'] #.t()
        new_state_dict[f'transformer.h.{layer}.mlp.c_proj.bias'] = state_dict[f'blocks.{layer}.mlp.b_out']
        
        # Layer norms (if present)
        if 'transformer.ln_f.weight' in hf_model.state_dict():
            new_state_dict['transformer.ln_f.weight'] = torch.ones(hf_config.n_embd)
            new_state_dict['transformer.ln_f.bias'] = torch.zeros(hf_config.n_embd)
        if f'transformer.h.{layer}.ln_1.weight' in hf_model.state_dict():
            new_state_dict[f'transformer.h.{layer}.ln_1.weight'] = torch.ones(hf_config.n_embd)
            new_state_dict[f'transformer.h.{layer}.ln_1.bias'] = torch.zeros(hf_config.n_embd)
        if f'transformer.h.{layer}.ln_2.weight' in hf_model.state_dict():
            new_state_dict[f'transformer.h.{layer}.ln_2.weight'] = torch.ones(hf_config.n_embd)
            new_state_dict[f'transformer.h.{layer}.ln_2.bias'] = torch.zeros(hf_config.n_embd)
        
    # LM head 
    # TransformerLens model removes the equals from the vocab, don't think this is supported by transformers lib
    new_state_dict['lm_head.weight'] = torch.concat([state_dict['unembed.W_U'].t(), torch.zeros(1, 128, device=device)])
    
    # Load the new state dict
    hf_model.load_state_dict(new_state_dict)
    
    return hf_model


def set_patchable(hooked_model, state: bool):
    hooked_model.set_use_attn_result(state)
    hooked_model.set_use_hook_mlp_in(state)
    hooked_model.set_use_split_qkv_input(state)
    return hooked_model

def main():
    DATA_SEED = 598
    torch.manual_seed(DATA_SEED)

    # Generate dataset
    p = 113
    frac_train = 0.3
    a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
    b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
    equals_vector = einops.repeat(torch.tensor(113), " -> (i j)", i=p, j=p)
    dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
    labels = (dataset[:, 0] + dataset[:, 1]) % p

    indices = torch.randperm(p*p)
    cutoff = int(p*p*frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    train_data = dataset[train_indices]
    train_labels = labels[train_indices]
    test_data = dataset[test_indices]
    test_labels = labels[test_indices]

    num_epochs = 32
    train_dataset = HfDataset.from_dict({
        "input_ids": train_data.repeat(num_epochs, 1),
        "labels": train_labels.repeat(num_epochs),
    })
    train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    # Define model
    hooked_cfg = HookedTransformerConfig(
        n_layers = 1,
        n_heads = 4,
        d_model = 128,
        d_head = 32,
        d_mlp = 512,
        act_fn = "relu",
        normalization_type=None,
        d_vocab=p+1,
        d_vocab_out=p,
        n_ctx=3,
        init_weights=True,
        device=device.type,
        seed = 999,
    )
    hooked_model = HookedTransformer(hooked_cfg)
    
    # Model checkpoints
    PTH_LOCATION = "workspace/_scratch/grokking_demo.pth"
    cached_data = torch.load(PTH_LOCATION)
    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data["checkpoint_epochs"]

    checkpoint_ablation_loss_increases = []
    for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))[::10]):
        # Load model checkpoint
        hooked_model.reset_hooks()
        hooked_model.load_state_dict(state_dict)

        # Train SAE on checkpoints
        hf_transformer = hooked_to_hf(hooked_model, hooked_cfg).to(device)
        
        run_name = f"grok.{epoch}"
        cfg = TrainConfig(
            SaeConfig(multi_topk=True), 
            batch_size=16,
            run_name=run_name,
        )

        trainer = SaeTrainer(cfg, train_dataset, hf_transformer)
        trainer.fit()

        # Load SAE
        sae = Sae.load_from_disk(run_name)

        # Run EAP on Patchable model with SAEs hooked in
        with patchable(hooked_model) as model:
            ablation_model = patchable_model(deepcopy(model), factorized=True, device=device, separate_qkv=True, seq_len=train_data.shape[-1], slice_output="last_seq")
        
        loss_increases = []
        answers = [batch.unsqueeze(dim=0) for batch in list(train_labels.unbind())]

        patching_ds = PromptDataset(train_data[:-1], train_data[1:], answers[:-1], answers[1:])
        dataloader = PromptDataLoader(patching_ds, seq_len=train_data.shape[-1], diverge_idx=0)
        
        edge_prune_scores: PruneScores = mask_gradient_prune_scores(model=ablation_model, dataloader=dataloader,official_edges=set(),grad_function="logit",answer_function="avg_diff",mask_val=0.0)
        num_edges = 10
        logits = run_circuits(ablation_model, dataloader, [num_edges], prune_scores=edge_prune_scores, ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT)
        batch_logits = list(logits.values())[0] # this key will be edges - num_edges
        batch_logits = assert_type(dict, batch_logits)
        
        for batch in dataloader: 
            patched_logits = batch_logits[batch.key]
            unpatched_logits = model(batch.clean)[:, -1, :]
            patched_loss = F.cross_entropy(patched_logits.to(device).squeeze(), batch.answers.to(device).squeeze())
            unpatched_loss = F.cross_entropy(unpatched_logits.squeeze(), batch.answers.to(device).squeeze())
            loss_increases.append((patched_loss - unpatched_loss).item())

        checkpoint_ablation_loss_increases.append(np.mean(loss_increases))                
        print(checkpoint_ablation_loss_increases[-1])
        
        def reconstr_hook(activations, hook, sae_out):
            return sae(cache[sae.cfg.hook_name])

        # Run fwd pass on model with SAE hooked in
        print(
            hooked_model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[
                    (
                        sae.cfg.hook_name,
                        partial(reconstr_hook, sae_out=sae_out),
                    )
                ],
                return_type="loss",
            ).item(),
        )


if __name__ == "__main__":
    # sae = Sae.load_from_disk("sae-ckpts/transformer.h.0")
    # load the model
    main()