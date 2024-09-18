import os
from typing import Any

import torch.nn.functional as F
import torch
import einops
from tqdm import tqdm
from ngrams_across_time.grok.transformers import CustomTransformer, TransformerConfig

from sae import SaeConfig, SaeTrainer, TrainConfig, Sae
from datasets import Dataset as HfDataset
import lovely_tensors as lt
lt.monkey_patch()

device = torch.device("cuda")

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
    
    # the SAEs like epochs
    num_epochs = 32
    sae_train_dataset = HfDataset.from_dict({
        "input_ids": train_data.repeat(num_epochs, 1),
        "labels": train_labels.repeat(num_epochs),
    })
    sae_train_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    # Define model with existing on-disk checkpoints
    PTH_LOCATION = "workspace/grokking_demo.pth"
    cached_data = torch.load(PTH_LOCATION)
    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data["checkpoint_epochs"]

    model = CustomTransformer(TransformerConfig(
        vocab_size=p + 1,
        hidden_size=128,
        n_ctx=5,
        num_layers=1,
        num_heads=4,
        intermediate_size=512,
        act_type = "ReLU",
        use_ln = False
    ))

    for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))[::10]):
        model.load_state_dict(state_dict)
        model.cuda() # type: ignore

        # Train SAE on checkpoints
        run_name = f"sae/grok.{epoch}"
        if not os.path.exists(run_name):
            cfg = TrainConfig(
                SaeConfig(multi_topk=True), 
                batch_size=16,
                run_name=run_name,
                log_to_wandb=False,
                hookpoints=[
                    "blocks.0", 
                    "blocks.*.hook_attn_out", 
                    "blocks.*.hook_mlp_out",
                    "blocks.*.hook_resid_post"
                ],
            )
            trainer = SaeTrainer(cfg, sae_train_dataset, model)
            trainer.fit()


if __name__ == "__main__":
    main()