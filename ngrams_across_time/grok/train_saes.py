import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm
from ngrams_across_time.grok.transformers import CustomTransformer, TransformerConfig
from ngrams_across_time.utils.utils import assert_type

from sae import SaeConfig, SaeTrainer, TrainConfig
from datasets import Dataset as HfDataset
import lovely_tensors as lt
lt.monkey_patch()

device = torch.device("cuda")

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_seed", type=int, default=1, help="Model seed to load checkpoints for")
    parser.add_argument("--run_name", type=str, default='')
    return parser.parse_args()

def main():
    args = get_args()
    DATA_SEED = 598
    torch.manual_seed(DATA_SEED)

    run_identifier = f"{args.model_seed}{'-' + args.run_name if args.run_name else ''}"

    # Use existing on-disk model checkpoints
    MODEL_PATH = Path(f"workspace/grok/{run_identifier}.pth")
    cached_data = torch.load(MODEL_PATH)

    config = cached_data['config']
    config = assert_type(TransformerConfig, config)

    dataset = cached_data['dataset']
    labels = cached_data['labels']
    train_indices = cached_data['train_indices']
    test_indices = cached_data['test_indices']

    train_data = dataset[train_indices]
    train_labels = labels[train_indices]

    test_data = dataset[test_indices]
    test_labels = labels[test_indices]

    # SAEs like multiple epochs
    num_epochs = 32
    sae_train_dataset = HfDataset.from_dict({
        "input_ids": train_data.repeat(num_epochs, 1),
        "labels": train_labels.repeat(num_epochs),
    })
    sae_train_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    # Define model with existing on-disk checkpoints
    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data["checkpoint_epochs"]

    model = CustomTransformer(config)

    for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))[::10]):
        model.load_state_dict(state_dict)
        model.cuda() # type: ignore

        # Train SAE on checkpoints
        run_name = f"sae/{run_identifier}/grok.{epoch}"
        if not os.path.exists(run_name):
            cfg = TrainConfig(
                SaeConfig(multi_topk=True), 
                batch_size=16,
                run_name=run_name,
                log_to_wandb=True,
                hookpoints=[
                    "blocks.*.hook_resid_pre", # start of block
                    "blocks.*.hook_resid_mid", # after ln / attention
                    "blocks.*.hook_resid_post", # after ln / mlp
                    "blocks.*.hook_mlp_out"
                ],
            )
            trainer = SaeTrainer(cfg, sae_train_dataset, model)
            trainer.fit()


if __name__ == "__main__":
    main()