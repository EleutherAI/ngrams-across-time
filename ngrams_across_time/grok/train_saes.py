from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm
from ngrams_across_time.grok.transformers import CustomTransformer, TransformerConfig
from ngrams_across_time.utils.utils import assert_type
from safetensors.torch import load_model

from sae import SaeConfig, SaeTrainer, TrainConfig
from datasets import Dataset as HfDataset
import lovely_tensors as lt

lt.monkey_patch()
device = torch.device("cuda")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_seed", type=int, default=1, help="Model seed to load checkpoints for")
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune the model from the previous checkpoint\
                        or train from scratch") 
    parser.add_argument("--model_run_name", type=str, default='')
    parser.add_argument("--sae_run_name", type=str, default='')
    parser.add_argument("--sae_suffix", type=str, default='')
    return parser.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.model_seed)

    # Use existing on-disk model checkpoints
    model_identifier = f"{args.model_seed}{'-' + args.model_run_name if args.model_run_name else ''}"
    cached_data = torch.load(model_path := Path(f"workspace/grok/{model_identifier}.pth"))

    # Create SAE training dataset using multiple epochs over the training data
    num_epochs = 256 if args.finetune else 128
    train_indices = cached_data['train_indices']
    train_labels = cached_data['labels'][train_indices]
    train_data = cached_data['dataset'][train_indices]
    sae_train_dataset = HfDataset.from_dict({
        "input_ids": train_data.repeat(num_epochs, 1),
        "labels": train_labels.repeat(num_epochs),
    })
    sae_train_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    # Load models from existing on-disk checkpoints
    config = cached_data['config']
    config = assert_type(TransformerConfig, config)
    model = CustomTransformer(config)

    model_checkpoints = cached_data["checkpoints"][::10]
    checkpoint_epochs = cached_data["checkpoint_epochs"][::10]   

    for i, (epoch, state_dict) in tqdm(enumerate(zip(checkpoint_epochs, model_checkpoints))):
        model.load_state_dict(state_dict)
        model.cuda() # type: ignore

        # Train SAE on checkpoints
        run_name = Path(f"sae/{model_identifier}/grok.{epoch}\
{'-' + args.sae_run_name if args.sae_run_name else ''}\
{'.finetune' if args.finetune else ''}\
{'.' + args.sae_suffix if args.sae_suffix else ''}")
        if run_name.exists():
            continue

        cfg = TrainConfig(
            SaeConfig(multi_topk=True),
            batch_size=64,
            run_name=str(run_name),
            log_to_wandb=True,
            hookpoints=[
                "blocks.*.hook_resid_pre", # start of block
                "blocks.*.hook_resid_mid", # after ln / attention
                "blocks.*.hook_resid_post", # after ln / mlp
                "blocks.*.hook_mlp_out"
            ],
        )
        trainer = SaeTrainer(cfg, sae_train_dataset, model)
            
        # finetune from previous checkpoints
        if args.finetune and i > 0:
            prev_run_name = Path(f"sae/{model_identifier}/grok.{checkpoint_epochs[i - 1]}")
            for name, sae in trainer.saes.items():
                load_model(sae, f"{prev_run_name}/{name}/sae.safetensors", device=str(device))

            trainer.cfg.lr = 1e-5
        
        trainer.fit()


if __name__ == "__main__":
    main()