import os
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from nnsight import LanguageModel
from sae import SaeConfig, Sae, TrainConfig, SaeTrainer
from datasets import Dataset as HfDataset
import lovely_tensors as lt

# from ngrams_across_time.feature_circuits.circuit import get_mean_sae_entropy
from ngrams_across_time.utils.utils import assert_type, set_seeds
from ngrams_across_time.clearnets.transformers import CustomTransformer, TransformerConfig

from ngrams_across_time.clearnets.inference.mnist_vit import concatenate_values, get_sparsity_metrics, get_sae_acts
from ngrams_across_time.clearnets.plot.plot_sparsity import plot_sparsity

import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469
lt.monkey_patch()
set_seeds(598)
device = torch.device("cuda")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_seed", type=int, default=1, help="Model seed to load checkpoints for")
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--sae_name", type=str, default='')
    return parser.parse_args()


def main():
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)

    args = get_args()
    run_identifier = f"{args.model_seed}{'-' + args.run_name if args.run_name else ''}"

    # Use existing on-disk model checkpoints
    MODEL_PATH = Path(f"workspace/grok/{run_identifier}.pth")
    cached_data = torch.load(MODEL_PATH)
    
    OUT_PATH = Path(f"workspace/inference/{run_identifier}.pth")
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

    model_checkpoints = cached_data["checkpoints"][::5]
    checkpoint_epochs = cached_data["checkpoint_epochs"][::5]

    test_early_index = len(model_checkpoints)
    model_checkpoints = model_checkpoints[:test_early_index]
    checkpoint_epochs = checkpoint_epochs[:test_early_index]    
    
    checkpoint_data = torch.load(OUT_PATH) if OUT_PATH.exists() else {}

    config = cached_data['config']
    config = assert_type(TransformerConfig, config)

    dataset = cached_data['dataset']
    labels = cached_data['labels']
    train_indices = cached_data['train_indices']
    test_indices = cached_data['test_indices']

    train_data = dataset[train_indices]
    train_labels = labels[train_indices]
    print(len(train_data), "train data items")

    test_data = dataset[test_indices]
    test_labels = labels[test_indices]
    
    # SAEs get lower FVU on multiple epochs
    num_epochs = 128
    sae_train_dataset = HfDataset.from_dict({
        "input_ids": train_data.repeat(num_epochs, 1),
        "labels": train_labels.repeat(num_epochs),
    })
    sae_train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    # Subset of data to collect eval metrics
    len_sample_data = 4 if args.debug else 40

    model = CustomTransformer(config)

    # NNSight requires a tokenizer. We are passing in tensors so any tokenizer will do
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))):
        if epoch not in checkpoint_data:
            checkpoint_data[epoch] = {}

        # Load model and SAEs
        model.load_state_dict(state_dict)
        model.to(device) # type: ignore
        nnsight_model = LanguageModel(model, tokenizer=tokenizer)

        sae_path = Path(f"sae/{run_identifier}{'_' + args.sae_name if args.sae_name else ''}/grok.{epoch}")
        if not sae_path.exists():
            cfg = TrainConfig(
                SaeConfig(multi_topk=True, expansion_factor=128, k=model.config.d_model), 
                batch_size=64,
                run_name=str(sae_path),
                log_to_wandb=True,
                hookpoints=[
                    "blocks.*.hook_resid_pre", # start of block
                    "blocks.*.hook_attn_out", # after ln / attention
                    "blocks.*.hook_mlp_out" # basically the same as resid_post but without the resid added
                ],
            )
            trainer = SaeTrainer(cfg, sae_train_dataset, model)
            trainer.fit()
        
        attns = [block.hook_attn_out for block in nnsight_model.blocks]
        mlps = [block.hook_mlp_out for block in nnsight_model.blocks]

        dictionaries = {}
        for i in range(len(nnsight_model.blocks)): # type: ignore
            dictionaries[attns[i]] = Sae.load_from_disk(
                os.path.join(sae_path, f'blocks.{i}.hook_attn_out'),
                device=device
            )
            dictionaries[mlps[i]] = Sae.load_from_disk(
                os.path.join(sae_path, f'blocks.{i}.hook_mlp_out'),
                device=device
            )
        all_submods = [submod for layer_submods in zip(attns, mlps) for submod in layer_submods]

        dataset = TensorDataset(train_data[:len_sample_data])
        dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

        nodes, fvu, multi_topk_fvu, metrics = get_sae_acts(
            nnsight_model, all_submods, dictionaries, 
            dataloader, seq_len=len(train_data[0]), aggregate=True,
        )
        
        # Collect metrics
        checkpoint_data[epoch]['parameter_norm'] = torch.cat([parameters.flatten() for parameters in model.parameters()]).flatten().norm(p=2).item()
        checkpoint_data[epoch]['train_loss'] = F.cross_entropy(model(train_data[:len_sample_data].to(device)).logits.to(torch.float64)[:, -1], train_labels[:len_sample_data].to(device)).mean().item()
        checkpoint_data[epoch]['test_loss'] = F.cross_entropy(model(test_data[:len_sample_data].to(device)).logits.to(torch.float64)[:, -1], test_labels[:len_sample_data].to(device)).mean().item()
        
        checkpoint_data[epoch][f'sae_fvu'] = np.mean([v.item() for v in fvu.values()])
        checkpoint_data[epoch][f'sae_multi_topk_fvu'] = np.mean([v.item() for v in multi_topk_fvu.values()])
        
        checkpoint_data[epoch][f'sae_entropy_nodes'] = {'nodes': nodes}   

        node_scores = concatenate_values(nodes)

        checkpoint_data[epoch].update(get_sparsity_metrics(node_scores))

    torch.save(checkpoint_data, OUT_PATH)

    if args.plot:
        # Plots data over five model seeds
        plot_sparsity()
        

if __name__ == "__main__":
    main()