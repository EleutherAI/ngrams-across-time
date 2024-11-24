import os
from argparse import ArgumentParser
from pathlib import Path

import torch.nn.functional as F
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer
from nnsight import LanguageModel
from sae import SaeConfig, Sae, TrainConfig, SaeTrainer
from datasets import Dataset as HfDataset
import plotly.graph_objects as go
import lovely_tensors as lt

from ngrams_across_time.feature_circuits.circuit import get_mean_sae_entropy
from ngrams_across_time.utils.utils import assert_type, set_seeds
from ngrams_across_time.grok.transformers import CustomTransformer, TransformerConfig
from ngrams_across_time.grok.metrics import mean_l2, var_trace, gini, hoyer, hoyer_square
from ngrams_across_time.grok.conc_pythia import all_node_scores, abs_score_entropy

lt.monkey_patch()
set_seeds(598)
device = torch.device("cuda")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--inference", action="store_true")
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
    
    if args.inference:
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

            # Collect metrics
            checkpoint_data[epoch]['parameter_norm'] = torch.cat([parameters.flatten() for parameters in model.parameters()]).flatten().norm(p=2).item()
            checkpoint_data[epoch]['train_loss'] = F.cross_entropy(model(train_data[:len_sample_data].to(device)).logits.to(torch.float64)[:, -1], train_labels[:len_sample_data].to(device)).mean().item()
            checkpoint_data[epoch]['test_loss'] = F.cross_entropy(model(test_data[:len_sample_data].to(device)).logits.to(torch.float64)[:, -1], test_labels[:len_sample_data].to(device)).mean().item()

            nodes, fvu, multi_topk_fvu = get_mean_sae_entropy(
                nnsight_model, all_submods, dictionaries, 
                train_data, aggregate=True, batch_size=64
            )
            mean_fvu = np.mean([v.item() for v in fvu.values()])
            mean_multi_topk_fvu = np.mean([v.item() for v in multi_topk_fvu.values()])

            checkpoint_data[epoch][f'sae_fvu'] = mean_fvu
            checkpoint_data[epoch][f'sae_multi_topk_fvu'] = mean_multi_topk_fvu
            checkpoint_data[epoch][f'sae_entropy_nodes'] = {'nodes': nodes}   

            checkpoint_data[epoch][f'sae_entropy'] = abs_score_entropy(nodes)

            node_scores = all_node_scores(nodes)
            checkpoint_data[epoch][f'hoyer'] = hoyer(node_scores)
            checkpoint_data[epoch][f'hoyer_square'] = hoyer_square(node_scores)
            checkpoint_data[epoch][f'gini'] = gini(node_scores)

        torch.save(checkpoint_data, OUT_PATH)
        
    # Plot data
    checkpoint_data = torch.load(OUT_PATH)

    def add_scores_if_exists(fig, key: str, score_type: str | None, name: str, normalize: bool = True):
        # Check whether data exists
        first_epoch_data = checkpoint_data[checkpoint_epochs[0]]
        if score_type:
            if score_type not in first_epoch_data.keys():
                print(f"Skipping {name} because {score_type} does not exist in the checkpoint")
                return
            first_epoch_data = first_epoch_data[score_type]

        if key not in first_epoch_data.keys():
            print(f"Skipping {name} because it does not exist in the checkpoint")
            return
        
        # Plot data
        scores = (
            [scores[score_type][key] for scores in list(checkpoint_data.values())[:test_early_index]]
            if score_type
            else [scores[key] for scores in list(checkpoint_data.values())[:test_early_index]]
        )

        # Normalize scores to be between 0 and 1
        if normalize:
            max_score = max(scores)
            min_score = min(scores)
            scores = [(score - min_score) / (max_score - min_score) for score in scores]
        fig.add_trace(go.Scatter(x=checkpoint_epochs, y=scores, mode='lines', name=name))
            
    fig = go.Figure()

    # Underlying model and task stats
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss')
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss')
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm')
    # Sparsity metrics
    add_scores_if_exists(fig, 'sae_entropy', None, 'SAE Entropy') # normalize=False
    add_scores_if_exists(fig, 'hoyer', None, 'Hoyer')
    add_scores_if_exists(fig, 'hoyer_square', None, 'Hoyer Square')
    add_scores_if_exists(fig, 'gini', None, 'Gini coefficient')
    # SAE fraction of variance unexplained metrics
    add_scores_if_exists(fig, 'sae_fvu', None, 'SAE FVU')
    add_scores_if_exists(fig, 'sae_multi_topk_fvu', None, 'SAE Multi Top-K FVU')

    fig.update_layout(
        title=f"Grokking Loss and Node Entropy over Epochs for {run_identifier}", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        width=1000
    )                                           
    fig.write_image(images_path / f'detect_sae_entropy_{run_identifier}.pdf', format='pdf')

    
if __name__ == "__main__":
    main()