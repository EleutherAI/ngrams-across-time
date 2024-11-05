import os
import math
from argparse import ArgumentParser
from pathlib import Path

import torch.nn.functional as F
from torchvision import transforms
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer
from nnsight import LanguageModel, NNsight
from datasets import load_dataset
from sae import SaeConfig, Sae, TrainConfig, SaeTrainer
from datasets import Dataset as HfDataset
import plotly.graph_objects as go
import lovely_tensors as lt
from transformers import ViTForImageClassification, ViTConfig

from ngrams_across_time.feature_circuits.circuit import get_circuit
from ngrams_across_time.utils.utils import assert_type, set_seeds
from ngrams_across_time.feature_circuits.patch_nodes import patch_nodes
from ngrams_across_time.feature_circuits.sae_loss import sae_loss
from ngrams_across_time.grok.detect import get_top_sae_nodes, get_top_residual_nodes, all_node_scores, abs_score_entropy

def get_test_dataset():
    train_dataset: HfDataset = load_dataset("mnist", split='train') # type: ignore

    def map_fn(ex):
        return {
            'input_ids': transforms.ToTensor()(ex['image']),
            'label': ex['label']
        }

    train_dataset = train_dataset.map(
        function=map_fn,
        remove_columns=['image'],
        new_fingerprint='transformed_mnist', # type: ignore
        keep_in_memory=True # type: ignore
    )
    train_dataset = train_dataset.with_format('torch')
    train_dataset.set_format(type='torch', columns=['input_ids', 'label'])

    print("Final columns:", train_dataset.column_names)

    # Calculate mean and std of pixel values
    input_ids = assert_type(Tensor, train_dataset['input_ids'])
    mean = input_ids.mean().item()
    std = input_ids.std().item()
    def normalize(image):
        transform = transforms.Compose([
            transforms.Normalize((mean,), (std,))
        ])
        return transform(image)


    test_dataset: HfDataset = load_dataset('mnist', split='test') # type: ignore

    test_dataset = test_dataset.map(
        function=map_fn,
        remove_columns=['image'],
        new_fingerprint='transformed_mnist', # type: ignore
        keep_in_memory=True # type: ignore
    )
    test_dataset.set_format(type='torch', columns=['input_ids', 'label'])

    test_dataset = test_dataset.map(
        lambda example: {'input_ids': normalize(example['input_ids'])},
        new_fingerprint='transformed_mnist'
    )

    return test_dataset



lt.monkey_patch()
set_seeds(598)
device = torch.device("cuda")


def min_nodes_to_random(node_scores, node_type, train_data, language_model, 
                        all_submods, metric_fn, dictionaries, num_classes):
    random_loss = -math.log(1/num_classes)

    start = 0
    end = len(all_node_scores(node_scores))
    mid = (start + end) // 2
    while start < end:
        if node_type == "residual":
            top_nodes = get_top_residual_nodes(node_scores, mid)
        else:
            top_nodes = get_top_sae_nodes(node_scores, mid)
        # Always hit max number of nodes with zero ablations because there is some information in every node
        # breakpoint()
        loss, patch_loss, clean_loss = patch_nodes(
            train_data, None, language_model, all_submods, dictionaries, metric_fn, top_nodes, dummy_input=train_data[:3]
            )
        if loss.mean().item() >= random_loss:
            end = mid
        else:
            start = mid + 1
        mid = (start + end) // 2
    return mid


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_seed", type=int, default=1, help="Model seed to load checkpoints for")
    parser.add_argument("--run_name", type=str, default='')
    return parser.parse_args()


def main():
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)

    args = get_args()
    run_identifier = f'mnist_seed_{args.model_seed}'

    # Use existing on-disk model checkpoints
    MODEL_PATH = Path("vit_ckpts") / run_identifier / "final.pth"
    cached_data = torch.load(MODEL_PATH)
    
    OUT_PATH = Path(f"workspace") / 'inference' / run_identifier / "inference.pth"
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data["checkpoint_epochs"]
    
    config = cached_data['config']
    config = assert_type(ViTConfig, config)

    model = cached_data['model']
    checkpoints = cached_data['checkpoints']
    checkpoint_epochs = cached_data['checkpoint_epochs']
    dataset = cached_data['dataset']
    
    train_data = dataset['input_ids']
    print(len(train_data))
    train_labels = dataset['label']

    test_dataset= get_test_dataset()
    test_data = test_dataset['input_ids']
    test_labels = test_dataset['label']

    num_classes = 10

    if args.inference:
        # SAEs like multiple epochs
        num_epochs = 128
        sae_train_dataset = HfDataset.from_dict({
            "input_ids": train_data.repeat(num_epochs, 1, 1, 1), # .flatten(1, 3)
            "label": train_labels.repeat(num_epochs),
        })
        sae_train_dataset.set_format(type="torch", columns=["input_ids", "label"])
        dummy_inputs={'pixel_values': torch.randn(3, 1, 28, 28)}
        
        train_data = train_data.cuda()
        train_labels = train_labels.cuda()
        
        def metric_fn(logits, repeat: int = 1):
            labels = train_labels.repeat(repeat)
            return (
                F.cross_entropy(logits.to(torch.float64), labels, reduction="none")
            )
        
        model = ViTForImageClassification(config).to(device)

        # NNSight requires a tokenizer, we are passing in tensors so any tokenizer will do
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        checkpoint_data = torch.load(OUT_PATH) if OUT_PATH.exists() else {}
    
        for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))):
            model.load_state_dict(state_dict)
            model.cuda() # type: ignore

            if epoch not in checkpoint_data:
                checkpoint_data[epoch] = {}

            checkpoint_data[epoch]['parameter_norm'] = torch.cat([parameters.flatten() for parameters in model.parameters()]).flatten().norm(p=2).item()
            checkpoint_data[epoch]['train_loss'] = F.cross_entropy(model(train_data.cuda()).logits.to(torch.float64), train_labels.cuda()).mean().item()

            test_losses = []
            for i in range(0, len(test_data), 64):
                test_losses.append(F.cross_entropy(model(test_data[i:i + 64].cuda()).logits.to(torch.float64), test_labels[i:i+64].cuda()).mean().item())
            checkpoint_data[epoch]['test_loss'] = np.mean(test_losses)

            nnsight_model = NNsight(model)
            embed_hookpoints = ["vit.embeddings"]
            attn_hookpoints = [f"vit.encoder.layer.{i}.attention.output" for i in range(len(nnsight_model.vit.encoder.layer))]
            mlp_hookpoints = [f"vit.encoder.layer.{i}.output" for i in range(len(nnsight_model.vit.encoder.layer))]
            resid_hookpoints = [f"vit.encoder.layer.{i}.layernorm_after" for i in range(len(nnsight_model.vit.encoder.layer))]

            sae_path = Path(f"sae/vit/{run_identifier}/{epoch}")
            if not sae_path.exists():
                cfg = TrainConfig(
                    SaeConfig(multi_topk=True), 
                    batch_size=64,
                    run_name=str(sae_path),
                    log_to_wandb=not args.debug,
                    hookpoints=embed_hookpoints + attn_hookpoints + mlp_hookpoints + resid_hookpoints,
                )
                trainer = SaeTrainer(cfg, sae_train_dataset, model, dummy_inputs)
                trainer.fit()
            
            embed = nnsight_model.vit.embeddings
            resids = [block.layernorm_after for block in nnsight_model.vit.encoder.layer]
            attns = [block.attention.output for block in nnsight_model.vit.encoder.layer]
            mlps = [block.output for block in nnsight_model.vit.encoder.layer]
            all_submods = [embed] + [submod for layer_submods in zip(attns, mlps, resids) for submod in layer_submods]

            dictionaries = {}
            dictionaries[embed] = Sae.load_from_disk(
                os.path.join(sae_path, 'vit.embeddings'),
                device=device
            )
            for i in range(len(nnsight_model.vit.encoder.layer)): # type: ignore
                dictionaries[resids[i]] = Sae.load_from_disk(
                    os.path.join(sae_path, f'vit.encoder.layer.{i}.layernorm_after'),
                    device=device
                )

                dictionaries[attns[i]] = Sae.load_from_disk(
                    os.path.join(sae_path, f'vit.encoder.layer.{i}.attention.output'),
                    device=device
                )

                dictionaries[mlps[i]] = Sae.load_from_disk(
                    os.path.join(sae_path, f'vit.encoder.layer.{i}.output'),
                    device=device
                )            

            loss_all_saes, mean_res_norm, mse = sae_loss(
                train_data, nnsight_model, all_submods, dictionaries, metric_fn, dummy_inputs=dummy_inputs['pixel_values']
            )
            checkpoint_data[epoch]['sae_loss'] = loss_all_saes.mean().item()
            checkpoint_data[epoch]['sae_mse'] = mse
            checkpoint_data[epoch]['mean_res_norm'] = mean_res_norm
            
            for method in ["ig", "grad", "attrib"]:
                # Residual node scores with zero ablation
                nodes = get_circuit(train_data, None, nnsight_model, 
                all_submods, {}, metric_fn, aggregate=True, method=method)
                checkpoint_data[epoch][f'residual_{method}_zero'] = {'nodes': nodes}

                # SAE node scores with zero ablation
                nodes = get_circuit(train_data, None, 
                                        nnsight_model, all_submods, dictionaries, metric_fn, aggregate=True,
                                        method=method)
                checkpoint_data[epoch][f'sae_{method}_zero'] = {'nodes': nodes}

                # Gradient scores don't use an activation delta
                if method == "grad":
                    continue

            # A random classifier's accuracy is 1/113 = 0.0088, resulting in a loss of 4.7330
            random_loss = -math.log(1/113)
            average_loss = checkpoint_data[epoch]['train_loss']
            loss_increase_to_random = random_loss - average_loss

            for node_score_type in [
                    'residual_ig_zero', 'sae_ig_zero', 'sae_attrib_zero', 'residual_attrib_zero', 'residual_grad_zero', 
                    'sae_grad_zero'
                ]:
                nodes = checkpoint_data[epoch][node_score_type]['nodes']
                
                # 1. Entropy measure
                # 2. Linearized approximation of number of nodes to ablate to achieve random accuracy
                checkpoint_data[epoch][node_score_type]['entropy'] = abs_score_entropy(nodes)

                scores = all_node_scores(nodes)
                sorted_scores = np.sort(scores)[::-1]
                positive_scores = sorted_scores[sorted_scores > 0]
                proxy_nodes_to_ablate = np.searchsorted(np.cumsum(positive_scores), loss_increase_to_random)
                checkpoint_data[epoch][node_score_type][f'linearized_circuit_feature_count'] = proxy_nodes_to_ablate

                # Binary search for the number of nodes to ablate to reach random accuracy
                node_type = 'residual' if 'residual' in node_score_type else 'sae'

                num_features = min_nodes_to_random(
                    checkpoint_data[epoch][node_score_type]['nodes'], node_type, train_data,
                    nnsight_model, all_submods, metric_fn, dictionaries,
                    num_classes=num_classes
                )
                checkpoint_data[epoch][node_score_type][f'circuit_feature_count'] = num_features
                # TODO lucia track residual score norm

        torch.save(checkpoint_data, OUT_PATH)
        
    checkpoint_data = torch.load(OUT_PATH)

    def add_scores_if_exists(fig, key: str, score_type: str | None, name: str, normalize: bool = True):
        if score_type and score_type not in checkpoint_data[checkpoint_epochs[0]].keys():
            print(f"Skipping {name} because {score_type} does not exist in the checkpoint")
            return

        if (score_type and key not in checkpoint_data[checkpoint_epochs[0]][score_type].keys()) or (
            not score_type and key not in checkpoint_data[checkpoint_epochs[0]].keys()
        ):
            print(f"Skipping {name} because it does not exist in the checkpoint")
            return

        scores = (
            [scores[score_type][key] for scores in checkpoint_data.values()]
            if score_type
            else [scores[key] for scores in checkpoint_data.values()]
        )

        if normalize:
            # Normalize scores to be between 0 and 1
            max_score = max(scores)
            min_score = min(scores)
            scores = [(score - min_score) / (max_score - min_score) for score in scores]
        fig.add_trace(go.Scatter(x=checkpoint_epochs, y=scores, mode='lines', name=name))
            
    
    for node_score_type in [
        'residual_ig_zero', 'sae_ig_zero', 'sae_attrib_zero', 'residual_attrib_zero',
        'residual_grad_zero', 'sae_grad_zero'
    ]: 
        fig = go.Figure()
        # Underlying model and task stats
        add_scores_if_exists(fig, 'train_loss', None, 'Train loss')
        add_scores_if_exists(fig, 'test_loss', None, 'Test loss')
        add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm')
        
        type = 'IG' if 'ig' in node_score_type else 'AtP'
        node_type = 'Residual' if 'residual' in node_score_type else 'SAE'

        add_scores_if_exists(fig, 'entropy', node_score_type, f'H({node_type} Nodes) with 0 ablation, {type}')
        # Nodes to ablate to reach random classifier performance
        add_scores_if_exists(fig, 'linearized_circuit_feature_count', node_score_type, f'Proxy num nodes in circuit, {ablation}, {type}, {node_type}')
        add_scores_if_exists(fig, 'circuit_feature_count', node_score_type, f'Num nodes in circuit, 0 ablation, {type}, {node_type}')
        
        fig.update_layout(
            title=f"Grokking Loss and Node Entropy over Epochs for seed {run_identifier} and node score type {node_score_type.replace('_', ' ')}", 
            xaxis_title="Epoch", 
            yaxis_title="Loss",
            width=1000
        )                                           
        fig.write_image(images_path / f'detect_{node_score_type}_{run_identifier}.pdf', format='pdf')

    
if __name__ == "__main__":
    main()