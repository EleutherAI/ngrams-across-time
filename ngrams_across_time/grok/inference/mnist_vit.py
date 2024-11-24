import os
from argparse import ArgumentParser
from pathlib import Path

from plotly.subplots import make_subplots
import torch.nn.functional as F
from torchvision import transforms
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer
from nnsight import NNsight
from datasets import load_dataset
from sae import SaeConfig, Sae, TrainConfig, SaeTrainer
from datasets import Dataset as HfDataset
import plotly.graph_objects as go
import lovely_tensors as lt
from transformers import ViTForImageClassification, ViTConfig
from torch.utils.data import DataLoader
import nnsight

from ngrams_across_time.utils.utils import assert_type, set_seeds
from ngrams_across_time.grok.metrics import mean_l2, var_trace, gini, hoyer, hoyer_square
from ngrams_across_time.grok.conc_pythia import all_node_scores, abs_score_entropy
from ngrams_across_time.feature_circuits.dense_act import to_dense

@torch.no_grad()
def get_sae_acts(
        model,
        ordered_submods: list,
        dictionaries,
        dataloader,
        num_patches: int,
        aggregate: bool = True, # or 'none' for not aggregating across sequence position
        # mean: bool = True,
        device: str | torch.device = torch.device("cuda"),
        input_key: str = 'input_ids',
):
    """Meaned over a large dataset"""
    # Collect metadata and preallocate tensors
    batch_size = 0
    is_tuple = {}
    accumulated_nodes = {}
    accumulated_fvu = {}
    accumulated_multi_topk_fvu = {}

    first_batch = next(iter(dataloader))
    with torch.amp.autocast(str(device)):  # Enable automatic mixed precision
        batch_size = first_batch[input_key].shape[0] # .save() unnecessary, shape inherits from tuple
        with model.trace(first_batch[input_key].to(device)):
            for submodule in ordered_submods:
                output = (
                    submodule.nns_output 
                    if hasattr(submodule, 'nns_output') 
                    else submodule.output
                )
                is_tuple[submodule] = isinstance(output, tuple) 

                num_latents = dictionaries[submodule].num_latents
                shape = (batch_size, num_latents) if aggregate else (batch_size, num_patches, num_latents)
                accumulated_nodes[submodule.path] = torch.zeros(shape, device="cpu", requires_grad=True).save()
                accumulated_fvu[submodule.path] = 0
                accumulated_multi_topk_fvu[submodule.path] = 0

    # Do inference
    total_samples = 0
    num_batches = 0
    mean_l2 = 0.
    var_trace = 0.

    for batch in dataloader:
        batch_nodes = {}
        batch_fvus = {}
        batch_multi_topk_fvus = {}
        
        with model.trace(batch[input_key].to(device)):
            for submodule in ordered_submods:
                num_latents = dictionaries[submodule].num_latents

                if hasattr(submodule, 'nns_output'):
                    input = submodule.nns_output if not is_tuple[submodule] else submodule.nns_output[0]
                else:
                    input = submodule.output if not is_tuple[submodule] else submodule.output[0]

                dictionary = dictionaries[submodule]

                flat_f = nnsight.apply(dictionary.forward, input.flatten(0, 1))
                batch_fvus[submodule.path] = flat_f.fvu.cpu().save()
                batch_multi_topk_fvus[submodule.path] = flat_f.multi_topk_fvu.cpu().save()

                latent_acts = flat_f.latent_acts.view(input.shape[0], input.shape[1], -1)
                latent_indices = flat_f.latent_indices.view(input.shape[0], input.shape[1], -1)
                dense = to_dense(latent_acts, latent_indices, num_latents)
                batch_nodes[submodule.path] = dense.cpu().save()
        
        for submodule in ordered_submods:
            # Aggregate across sequence positions (patches)
            if aggregate: 
                batch_nodes[submodule.path] = batch_nodes[submodule.path].sum(dim=1)

            accumulated_nodes[submodule.path] += batch_nodes[submodule.path]
            accumulated_fvu[submodule.path] += batch_fvus[submodule.path].item()
            accumulated_multi_topk_fvu[submodule.path] += batch_multi_topk_fvus[submodule.path].item()
            
        # Calculate mean L2 norm of all activations
        # breakpoint()
        acts = torch.cat([batch_nodes[submodule.path] for submodule in ordered_submods], dim=1)
        mean_l2 += torch.linalg.vector_norm(acts, ord=2, dim=1).mean()
        var_trace += acts.var(dim=1).sum()

        total_samples += batch_size
        num_batches += 1

    nodes = {k: v.sum(0) / total_samples for k, v in accumulated_nodes.items()}
    fvu = {k: v / num_batches for k, v in accumulated_fvu.items()}
    multi_topk_fvu = {k: v / num_batches for k, v in accumulated_multi_topk_fvu.items()}
    mean_l2 = mean_l2 / num_batches
    var_trace = var_trace / total_samples


    return nodes, fvu, multi_topk_fvu, mean_l2, var_trace



def compute_losses(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)

            logits = outputs.logits.to(torch.float64)
            logits = logits.reshape(-1, logits.size(-1))
            
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


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


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_seed", type=int, default=1, help="Model seed to load checkpoints for")
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--sae_name", type=str, default='v3')
    return parser.parse_args()


def main():
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)

    batch_size = 40 # Dataset is only of len 40 to cause overfitting

    args = get_args()
    run_identifier = f'mnist_seed_{args.model_seed}'

    # Use existing on-disk model checkpoints
    MODEL_PATH = Path("data/vit_ckpts") / run_identifier / "final.pth"
    cached_data = torch.load(MODEL_PATH)
    
    OUT_PATH = Path(f"workspace") / 'inference' / run_identifier / "inference.pth"
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data["checkpoint_epochs"]
    
    config = cached_data['config']
    config = assert_type(ViTConfig, config)

    model = ViTForImageClassification(config)

    checkpoints = cached_data['checkpoints']
    checkpoint_epochs = cached_data['checkpoint_epochs']
    dataset = cached_data['dataset']

    if args.inference:
        # TODO increase data size
        train_data = dataset['input_ids']
        print(len(train_data))
        train_labels = dataset['label']

        test_dataset= get_test_dataset()
        test_data = test_dataset['input_ids']
        test_labels = test_dataset['label']

        # SAEs like multiple epochs
        num_epochs = 128
        sae_train_dataset: HfDataset = HfDataset.from_dict({
            "input_ids": train_data.repeat(num_epochs, 1, 1, 1), # .flatten(1, 3)
            "label": train_labels.repeat(num_epochs),
        })
        sae_train_dataset.set_format(type="torch", columns=["input_ids", "label"])
        dummy_inputs={'pixel_values': torch.randn(3, 1, 28, 28)}

        # NNSight requires a tokenizer, we are passing in tensors so any tokenizer will do
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        checkpoint_data = torch.load(OUT_PATH) if OUT_PATH.exists() else {}
    
        for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))):
            model.load_state_dict(state_dict)
            model.cuda() # type: ignore

            if epoch not in checkpoint_data:
                checkpoint_data[epoch] = {}

            train_dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

            checkpoint_data[epoch]['parameter_norm'] = torch.cat([parameters.flatten() for parameters in model.parameters()]).flatten().norm(p=2).item()
            checkpoint_data[epoch]['train_loss'] = compute_losses(model, train_dataloader, device)
            checkpoint_data[epoch]['test_loss'] = compute_losses(model, test_dataloader, device)

            nnsight_model = NNsight(model)
            
            # embed_hookpoints = ["vit.embeddings"]
            attn_hookpoints = [f"vit.encoder.layer.{i}.attention.output" for i in range(len(nnsight_model.vit.encoder.layer))]
            mlp_hookpoints = [f"vit.encoder.layer.{i}.output" for i in range(len(nnsight_model.vit.encoder.layer))]
            # resid_hookpoints = [f"vit.encoder.layer.{i}.layernorm_after" for i in range(len(nnsight_model.vit.encoder.layer))]

            sae_path = Path(f"sae/vit/{run_identifier}_{args.sae_name}/{epoch}")
            if not sae_path.exists():
                cfg = TrainConfig(
                    SaeConfig(multi_topk=True), 
                    batch_size=32,
                    run_name=str(sae_path),
                    log_to_wandb=not args.debug,
                    hookpoints=attn_hookpoints + mlp_hookpoints #+ resid_hookpoints, # embed_hookpoints + 
                )
                trainer = SaeTrainer(cfg, sae_train_dataset, model, dummy_inputs)
                trainer.fit()
            
            # embed = nnsight_model.vit.embeddings
            # resids = [block.layernorm_after for block in nnsight_model.vit.encoder.layer]
            attns = [block.attention.output for block in nnsight_model.vit.encoder.layer]
            mlps = [block.output for block in nnsight_model.vit.encoder.layer]
            all_submods = [submod for layer_submods in zip(attns, mlps) for submod in layer_submods] # [embed] + resids

            dictionaries = {}
            # dictionaries[embed] = Sae.load_from_disk(
            #     os.path.join(sae_path, 'vit.embeddings'),
            #     device=device
            # )
            for i in range(len(nnsight_model.vit.encoder.layer)): # type: ignore
                # dictionaries[resids[i]] = Sae.load_from_disk(
                #     os.path.join(sae_path, f'vit.encoder.layer.{i}.layernorm_after'),
                #     device=device
                # )

                dictionaries[attns[i]] = Sae.load_from_disk(
                    os.path.join(sae_path, f'vit.encoder.layer.{i}.attention.output'),
                    device=device
                )

                dictionaries[mlps[i]] = Sae.load_from_disk(
                    os.path.join(sae_path, f'vit.encoder.layer.{i}.output'),
                    device=device
                )

            # Check what everything represents now this is meaned over a large dataset rather than individual observations over a small dataset
            # Add max_samples arg to get_sae_acts then enabled non-meaning

            num_patches = (config.image_size // config.patch_size) ** 2 + 1 # +1 for the class token
            
            mean_nodes, fvu, multi_topk_fvu, mean_l2, var_trace = get_sae_acts(
                nnsight_model, all_submods, dictionaries, 
                train_dataloader, aggregate=True, input_key='input_ids', # mean=False, 
                num_patches=num_patches
            )

            mean_fvu = np.mean([v for v in fvu.values()])
            mean_multi_topk_fvu = np.mean([v for v in multi_topk_fvu.values()])

            checkpoint_data[epoch][f'sae_entropy_nodes'] = {'nodes': mean_nodes}   

            # mean_nodes = {k: v.mean(dim=0) for k, v in nodes.items()}

            checkpoint_data[epoch][f'sae_fvu'] = mean_fvu
            checkpoint_data[epoch][f'sae_multi_topk_fvu'] = mean_multi_topk_fvu

            checkpoint_data[epoch][f'sae_entropy'] = abs_score_entropy(mean_nodes)

            mean_node_scores = all_node_scores(mean_nodes)
            checkpoint_data[epoch][f'hoyer'] = hoyer(mean_node_scores)
            checkpoint_data[epoch][f'hoyer_square'] = hoyer_square(mean_node_scores)
            checkpoint_data[epoch][f'gini'] = gini(mean_node_scores)

            # node_scores = all_node_scores(nodes)
            checkpoint_data[epoch]['mean_l2'] = mean_l2 # mean_l2(torch.tensor(node_scores))
            checkpoint_data[epoch]['var_trace'] = var_trace # var_trace(torch.tensor(node_scores))

            del mean_node_scores, mean_nodes, fvu, multi_topk_fvu, # node_scores, nodes, 

            test_nodes, test_fvu, test_multi_topk_fvu, mean_l2, var_trace = get_sae_acts(
                nnsight_model, all_submods, dictionaries, 
                train_dataloader, aggregate=True, num_patches=num_patches # mean=False, 
            )
            checkpoint_data[epoch]['test_sae_fvu'] = np.mean([v for v in test_fvu.values()])
            checkpoint_data[epoch]['test_sae_multi_topk_fvu'] = np.mean([v for v in test_multi_topk_fvu.values()])
            checkpoint_data[epoch]['test_sae_entropy_nodes'] = {'nodes': test_nodes}   

            mean_test_nodes = {k: v.mean(dim=0) for k, v in test_nodes.items()}
            checkpoint_data[epoch]['test_sae_entropy'] = abs_score_entropy(mean_test_nodes)

            mean_test_node_scores = all_node_scores(mean_test_nodes)
            checkpoint_data[epoch]['test_hoyer'] = hoyer(mean_test_node_scores)
            checkpoint_data[epoch]['test_hoyer_square'] = hoyer_square(mean_test_node_scores)
            checkpoint_data[epoch]['test_gini'] = gini(mean_test_node_scores)

            # test_node_scores = all_node_scores(test_nodes)
            checkpoint_data[epoch]['test_mean_l2'] = mean_l2 # mean_l2(torch.tensor(test_node_scores))
            checkpoint_data[epoch]['test_var_trace'] = var_trace # var_trace(torch.tensor(test_node_scores))


        torch.save(checkpoint_data, OUT_PATH)
        
    checkpoint_data = torch.load(OUT_PATH)

    def add_scores_if_exists(fig, key: str, name: str, normalize: bool = False, use_secondary_y=False):
        if key not in checkpoint_data[checkpoint_epochs[0]].keys():
            print(f"Skipping {name} because it does not exist in the checkpoint")
            return

        scores = [scores[key] for scores in checkpoint_data.values()]
        if normalize:
            # Normalize scores to be between 0 and 1
            max_score = max(scores)
            min_score = min(scores)
            scores = [(score - min_score) / (max_score - min_score) for score in scores]

        fig.add_trace(go.Scatter(x=checkpoint_epochs, y=scores, mode='lines', name=name), secondary_y=use_secondary_y)
            
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)
    
    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', normalize=False)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', normalize=False)
    # add_scores_if_exists(fig, 'parameter_norm', 'Parameter norm')
    add_scores_if_exists(fig, 'sae_entropy', f'H(SAE Nodes)', use_secondary_y=True)
    add_scores_if_exists(fig, 'test_sae_entropy', 'Test H(SAE Nodes)', False, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and SAE Node Entropy over Epochs (MNIST)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'mnist_feature_sharing_entropy.pdf', format='pdf')

    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', False)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', False)
    # add_scores_if_exists(fig, 'parameter_norm', 'Parameter norm', False)

    add_scores_if_exists(fig, 'hoyer', 'Hoyer', False, use_secondary_y=True)
    add_scores_if_exists(fig, 'test_hoyer', 'Test Hoyer', False, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and SAE Hoyer Norm over Epochs (MNIST)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'mnist_feature_sharing_hoyer.pdf', format='pdf')


    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', False)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', False)
    # add_scores_if_exists(fig, 'parameter_norm', 'Parameter norm', False)

    add_scores_if_exists(fig, 'gini', 'Gini coefficient', False, use_secondary_y=True)
    add_scores_if_exists(fig, 'test_gini', 'Test Gini', False, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and Gini Coefficient of Mean SAE Activations over Epochs (mnist)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'mnist_feature_sharing_gini.pdf', format='pdf')


if __name__ == "__main__":
    main()