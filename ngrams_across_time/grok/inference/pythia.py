import os
from argparse import ArgumentParser
from pathlib import Path

from plotly.subplots import make_subplots


from scipy import stats
import torch.nn.functional as F
import torch
import torch as t
import numpy as np
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import nnsight
from nnsight.intervention import InterventionProxy
from nnsight import LanguageModel
from sae import SaeConfig, Sae, TrainConfig, SaeTrainer
import plotly.graph_objects as go
import lovely_tensors as lt
from sae.data import MemmapDataset
from torch.utils.data import DataLoader

from ngrams_across_time.grok.metrics import mean_l2, var_trace, gini, hoyer, hoyer_square, abs_score_entropy
from ngrams_across_time.feature_circuits.dense_act import to_dense
from ngrams_across_time.utils.utils import assert_type, set_seeds
from ngrams_across_time.language.hf_client import get_model_checkpoints

lt.monkey_patch()
set_seeds(598)
device = torch.device("cuda")

@torch.no_grad()
def get_sae_acts(
        model,
        ordered_submods: list,
        dictionaries,
        dataloader,
        aggregate: bool = True, # or 'none' for not aggregating across sequence position
        mean: bool = True,
        device: str | t.device = t.device("cuda"),
        input_key: str = 'input_ids',
):
    # Collect metadata and preallocate tensors
    num_latents = next(iter(dictionaries.values())).num_latents
    batch_size = 0
    seq_len = 0
    is_tuple = {}
    accumulated_nodes = {}
    accumulated_fvu = {}
    accumulated_multi_topk_fvu = {}
    with torch.amp.autocast(str(device)):  # Enable automatic mixed precision
        for batch in dataloader:
            with model.scan(batch[input_key].to(device)):
                batch_size = batch[input_key].shape[0]
                seq_len = batch[input_key].shape[1]
                for submodule in ordered_submods:
                    is_tuple[submodule] = isinstance(submodule.output.shape, tuple) 
            break

    for submodule in ordered_submods:
        shape = (batch_size, num_latents) if aggregate else (batch_size, seq_len, num_latents)
        accumulated_nodes[submodule.path] = torch.zeros(shape)
        accumulated_fvu[submodule.path] = 0
        accumulated_multi_topk_fvu[submodule.path] = 0

    # Do inference
    total_samples = 0
    for batch in dataloader:
        batch_nodes = {}
        batch_fvus = {}
        batch_multi_topk_fvus = {}
        
        with model.trace(batch[input_key].to(device)):
            for submodule in ordered_submods:
                input = submodule.output if not is_tuple[submodule] else submodule.output[0]

                dictionary = dictionaries[submodule]
                flat_f: InterventionProxy = nnsight.apply(dictionary.forward, input.flatten(0, 1))
                batch_fvus[submodule.path] = flat_f.fvu.cpu().save()
                batch_multi_topk_fvus[submodule.path] = flat_f.multi_topk_fvu.cpu().save()
                latent_acts = flat_f.latent_acts.view(input.shape[0], input.shape[1], -1)
                latent_indices = flat_f.latent_indices.view(input.shape[0], input.shape[1], -1)
                dense = to_dense(latent_acts, latent_indices, num_latents)
                batch_nodes[submodule.path] = dense.cpu().save()
        
        if aggregate:
            batch_nodes = {k: v.sum(dim=1) for k, v in batch_nodes.items()}
        
        for submodule in ordered_submods:
            accumulated_nodes[submodule.path] += batch_nodes[submodule.path] * batch_size
            accumulated_fvu[submodule.path] += batch_fvus[submodule.path].item() * batch_size
            accumulated_multi_topk_fvu[submodule.path] += batch_multi_topk_fvus[submodule.path].item() * batch_size

        total_samples += batch_size

    nodes = {k: v / total_samples for k, v in accumulated_nodes.items()}
    fvu = {k: v / total_samples for k, v in accumulated_fvu.items()}
    multi_topk_fvu = {k: v / total_samples for k, v in accumulated_multi_topk_fvu.items()}

    if mean:
        nodes = {k: v.mean(dim=0) for k, v in nodes.items()}
    
    return nodes, fvu, multi_topk_fvu


def compute_losses(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch['input_ids'].to(device)
            
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs, targets).logits

            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def all_node_scores(checkpoint_scores) -> np.ndarray:
        scores = []
        for node, values in checkpoint_scores.items():
            if node == 'y':
                continue
            if isinstance(values, Tensor):
                scores.extend(values.flatten().tolist())
            else:
                scores.extend(values.act.flatten().tolist())
        return np.array(scores)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--run_name", type=str, default='')
    # parser.add_argument("--sae_name", type=str, default='')
    return parser.parse_args()


def main():
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)
    model_name = "pythia-160m"
    batch_size = 16  # Adjust based on your GPU memory

    args = get_args()
    run_identifier = f"{model_name}"

    OUT_PATH = Path(f"workspace/inference/{run_identifier}.pth")
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

    checkpoints = get_model_checkpoints(f"EleutherAI/{model_name}")    
    log2_keys = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16_000, 33_000, 66_000, 131_000, 143_000]
    log_checkpoints = {key: checkpoints[key] for key in log2_keys}
    test_early_index = len(log2_keys)
    # log_checkpoints = log_checkpoints[:test_early_index]
    
    if args.inference:
        checkpoint_data = torch.load(OUT_PATH) if OUT_PATH.exists() else {}

        data = MemmapDataset("/mnt/ssd-1/pile_preshuffled/deduped/document.bin", ctx_len=2049)
        train_data = data.select(rng=range(100_000))
        test_data = data.select(rng=range(100_000, 100_000 + 2048))
        # Subset of data to collect eval metrics
        len_sample_data = 4 if args.debug else 512

        # NNSight requires a tokenizer. We are passing in tensors so any tokenizer will do
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        for step_number, checkpoint in tqdm(log_checkpoints.items()):
            if step_number not in checkpoint_data:
                checkpoint_data[step_number] = {}
            elif 'gini' in checkpoint_data[step_number]:
                print(f"Skipping step {step_number} because it already exists")
                continue

            # Load model and SAEs
            model = AutoModelForCausalLM.from_pretrained(
                f"EleutherAI/{model_name}",
                torch_dtype=torch.float16,
                revision=checkpoint,
                cache_dir=".cache",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True) if "12b" in model_name else None
            )
            if not "12b" in model_name:
                model = model.cuda()

            nnsight_model = LanguageModel(model, tokenizer=tokenizer)


            # sae_path = Path(f"/mnt/ssd-1/lucia/sae/{run_identifier}/grok.{checkpoint}")
            # if not sae_path.exists():
            #     hookpoints = ["gpt_neox.embed_in"]
            #     for layer in range(len(model.gpt_neox.layers)):
            #         hookpoints.extend([f"gpt_neox.layers.{layer}.attention", f"gpt_neox.layers.{layer}.mlp"])

            #     cfg = TrainConfig(
            #         SaeConfig(multi_topk=True, expansion_factor=32, k=model.config.hidden_size // 2), 
            #         batch_size=batch_size,
            #         run_name=str(sae_path),
            #         log_to_wandb=True,
            #         hookpoints=hookpoints,
            #     )
            #     trainer = SaeTrainer(cfg, train_data, model)
            #     trainer.fit()
            
            # attns = [layer.attention for layer in nnsight_model.gpt_neox.layers]
            # mlps = [layer.mlp for layer in nnsight_model.gpt_neox.layers]
            dictionaries = {}
            layer_indices = list(range(len(nnsight_model.gpt_neox.layers)))[1::2]
            resids = [layer for layer in list(nnsight_model.gpt_neox.layers)[1::2]]

            for i, resid in zip(layer_indices, resids):
                sae_path = Path(f"/mnt/ssd-1/lucia/sae/{run_identifier} step {step_number} L{i}")
                # dictionaries[attns[i]] = Sae.load_from_disk(
                #     os.path.join(sae_path, f'layers.{i}.attention'),
                #     device=device
                # )
                # dictionaries[mlps[i]] = Sae.load_from_disk(
                #     os.path.join(sae_path, f'layers.{i}.mlp'),
                #     device=device
                # )                
                dictionaries[resid] = Sae.load_from_disk(
                    os.path.join(sae_path, f'layers.{i}'),
                    device=device
                )
            all_submods = resids

            # Collect metrics
            checkpoint_data[step_number]['parameter_norm'] = torch.cat([parameters.flatten() for parameters in model.parameters()]).flatten().norm(p=2).item()

            train_dataloader = DataLoader(train_data.select(range(len_sample_data)), batch_size=batch_size, drop_last=True) 
            test_dataloader = DataLoader(test_data.select(range(len_sample_data)), batch_size=batch_size, drop_last=True) 
            checkpoint_data[step_number]['train_loss'] = compute_losses(model, train_dataloader, device)
            checkpoint_data[step_number]['test_loss'] = compute_losses(model, test_dataloader, device)

            nnsight_dataloader = DataLoader(data.select(range(len_sample_data)), batch_size=batch_size, drop_last=True)

            nodes, fvu, multi_topk_fvu = get_sae_acts(
                nnsight_model, all_submods, dictionaries, 
                nnsight_dataloader, aggregate=True, mean=False
            )

            mean_fvu = np.mean([v for v in fvu.values()])
            mean_multi_topk_fvu = np.mean([v for v in multi_topk_fvu.values()])

            checkpoint_data[step_number][f'sae_fvu'] = mean_fvu
            checkpoint_data[step_number][f'sae_multi_topk_fvu'] = mean_multi_topk_fvu
            checkpoint_data[step_number][f'sae_entropy_nodes'] = {'nodes': nodes}   

            mean_nodes = {k: v.mean(dim=0) for k, v in nodes.items()}
            mean_node_scores = all_node_scores(mean_nodes)

            checkpoint_data[step_number][f'sae_entropy'] = abs_score_entropy(mean_node_scores)
            checkpoint_data[step_number][f'hoyer'] = hoyer(mean_node_scores)
            checkpoint_data[step_number][f'hoyer_square'] = hoyer_square(mean_node_scores)
            checkpoint_data[step_number][f'gini'] = gini(mean_node_scores)

            node_scores = all_node_scores(nodes)
            checkpoint_data[step_number]['mean_l2'] = mean_l2(torch.tensor(node_scores))
            checkpoint_data[step_number]['var_trace'] = var_trace(torch.tensor(node_scores))

            torch.save(checkpoint_data, OUT_PATH)

            # Print all metrics
            print(f"Step {step_number}:")
            print(f"Parameter norm: {checkpoint_data[step_number]['parameter_norm']}")
            print(f"Train loss: {checkpoint_data[step_number]['train_loss']}")
            print(f"Test loss: {checkpoint_data[step_number]['test_loss']}")
            print(f"SAE FVU: {checkpoint_data[step_number]['sae_fvu']}")
            print(f"SAE Multi Top-K FVU: {checkpoint_data[step_number]['sae_multi_topk_fvu']}")
            print(f"SAE Entropy: {checkpoint_data[step_number]['sae_entropy']}")
            print(f"Hoyer: {checkpoint_data[step_number]['hoyer']}")
            print(f"Hoyer Square: {checkpoint_data[step_number]['hoyer_square']}")
            print(f"Gini coefficient: {checkpoint_data[step_number]['gini']}") 
            print(f"Mean L2: {checkpoint_data[step_number]['mean_l2']}")
            print(f"Var Trace: {checkpoint_data[step_number]['var_trace']}")

            # TODO add var entropy

        torch.save(checkpoint_data, OUT_PATH)
        
    # Plot data
    checkpoint_data = torch.load(OUT_PATH)

    def add_scores_if_exists(
            fig, key: str, name: str, normalize: bool = False,
            use_secondary_y=False
        ):
        if key not in checkpoint_data[list(log_checkpoints.keys())[0]].keys():
            print(f"Skipping {name} because it does not exist in the checkpoint")
            return
        
        scores = [scores[key] for scores in list(checkpoint_data.values())[:test_early_index]]
        # Normalize scores to be between 0 and 1
        if normalize:
            max_score = max(scores)
            min_score = min(scores)
            scores = [(score - min_score) / (max_score - min_score) for score in scores]
        
        fig.add_trace(
            go.Scatter(x=list(log_checkpoints.keys()), y=scores, mode='lines', name=name), 
            secondary_y=use_secondary_y
        )
            
    # fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])

    # # Underlying model and task stats
    # add_scores_if_exists(fig, 'train_loss', None, 'Train loss')
    # add_scores_if_exists(fig, 'test_loss', None, 'Test loss')
    # add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm')
    # # Sparsity metrics
    # add_scores_if_exists(fig, 'sae_entropy', None, 'SAE Entropy')
    # add_scores_if_exists(fig, 'hoyer', None, 'Hoyer')
    # add_scores_if_exists(fig, 'hoyer_square', None, 'Hoyer Square')
    # add_scores_if_exists(fig, 'gini', None, 'Gini coefficient')
    # # SAE fraction of variance unexplained metrics
    # add_scores_if_exists(fig, 'sae_fvu', None, 'SAE FVU')
    # add_scores_if_exists(fig, 'sae_multi_topk_fvu', None, 'SAE Multi Top-K FVU')


    images_path = Path("images")
    images_path.mkdir(exist_ok=True)
    
    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', checkpoint_data)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', checkpoint_data)
    add_scores_if_exists(fig, 'parameter_norm', 'Parameter norm', checkpoint_data)
    add_scores_if_exists(fig, 'sae_entropy', f'H(SAE Nodes)', checkpoint_data, use_secondary_y=True
    )
    fig.update_layout(
        title=f"Loss and SAE Node Entropy over Epochs (Pythia-160M)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'pythia_feature_sharing_entropy.pdf', format='pdf')

    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', checkpoint_data)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', checkpoint_data)
    # add_scores_if_exists(fig, 'parameter_norm', 'Parameter norm', checkpoint_data)

    add_scores_if_exists(fig, 'hoyer', 'Hoyer', checkpoint_data, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and SAE Hoyer Norm over Epochs (Pythia-160M)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'pythia_feature_sharing_hoyer.pdf', format='pdf')


    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', checkpoint_data)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', checkpoint_data)
    # add_scores_if_exists(fig, 'parameter_norm', 'Parameter norm', checkpoint_data)

    add_scores_if_exists(fig, 'gini', 'Gini coefficient', checkpoint_data, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and Gini Coefficient of Mean SAE Activations over Epochs (Pythia-160M)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'pythia_feature_sharing_gini.pdf', format='pdf')

    
if __name__ == "__main__":
    main()