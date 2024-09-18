import os
from typing import Any

import torch.nn.functional as F
import torch
import numpy as np
from torch import Tensor
import einops
from tqdm import tqdm
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers import AutoTokenizer
from nnsight import LanguageModel
from sae import Sae
from datasets import Dataset as HfDataset
from ngrams_across_time.feature_circuits.circuit import get_circuit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random

def set_seeds(seed=16):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seeds()
device = torch.device("cuda")

def permute_train_data_with_different_answers(train_data, train_labels, p):
    new_train_data = []
    new_train_labels = []
    
    for i in range(len(train_data)):
        original_a, original_b, _ = train_data[i]
        original_sum = (original_a + original_b) % p
        
        while True:
            # Randomly select a new pair (a, b)
            new_a = random.randint(0, p-1)
            new_b = random.randint(0, p-1)
            new_sum = (new_a + new_b) % p
            
            # Check if the new sum is different from the original
            if new_sum != original_sum:
                new_train_data.append(torch.tensor([new_a, new_b, p]))
                new_train_labels.append(new_sum)
                break
    
    return torch.stack(new_train_data), torch.tensor(new_train_labels)

def all_edge_scores(checkpoint_scores):
        scores = []
        for node, values in checkpoint_scores.items():
            if isinstance(values, Tensor):
                scores.extend(values.flatten().view(-1).tolist())
            else:
                scores.extend(values.act.flatten().view(-1).tolist())
        return scores

def plot_histograms(combined_scores):
    threshold = 2
    fig = make_subplots(rows=len(combined_scores), cols=4, column_widths=[1, 0.3, 1, 0.3])

    for i, scores in enumerate(combined_scores, 1):
        positive_small = scores[scores > 0]
        positive_small = positive_small[positive_small < threshold]
        positive_large = scores[scores >= threshold]
        negative = scores[scores < 0]
        zero_count = np.sum(scores == 0)

        fig.add_trace(go.Histogram(x=positive_small, name="Positive values", nbinsx=50), 
                        row=i, col=1)

        fig.add_trace(go.Histogram(x=positive_large, name="Positive values", nbinsx=50,),
                        row=i, col=2)

        # Histogram for negative values
        fig.add_trace(
            go.Histogram(x=negative, name="Negative values", nbinsx=50),
            row=i, col=3
        )

        # Bar for zero count
        fig.add_trace(
            go.Bar(x=['Zero values'], y=[zero_count], name="Zero values"),
            row=i, col=4
        )

        # fig.update_xaxes(title="Value", type="log", row=i, col=1)
        fig.update_xaxes(title="Value", row=i, col=2)
        fig.update_yaxes(title="Frequency", row=i, col=3)
        # fig.update_yaxes(title="Frequency", row=i, col=1)
        fig.update_yaxes(title="Count", row=i, col=1)

    # fig.update_layout(
    #     title="Distribution of Values",
    #     showlegend=False,
    #     height=900
    # )

    fig.update_layout(
        height=300 * len(combined_scores),  # Adjust height based on number of plots
        width=1200,
        title_text="Distribution of EAP Scores over Checkpoints",
        showlegend=False
    )
    fig.show()

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
    
    # SAEs like multiple epochs
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

    model = GPTNeoXForCausalLM(GPTNeoXConfig(
        vocab_size=p + 1,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_act="relu",
        max_position_embeddings=4,
        hidden_dropout=0,
        classifier_dropout=0.,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        device=device.type,
        seed=999,
        init_weights=True,
        normalization_type=None,
    ))

    checkpoint_ablation_loss_increases = []
    checkpoint_edge_scores = []
    PATCH = True
    if PATCH:
        for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))[::10]):
            if epoch > 13_000:
                break

            model.load_state_dict(state_dict)
            model.cuda() # type: ignore

            # NNSight requires a tokenizer, we are passing in tensors so any tokenizer will do
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            language_model: LanguageModel = LanguageModel(model, tokenizer=tokenizer) # type: ignore 

            # Load SAE for epoch
            run_name = f"sae/grok.{epoch}"
            embed = language_model.gpt_neox.embed_in
            attns = [layer.attention for layer in language_model.gpt_neox.layers]
            mlps = [layer.mlp for layer in language_model.gpt_neox.layers]
            resids = []
            if len(language_model.gpt_neox.layers) > 1: # type: ignore
                for i, layer in enumerate(language_model.gpt_neox.layers): # type: ignore
                    resids.append(layer)
            resids.append(language_model.gpt_neox.final_layer_norm)
            dictionaries = {}
            dictionaries[embed] = Sae.load_from_disk(
                os.path.join(run_name, 'gpt_neox.embed_in'),
                device=device
            )

            for i in range(len(language_model.gpt_neox.layers)): # type: ignore
                dictionaries[attns[i]] = Sae.load_from_disk(
                    os.path.join(run_name, f'gpt_neox.layers.{i}.attention.dense'),
                    device=device
                )
                dictionaries[mlps[i]] = Sae.load_from_disk(
                    os.path.join(run_name, f'gpt_neox.layers.{i}.mlp.dense_4h_to_h'),
                    device=device
                )
                if i == len(language_model.gpt_neox.layers) - 1: # type: ignore
                    dictionaries[resids[i]] = Sae.load_from_disk(
                        os.path.join(run_name, f'gpt_neox.final_layer_norm'),
                        device=device
                    )
                else:
                    dictionaries[resids[i]] = Sae.load_from_disk(
                        os.path.join(run_name, f'gpt_neox.layers.{i + 1}'),
                        device=device
                    )

            # Run EAP with SAEs
            # Count the loss increase when ablating an arbitrary number of edges (10):
            debug = True
            if debug:
                len_data = 4
            else:
                len_data = 40
            train_data = train_data[:len_data]
            train_data = train_data.cuda()

            permuted_train_data, permuted_train_labels = permute_train_data_with_different_answers(train_data, train_labels, p)
            permuted_train_data = permuted_train_data.cuda()

            train_labels = train_labels[:len_data]

            
            # Verify that the shapes match
            assert permuted_train_data.shape == train_data.shape
            assert permuted_train_labels.shape == train_labels.shape

            # Verify that all labels are different
            assert torch.all(permuted_train_labels != train_labels.cpu())

            def metric_fn(logits, repeat: int = 1):
                labels = train_labels
                if repeat > 1:
                    labels = labels.repeat(repeat)
                return (
                    F.cross_entropy(logits[:, -1, :], labels, reduction="none")
                )

            threshold = 0.01
            nodes, edges = get_circuit(train_data, permuted_train_data, 
                                    language_model, embed, attns, mlps, 
                                    resids, dictionaries, metric_fn, 
                                    aggregation="sum", node_threshold=threshold, edge_threshold=threshold, nodes_only=False)
            checkpoint_edge_scores.append({
                    'epoch': epoch,
                    'nodes': nodes,
                    'edges': dict(edges)
                 })
            torch.save(checkpoint_edge_scores, "workspace/checkpoint_edge_scores.pth")
    
    checkpoint_edge_scores = torch.load("workspace/checkpoint_edge_scores.pth")

    combined_scores: list[list[float]] = [all_edge_scores(scores) for scores in checkpoint_edge_scores]
    plot_histograms(np.array(combined_scores))

    # submodules = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
    
    # unpatched_loss = F.cross_entropy(
    #     model(train_data[:-1]).logits[:, -1, :], 
    #     train_labels[:-1], 
    #     reduction="mean"
    # )

    # # Patch nodes with scores above threshold
    # # first run through the fake inputs to figure out which hidden states are tuples
    # is_tuple = {}
    # with model.scan("_"):
    #     for submodule in submodules:
    #         is_tuple[submodule] = type(submodule.output.shape) == tuple

    # with language_model.trace(train_data[1:]) as tracer:
    #     num_latents = next(iter(dictionaries.values())).num_latents

    #     # Clean 
    #     hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}
    #     hidden_states_patch = {}
    #     with model.trace(patch, **tracer_kwargs), t.no_grad():
    #         for submodule in submodules:
    #             dictionary = dictionaries[submodule]
    #             x = submodule.output
    #             if is_tuple[submodule]:
    #                 x = x[0]
    #             f = nnsight.apply(dictionary.encode, x)
    #             x_hat = nnsight.apply(dictionary.decode, f.top_acts, f.top_indices)
    #             residual = x - x_hat
    #             hidden_states_patch[submodule] = DenseAct(_dense(f, num_latents), residual.save())

    #         metric_patch = metric_fn(model, **metric_kwargs).save()
    #     total_effect = (metric_patch.value - metric_clean.value).detach()
    #     hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}


    
    # hidden_states_clean = {}
    # with model.trace(clean, **tracer_kwargs) as tracer, t.no_grad():    
    #     for submodule in submodules:
    #         dictionary = dictionaries[submodule]
    #         x = submodule.output
    #         if is_tuple[submodule]:
    #             x = x[0]
            
    #         f = nnsight.apply(dictionary.encode, x)
    #         x_hat = nnsight.apply(dictionary.decode, f.top_acts, f.top_indices)
    #         residual = x - x_hat
    #         dense = _dense(f, num_latents)
    #         hidden_states_clean[submodule] = DenseAct(dense, residual.save())
        
    #     metric_clean = metric_fn(model, **metric_kwargs).save()

    # # Get loss with top 10 nodes ablated

    # # Save activations on corrupted prompts (prompts offset by 1)
    # patch = {}
    # with language_model.trace(train_data[1:]) as tracer:
    #     # Run the model on the clean prompts with the dictionaries and get the loss
    #     language_model.output.save()
    #     for name, effect in nodes.items():
    #         # dictionary features to ablate
    #         effect[effect > threshold]
    #         patch[name] = getattr(language_model, name).output.save()
    #         node.restore()

    # patched_loss = F.cross_entropy(patched_logits.to(device).squeeze(), train_labels[:-1], reduction="mean")
    # checkpoint_ablation_loss_increases.append((patched_loss - unpatched_loss).item())


if __name__ == "__main__":
    main()