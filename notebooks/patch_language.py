# %%
import random

from torch import Tensor
import torch
import numpy as np
from datasets import load_from_disk
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.data import PromptDataset, PromptDataLoader
from auto_circuit.types import AblationType
from auto_circuit.utils.graph_utils import patchable_model

from src.language.hf_client import load_with_retries, get_pythia_model_size

device = "cuda"
model_name = "EleutherAI/pythia-160m"
model_size = get_pythia_model_size(model_name)
revision_1 = "step128"
revision_2 = "step256"

def get_patchable_model(model_name, revision, device="cuda"):
    model_size = get_pythia_model_size(model_name)
    model = load_with_retries(model_name, revision, model_size)
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    return patchable_model(model, factorized=True, device=device, separate_qkv=True)

model_1 = get_patchable_model(model_name, revision_1)
model_2 = get_patchable_model(model_name, revision_2)

ngrams_dataset_name = "/mnt/ssd-1/lucia/ngrams-across-time/data/val_tokenized.hf"
dataset = load_from_disk(ngrams_dataset_name)

top_tokens = pd.read_csv("/mnt/ssd-1/david/ngrams_across_time/data/top_kl_divs.csv")

weird_data = dataset[top_tokens.sample_idx]

answers = weird_data['input_ids'][range(len(weird_data)), top_tokens.token_idx].unsqueeze(1)
wrong_answers = []
for i, idx in enumerate(top_tokens.token_idx):
    logits = model_1(weird_data['input_ids'][i, :idx + 1])[:, -1]
    topk = torch.topk(logits, k=5, dim=-1).indices.squeeze()
    wrong_answers.append(topk)


pdset = PromptDataset(
    clean_prompts=weird_data['input_ids'],
    corrupt_prompts=weird_data['input_ids'],
    answers=answers,
    wrong_answers=wrong_answers
)

ploader = PromptDataLoader(pdset, None, 0, batch_size=1, shuffle=True)
ablations = batch_src_ablations(model_1, ploader, AblationType.RESAMPLE, 'clean')

print(ablations[-323570240853508540][0].sum())

# Function to generate synthetic sequences
def generate_synthetic_sequence(dataset, weird_token, seq_length=2048):
    random_seq = dataset[random.randint(0, len(dataset)-1)]['input_ids'][:seq_length].clone()
    replace_idx = random.randint(0, seq_length-2)  # -2 to ensure there's a token after
    random_seq[replace_idx] = weird_token
    return random_seq, replace_idx

# Generate synthetic sequences
weird_token = weird_data['input_ids'][0, top_tokens.token_idx[0] - 1]
num_synthetic = 1000  # Number of synthetic sequences to generate
synthetic_seqs = [generate_synthetic_sequence(dataset, weird_token) for _ in range(num_synthetic)]

# Function to compute KL divergence
def compute_kl_div(p, q):
    return F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')

# Compute logits for weird data
weird_logits = model_2(weird_data['input_ids'][0, :top_tokens.token_idx[0]+1].unsqueeze(0))[:, -1, :]

# Compute KL divergences for synthetic sequences
synthetic_kl_divs = []
for seq, idx in synthetic_seqs:
    logits = model_2(seq.unsqueeze(0))[:, idx + 1, :]
    kl_div = compute_kl_div(logits, weird_logits)
    synthetic_kl_divs.append(kl_div.item())

# Compute KL divergences for random samples
random_kl_divs = []
for _, idx in synthetic_seqs:
    random_seq = dataset[random.randint(0, len(dataset)-1)]['input_ids'][:2048]
    logits = model_2(random_seq.unsqueeze(0))[:, idx + 1, :]
    kl_div = compute_kl_div(logits, weird_logits)
    random_kl_divs.append(kl_div.item())

# Compare distributions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(synthetic_kl_divs, bins=50, alpha=0.5, label='Synthetic')
plt.hist(random_kl_divs, bins=50, alpha=0.5, label='Random')
plt.xlabel('KL Divergence')
plt.ylabel('Frequency')
plt.title('Distribution of KL Divergences')
plt.legend()
plt.show()

# Print summary statistics
print(f"Synthetic KL Div: Mean = {np.mean(synthetic_kl_divs):.4f}, Std = {np.std(synthetic_kl_divs):.4f}")
print(f"Random KL Div: Mean = {np.mean(random_kl_divs):.4f}, Std = {np.std(random_kl_divs):.4f}")

# Generate synthetic sequences
weird_token = weird_data['input_ids'][0, top_tokens.token_idx[0] - 1]
num_synthetic = 1000  # Number of synthetic sequences to generate
synthetic_seqs = [generate_synthetic_sequence(dataset, weird_token) for _ in range(num_synthetic)]

# Compute logits for weird data
weird_logits = model_1(weird_data['input_ids'][0, :top_tokens.token_idx[0]+1].unsqueeze(0))[:, -1, :]

# Compute KL divergences for synthetic sequences
synthetic_kl_divs = []
for seq, idx in synthetic_seqs:
    logits = model_1(seq.unsqueeze(0))[:, idx + 1, :]
    kl_div = compute_kl_div(logits, weird_logits)
    synthetic_kl_divs.append(kl_div.item())

# Compute KL divergences for random samples
random_kl_divs = []
for _, idx in synthetic_seqs:
    random_seq = dataset[random.randint(0, len(dataset)-1)]['input_ids'][:2048]
    logits = model_1(random_seq.unsqueeze(0))[:, idx + 1, :]
    kl_div = compute_kl_div(logits, weird_logits)
    random_kl_divs.append(kl_div.item())

# Compare distributions
plt.figure(figsize=(10, 6))
plt.hist(synthetic_kl_divs, bins=50, alpha=0.5, label='Synthetic')
plt.hist(random_kl_divs, bins=50, alpha=0.5, label='Random')
plt.xlabel('KL Divergence')
plt.ylabel('Frequency')
plt.title('Distribution of KL Divergences')
plt.legend()
plt.show()

# Print summary statistics
print(f"Synthetic KL Div: Mean = {np.mean(synthetic_kl_divs):.4f}, Std = {np.std(synthetic_kl_divs):.4f}")
print(f"Random KL Div: Mean = {np.mean(random_kl_divs):.4f}, Std = {np.std(random_kl_divs):.4f}")



