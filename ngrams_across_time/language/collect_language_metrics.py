from typing import List, Literal
import pandas as pd
import pickle
import numpy as np

from datasets import Dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F

from ngrams_across_time.utils.divergences import kl_divergence_log_space, js_divergence_log_space

def collect_language_divergences(model, dataloader, order_name: str, metrics: List[Literal['kl', 'js', 'loss']], vocab_size: int):
    model.eval()
    device = model.device
    divergences = {metric: [] for metric in metrics}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            tokens_batch = batch['base']['input_ids']
            ngram_batch = batch[order_name]
            tokens = tokens_batch.to(device)
            logits = model(tokens).logits[:, :, :vocab_size]

            if 'kl' in metrics:
                kl_div = kl_divergence_log_space(logits, ngram_batch.to(device), dim=-1)
                divergences['kl'].append(kl_div.detach().cpu())
            
            breakpoint()

            if 'js' in metrics:
                js_div = js_divergence_log_space(logits, ngram_batch.to(device), dim=-1)
                divergences['js'].append(js_div.detach().cpu())

            if 'loss' in metrics:
                batch_size = len(tokens)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), tokens[:, 1:].reshape(-1), reduction='none')
                divergences['loss'].append(loss.detach().cpu().reshape(batch_size, -1))

    return {metric: torch.cat(div_list, dim=0) for metric, div_list in divergences.items()}


def save_ngram_corruptions(df: pd.DataFrame, model: str, start: int, end: int, n: int):
    bigrams_path = "/mnt/ssd-1/lucia/features-across-time/data/pile-deduped/bigrams.pkl"
    with open(bigrams_path, "rb") as f:
        bigram_counts = pickle.load(f).toarray().astype(np.float32)
        unigram_counts = torch.tensor(bigram_counts).sum(dim=1).cuda()
        del bigram_counts

    ngram_data = {
        'target': [],
        'low_order': []
    }
    unique_ngrams = list(set(tuple(x) for x in df['example']))
    num_samples = 100
    for unique_ngram in unique_ngrams:
        end_token = torch.tensor(unique_ngram[-1], device='cuda')
        start_tokens = torch.multinomial(unigram_counts, (n - 1) * num_samples, replacement=True)
        # TODO could select these to be n-grams that aren't learned between checkpoints
        corrupt_ngrams = torch.cat([start_tokens.view(num_samples, n - 1), end_token.repeat(num_samples, 1)], dim=1)

        ngram_data['target'].append(unique_ngram)
        ngram_data['low_order'].append(corrupt_ngrams.cpu().numpy())

    # create huggingface dataset where clean is each unique n-gram and corrupt is the dataset of corrupt ngrams
    dataset_name = f"{n}-grams-{start}-{end}-{model.replace('/', '--')}"
    dataset = Dataset.from_dict(ngram_data)
    dataset.save_to_disk(dataset_name)    