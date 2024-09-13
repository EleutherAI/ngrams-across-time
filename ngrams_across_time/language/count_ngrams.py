from pathlib import Path
from argparse import ArgumentParser
import os
import time

import numpy as np
from tokengrams import ShardedInMemoryIndex
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from ml_dtypes import bfloat16 as np_bfloat16

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--num_shards', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--seq_len', type=int, default=2049)
    parser.add_argument('--cont', action="store_true")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--n', type=int, nargs="+", required=True)
        
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        args.num_shards = 1

    num_shards, num_samples, seq_len, cont, overwrite = (
        args.num_shards, args.num_samples, args.seq_len, args.cont, args.overwrite
    )

    data_path = Path(args.data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizers, suffix arrays, and validation data for the pile
    tokens_path = Path('/mnt/ssd-1/pile-ngrams-tokens')
    sa_path = Path('/mnt/ssd-1/pile-suffix-arrays')
    tokengrams_paths = [
        (str(sa_fp), str(t_fp)) 
        for sa_fp, t_fp in zip(sorted(tokens_path.iterdir()), sorted(sa_path.glob('*.idx')))
    ]

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    # This is correct; tokenizer.vocab_size is smaller.
    vocab_size = len(tokenizer)
    
    n_dict = {
        n: (
            data_path / f"{n}-gram-autoregressive-counts-{num_shards}-shards{'-debug' if args.debug else ''}.npy",
            data_path / f"{n}-gram-autoregressive-samples-{num_shards}-shards{'-debug' if args.debug else ''}.npy"
        )
        for n in args.n
    }

    # Fail early is existing data could be unintentionally overwritten
    for n, (count_file, sample_file) in n_dict.items():
        if overwrite or args.debug:
            count_file.unlink(missing_ok=True)
            sample_file.unlink(missing_ok=True)

        if count_file.exists() and not cont:
            print(f"Exiting as a counts file exists and cont flag not set: {count_file}")
            exit(1)

        if sample_file.exists() and not cont:
            print(f"Exiting as a logprobs file exists and cont flag not set: {sample_file}")
            exit(1)
    
    # Print total number of tokens across all shards
    print(f"Total tokens across all shards: \
          {sum([os.path.getsize(path[0]) // 2 for path in tokengrams_paths[:num_shards]])}")

    # Load index
    start = time.monotonic()
    index = ShardedInMemoryIndex(tokengrams_paths[:num_shards], vocab_size)
    print(f"Loaded index with {len(tokengrams_paths[:num_shards])} shards in {time.monotonic() - start}s...")
    
    # Count and sample n-grams
    unigram_counts = torch.tensor(index.count_next([]))
    for n, (count_file, sample_file) in n_dict.items():
        count_mmap = np.memmap(
            count_file,
            mode="w+" if not count_file.exists() else "r+",
            dtype=np.int64,
            shape=(num_samples, seq_len, vocab_size),
        )
        sample_mmap = np.memmap(
            filename=sample_file,
            mode="w+" if not sample_file.exists() else "r+",
            dtype=np.int32,
            shape=(num_samples, seq_len),
        )

        count_mmap[:, 0] = unigram_counts.numpy()
        unigram_samples = torch.multinomial(unigram_counts.float(), num_samples)
        
        sample_mmap[:, 0] = unigram_samples.numpy()

        # We want to be able to select the final n-1 tokens of each list with a single slice.
        # So a list of lists where each inner list is a batch of samples for a single column.
        prev = [[sample] for sample in unigram_samples.tolist()]
        for column in tqdm(range(1, seq_len)):
            n_prev = [sample[-(n - 1):] for sample in prev]
            print(len(n_prev), len(n_prev[0]))
            
            counts = np.array(index.batch_count_next(n_prev), dtype=np.int64)
            count_mmap[:, column] = counts
            
            samples = torch.multinomial(torch.tensor(counts, dtype=torch.float64), 1)
            sample_mmap[:, column: column + 1] = samples.numpy()
            
            prev = [prev[i] + [samples[i].item()] for i in range(num_samples)]

            print(sample_mmap[0])
            print(count_mmap[0])
    
    del index


def log_probs(n, num_shards, num_samples, seq_len, vocab_size, data_path):
    print(f"Generating log probabilities from counts for n={n}...")

    file_path = data_path / f"{n}-gram-pile-counts-{num_shards}_shards.npy"
    mmap = np.memmap(
        file_path,
        mode="r+",
        dtype=np.int64,
        shape=(num_samples * seq_len, vocab_size),
    )

    out_file_path = data_path / f"{n}-gram-pile-logprobs-bf16-{num_shards}_shards.npy"
    mmap_out = np.memmap(
        out_file_path,
        mode="w+",
        dtype=np_bfloat16,
        shape=(num_samples * seq_len, vocab_size),
    )

    chunk_size = 1000
    for i in range(0, num_samples * seq_len, chunk_size):
        end = min(i + chunk_size, num_samples * seq_len)
        counts = mmap[i : end]
        probs = counts / counts.sum(axis=1, keepdims=True)
        probs[probs == np.nan] = 1e-10 # Uniform prior on unseen n-grams

        log_probs = np.log(probs)
        mmap_out[i:end] = log_probs.astype(np_bfloat16)


if __name__ == "__main__":
    main()