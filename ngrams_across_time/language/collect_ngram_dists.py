from pathlib import Path
from argparse import ArgumentParser
import os

import numpy as np
from tokengrams import ShardedInMemoryIndex
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="test")
    parser.add_argument('--tokenizer', type=str, default="EleutherAI/pythia-70m")
    parser.add_argument('--num_shards', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=2049)
    parser.add_argument('--n', type=int, nargs="+")
        
    return parser.parse_args()


def float32_to_bfloat16(x):
    """Convert float32 to bfloat16 values represented as uint16."""
    x = np.asarray(x)
    x_view = x.view(np.uint32)
    return (x_view >> 16).astype(np.uint16)


def bfloat16_to_float32(x):
    """Convert bfloat16 values represented as uint16 to float32."""
    x = np.asarray(x)
    return np.frombuffer(
        np.left_shift(x.astype(np.uint32), 16).tobytes(), 
        dtype=np.float32
    ).reshape(x.shape)


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    os.makedirs(data_path, exist_ok=True)

    num_shards = args.num_shards
    num_samples = args.num_samples
    batch_size = args.batch_size
    seq_len = args.seq_len

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # This is correct; tokenizer.vocab_size is incorrect.
    vocab_size = len(tokenizer)

    tokens_path = Path('/mnt/ssd-1/pile-ngrams-tokens')
    sa_path = Path('/mnt/ssd-1/pile-suffix-arrays')
    tokengrams_paths = [
        (str(sa_fp), str(t_fp)) 
        for sa_fp, t_fp in zip(sorted(tokens_path.iterdir()), sorted(sa_path.glob('*.idx')))
    ]

    index = ShardedInMemoryIndex(tokengrams_paths[:num_shards], vocab_size)
    print(f"Loaded index with {num_shards} shards...")

    data: Dataset = load_from_disk(str(Path('data') / "val_tokenized.hf")) # type: ignore
    data = data.select(range(num_samples))
    
    for n in args.n:
        print(f"Generating smoothed {n}-gram logprobs...")

        data_loader = DataLoader(data, batch_size) # type: ignore

        if os.path.exists(data_path / f"smoothed-{n}-gram-pile-dists-bf16-{num_shards}_shards.npy"):
            print(f"Skipping {n}-gram")
            continue
        file_path = data_path / f"smoothed-{n}-gram-pile-dists-bf16-{num_shards}_shards.npy"
        mode = "w+" if not file_path.exists() else "r+"
        # Use uint16 as a bit container for truncated values which will be converted back to float32 later
        mmap = np.memmap(
            file_path,
            mode=mode,
            dtype=np.uint16,
            shape=(num_samples * seq_len, vocab_size),
        )

        chunk_len = batch_size * seq_len
        for i, batch in tqdm(enumerate(data_loader)):
            if i < 13:
                continue
            ngram_prefixes = []
            for row in batch["input_ids"]:
                ngram_prefixes.extend(
                    [row[max(0, i - (n - 1)) : i].tolist() for i in range(len(row))]
                )

            probs = index.batch_get_smoothed_probs(ngram_prefixes)
            logprobs = np.log(np.array(probs, dtype=np.float32))
            logprobs_bf16 = float32_to_bfloat16(logprobs)
            
            # float16 precision is too low for logprobs so we use the first 16 bits of a float32 (~1% precision loss)
            mmap[
                (i * chunk_len) : 
                ((i * chunk_len) + chunk_len)
            ] = logprobs_bf16

if __name__ == "__main__":
    main()