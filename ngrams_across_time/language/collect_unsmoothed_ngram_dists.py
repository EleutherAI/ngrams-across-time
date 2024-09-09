from pathlib import Path
from argparse import ArgumentParser
import os
import time

import numpy as np
from tokengrams import ShardedInMemoryIndex
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--tokenizer', type=str, default="EleutherAI/pythia-70m")
    parser.add_argument('--max_shards_in_memory', type=int, default=4)
    parser.add_argument('--num_shards', type=int, default=21)
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=2049)
    parser.add_argument('--cont', action="store_true")
    parser.add_argument('--overwrite', action="store_true")
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


def get_shard_size(file_path):
    return os.path.getsize(file_path) // 2  # 16-bit tokens, so divide by 2


def main():
    debug = False

    tokens_path = Path('/mnt/ssd-1/pile-ngrams-tokens')
    sa_path = Path('/mnt/ssd-1/pile-suffix-arrays')
    tokengrams_paths = [
        (str(sa_fp), str(t_fp)) 
        for sa_fp, t_fp in zip(sorted(tokens_path.iterdir()), sorted(sa_path.glob('*.idx')))
    ]

    args = parse_args()

    data_path = Path(args.data_path)
    os.makedirs(data_path, exist_ok=True)

    max_shards_in_memory = args.max_shards_in_memory
    num_shards = args.num_shards
    num_samples = args.num_samples
    batch_size = args.batch_size
    seq_len = args.seq_len
    if debug:
        max_shards_in_memory = 1
        num_shards = 2
        num_samples = 4

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # This is correct; tokenizer.vocab_size is smaller.
    vocab_size = len(tokenizer)

    total_tokens = sum(get_shard_size(file_path=path[0]) for path in tokengrams_paths[:num_shards])
    print(f"Total tokens across all shards: {total_tokens}")

    # Load data
    data: Dataset = load_from_disk(str(Path('data') / "val_tokenized.hf")) # type: ignore
    data = data.select(range(num_samples))

    # Check if the output files already exist and remove if necessary
    for n in args.n:
        file_path = data_path / f"{n}-gram-pile-dists-bf16-{num_shards}_shards.npy"
        out_file_path = data_path / f"{n}-gram-pile-dists-bf16-{num_shards}_shards-logprobs.npy"
        
        if args.overwrite:
            file_path.unlink(missing_ok=True)
            out_file_path.unlink(missing_ok=True)

        if file_path.exists() and not args.cont:
            print(f"Exiting as a file exists and cont flag not set")
            import sys; sys.exit(1)

    # Accumulate probabilities
    for i in range(0, num_shards, max_shards_in_memory):
        shard_group = tokengrams_paths[:num_shards][i:i + max_shards_in_memory]
        shard_group_size = sum(get_shard_size(path[0]) for path in shard_group)
        shard_weight = shard_group_size / total_tokens

        start = time.time()
        index = ShardedInMemoryIndex(shard_group, vocab_size)
        print(f"Loaded index with up to {min(num_shards, max_shards_in_memory)} shards in {time.time() - start}s...")
        print(f"Shard weight is {shard_weight}, shard group size is {shard_group_size}")

        for n in args.n:
            print(f"Generating unsmoothed {n}-gram probs for shard group {i}...")

            data_loader = DataLoader(data, batch_size) # type: ignore

            # Use uint16 as a bit container for custom bfloat16 representation.
            file_path = data_path / f"{n}-gram-pile-dists-bf16-{num_shards}_shards.npy"
            mode = "w+" if not file_path.exists() else "r+"
            mmap = np.memmap(
                file_path,
                mode=mode,
                dtype=np.uint16,
                shape=(num_samples * seq_len, vocab_size),
            )

            for i, batch in tqdm(enumerate(data_loader)):   
                ngram_prefixes = []
                for row in batch["input_ids"]:
                    ngram_prefixes.extend(
                        [row[max(0, i - (n - 1)) : i].tolist() for i in range(len(row))]
                    )

                counts = np.array(index.batch_count_next(ngram_prefixes), dtype=np.float32)

                row_sums = counts.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1 # Avoid nans
                probs = counts / row_sums 
                assert (probs >= 0).all()
                assert (probs <= 1).all()
                
                mmap_slice = slice(
                    i * batch_size * seq_len, 
                    (i * batch_size * seq_len) + batch_size * seq_len
                )

                # float16 precision loss is high so values are stored as a custom bfloat16 not natively supported by numpy (~1% precision loss)
                # TODO use ml_dtypes package
                accumulator_float32 = bfloat16_to_float32(mmap[mmap_slice]) if mmap[mmap_slice].sum() > 0 else 0
                mmap[mmap_slice] = float32_to_bfloat16(accumulator_float32 + (probs * shard_weight))
                
                if debug:
                    assert np.isfinite(mmap[mmap_slice]).all()
                    if i == 0:
                        print("Accumulated probability sum for first entry: ", bfloat16_to_float32(mmap[0]).sum())
        
        del index

    for n in args.n:
        print("Generating log probabilities from probabilities...")

        file_path = data_path / f"{n}-gram-pile-dists-bf16-{num_shards}_shards.npy"
        mmap = np.memmap(
            file_path,
            mode="r+",
            dtype=np.uint16,
            shape=(num_samples * seq_len, vocab_size),
        )

        out_file_path = data_path / f"{n}-gram-pile-dists-bf16-{num_shards}_shards-logprobs.npy"
        mmap_out = np.memmap(
            out_file_path,
            mode="w+",
            dtype=np.uint16,
            shape=(num_samples * seq_len, vocab_size),
        )

        chunk_size = 1000
        for i in range(0, num_samples * seq_len, chunk_size):
            end = min(i + chunk_size, num_samples * seq_len)
            data_float32 = bfloat16_to_float32(mmap[i : end])

            # Additive smoothing to avoid log(0)
            epsilon = 1e-10
            log_probs = np.log(np.maximum(data_float32, epsilon))
            mmap_out[i:end] = float32_to_bfloat16(log_probs)

    print("Done!")


if __name__ == "__main__":
    main()