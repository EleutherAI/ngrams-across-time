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

    args = parse_args()

    data_path = Path(args.data_path)
    os.makedirs(data_path, exist_ok=True)

    max_shards_in_memory = 4

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
    print('vocab size and alternate', vocab_size, tokenizer.vocab_size)

    tokens_path = Path('/mnt/ssd-1/pile-ngrams-tokens')
    sa_path = Path('/mnt/ssd-1/pile-suffix-arrays')
    tokengrams_paths = [
        (str(sa_fp), str(t_fp)) 
        for sa_fp, t_fp in zip(sorted(tokens_path.iterdir()), sorted(sa_path.glob('*.idx')))
    ]

    total_tokens = sum(get_shard_size(file_path=path[0]) for path in tokengrams_paths[:num_shards])
    print(f"Total tokens across all shards: {total_tokens}")

    data: Dataset = load_from_disk(str(Path('data') / "val_tokenized.hf")) # type: ignore
    data = data.select(range(num_samples))

    # open a timestamped log file in output
    # from datetime import datetime
    # os.makedirs(data_path / 'log', exist_ok=True)
    # with open(data_path / 'log' / f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}-log-{num_shards}_shards.txt', 'w') as f:
    #     f.write(f"Total tokens across all shards: {total_tokens})

    for i in range(0, num_shards, max_shards_in_memory):
        # if i == 0:
        #     continue
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

            # TODO clean up this garbage so overwrite/continuation logic isn't happening in a LOOP
            file_path = data_path / f"{n}-gram-pile-dists-bf16-{num_shards}_shards.npy"
            if args.overwrite and i == 0:
                file_path.unlink(missing_ok=True)
            mode = "w+" if not file_path.exists() else "r+"
            if mode == "r+" and not args.cont and i == 0:
                print(f"Exiting as a file exists and args.cont not set")
                import sys; sys.exit(1)
            # Use uint16 as a bit container for truncated values which will be converted back to float32 later
            mmap = np.memmap(
                file_path,
                mode=mode,
                dtype=np.uint16,
                shape=(num_samples * seq_len, vocab_size),
            )

            chunk_len = batch_size * seq_len
            for i, batch in tqdm(enumerate(data_loader)):   
                # Fixing the n == 2 run which is only missing the first shard group
                # assert n == 2 and i == 0
                # Get the missing probability weight from the first examples
                # first_example_probs_sum = bfloat16_to_float32(mmap[1]).sum()
                # import code; code.interact(local=locals())
                # assert abs(1 - first_example_probs_sum - shard_weight) < 0.1
                # print(first_example_probs_sum, shard_weight)
                # Proceed with writing the final probabilities

                ngram_prefixes = []
                for row in batch["input_ids"]:
                    ngram_prefixes.extend(
                        [row[max(0, i - (n - 1)) : i].tolist() for i in range(len(row))]
                    )

                counts = np.array(index.batch_count_next(ngram_prefixes), dtype=np.float32) # num rows, vocab size

                row_sums = counts.sum(axis=1, keepdims=True) # num rows, 1
                row_sums[row_sums == 0] = 1 # avoid nans
                probs = counts / row_sums 
                # probs[row_sums.squeeze() == 0] = uniform_probability_row[None, :] # 1D array
                assert (probs >= 0).all()
                assert (probs <= 1).all()
                
                mmap_slice = slice(i * chunk_len, (i * chunk_len) + chunk_len)

                # float16 precision is too low for logprobs so we use the first 16 bits of a float32 (~1% precision loss)
                # Use uint16 as a bit container for truncated values which will be converted back to float32 later
                accumulator_float32 = bfloat16_to_float32(mmap[mmap_slice]) if mmap[mmap_slice].sum() > 0 else 0
                mmap[mmap_slice] = float32_to_bfloat16(accumulator_float32 + (probs * shard_weight))
                assert np.isfinite(mmap[mmap_slice]).all()
                
                if i == 0:
                    print("accumulated probs sum", bfloat16_to_float32(mmap[0]).sum())
        
        del index

    for n in args.n:
        print("Converting probs to logprobs")

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

            # Add a small epsilon to avoid log(0) 
            epsilon = 1e-10
            log_probs = np.log(np.maximum(data_float32, epsilon))
            mmap_out[i:end] = float32_to_bfloat16(log_probs)

    print("Done!")

def test():
    n = 2
    num_shards = 2
    data_path = Path("test")
    out_file_path = data_path / f"{n}-gram-pile-dists-bf16-{num_shards}_shards-logprobs.npy"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    # This is correct; tokenizer.vocab_size is incorrect.
    vocab_size = len(tokenizer)

    memmap = np.memmap(
        out_file_path,
        mode="r",
        dtype=np.uint16,
        shape=(1024, 2049, vocab_size),
    )

    bfloat16_to_float32(memmap[0])

    breakpoint()



if __name__ == "__main__":
    main()
    # test()