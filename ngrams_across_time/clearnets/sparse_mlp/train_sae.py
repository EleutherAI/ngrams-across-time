from pathlib import Path
import torch
from sae.config import SaeConfig, TrainConfig
from sae.sae import Sae
from sae.trainer import SaeTrainer
from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from sae.data import chunk_and_tokenize

from ngrams_across_time.clearnets.autointerp_cache import get_gptneo_hookpoints
from ngrams_across_time.clearnets.sparse_mlp.sparse_gptneox_tinystories import TinyStoriesModel


def sae(model, tokenizer, wandb_name, args):
    hookpoints = get_gptneo_hookpoints(model)
    hookpoints = [hookpoint for hookpoint in hookpoints if not "attn" in hookpoint]

    dataset = load_dataset("roneneldan/TinyStories", split="train")
    # max_length from TinyStories paper https://arxiv.org/pdf/2305.07759
    dataset = chunk_and_tokenize(dataset, tokenizer, max_seq_len=512)
    dataset.set_format(type="torch", columns=["input_ids"])

    print("Training on 16x expansion factor to accommodate hidden size of 256")
    assert model.config.hidden_size == 256, "Hidden size is not 256, reduce expansion factor to 8x"

    cfg = TrainConfig(
        SaeConfig(multi_topk=True, expansion_factor=16, k=32),
        batch_size=8,
        run_name=str(Path('sae') / wandb_name),
        log_to_wandb=not args.debug,
        hookpoints=hookpoints,
        grad_acc_steps=2,
        micro_acc_steps=2,
    )
    trainer = SaeTrainer(
        cfg, dataset, model.cuda(),
    )
    trainer.fit()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("data/tinystories/restricted_tokenizer_v2")
    pl_model = TinyStoriesModel.load_from_checkpoint(
        "data/tinystories/dense-8m-max-e=200-esp=15-s=42/checkpoints/last.ckpt", 
        dense=True, 
        tokenizer=tokenizer
    )

    model = pl_model.model
    
    wandb_name = f"Dense TinyStories8M s=42 epoch 38 {args.tag}"
    sae(model, tokenizer, wandb_name, args)
    

if __name__ == "__main__":
    main()
