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


def train(model, tokenizer, wandb_name, args, transcode=False):
    hookpoints = get_gptneo_hookpoints(model)
    hookpoints = [hookpoint for hookpoint in hookpoints if not "attn" in hookpoint]

    dataset = load_dataset("roneneldan/TinyStories", split="train")
    # max_length from TinyStories paper https://arxiv.org/pdf/2305.07759
    dataset = chunk_and_tokenize(dataset, tokenizer, max_seq_len=512)
    dataset.set_format(type="torch", columns=["input_ids"])

    assert model.transformer.h[0].mlp.c_fc.out_features == 256 * 4

    cfg = TrainConfig(
        SaeConfig(multi_topk=True, expansion_factor=8, k=32),
        batch_size=8,
        run_name=str(Path('sae') / wandb_name),
        log_to_wandb=not args.debug,
        hookpoints=hookpoints,
        grad_acc_steps=2,
        micro_acc_steps=2,
        transcode=transcode
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
        "data/tinystories/mlp=1024-dense-8m-max-e=200-esp=15-s=42/checkpoints/last.ckpt", 
        dense=True, 
        tokenizer=tokenizer,
        map_location="cuda"
    )

    model = pl_model.model
    
    wandb_name = f"Dense TinyStories8M mlp=1024 s=42 epoch 21{' ' + args.tag if args.tag else ''}"
    # train(model, tokenizer, wandb_name, args)

    wandb_name = f"Dense TinyStories8M Transcoder mlp=1024 s=42 epoch 21{' ' + args.tag if args.tag else ''}"
    train(model, tokenizer, wandb_name, args, transcode=True)
    

if __name__ == "__main__":
    main()
