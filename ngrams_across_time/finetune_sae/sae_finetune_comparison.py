# Train finetuned and pretrained SAEs over Pythia-410m for comparison
from argparse import ArgumentParser
from pathlib import Path

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel
import torch
from ngrams_across_time.utils.utils import assert_type
from ngrams_across_time.language.hf_client import get_model_checkpoints, with_retries, get_pythia_model_size
from safetensors.torch import load_model

from sae.config import SaeConfig, TrainConfig
from sae.trainer import SaeTrainer
from sae.data import MemmapDataset
import lovely_tensors as lt

lt.monkey_patch()
device = torch.device("cuda")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m") 
    parser.add_argument("--tag", type=str, default='')
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=16)
    return parser.parse_args()


def get_batch_size(model_name: str, model_bytes: int, ctx_n: int, data_bytes: int, vram_bytes: int) -> int:
    model_size_bytes = get_pythia_model_size(model_name) * model_bytes

    # 2x for gradients
    data_size_bytes = ctx_n * data_bytes * 2
    
    return (vram_bytes - model_size_bytes) // data_size_bytes


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    if not args.steps:
        args.steps = sorted(list(get_model_checkpoints(args.model_name).keys()))
    else:
        args.steps = sorted(args.steps)

    for i, step in enumerate(args.steps):
        # Load model checkpoint from HuggingFace
        model_size = get_pythia_model_size(args.model_name)
        model = with_retries(
            lambda : AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype="auto",
            revision=f'step{step}',
            cache_dir=".cache",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True) if model_size > 6e9 else None
        ))
        model = assert_type(PreTrainedModel, model)
        
        if model_size <= 6e9 :
            model.cuda() # type: ignore
        
        # Load pile training data
        dataset = MemmapDataset("/mnt/ssd-1/pile_preshuffled/deduped/document.bin", ctx_len=2049)

        # Train SAE from scratch
        run_name = Path(f"sae/{args.model_name.replace('/', '--')}/{step}{('-' + args.tag) if args.tag else ''}")
        if run_name.exists():
            continue

        num_layers = model.config.num_hidden_layers
        hookpoints = [
            f"gpt_neox.layers.{i}.attention" for i in range(num_layers)
        ] + [
            f"gpt_neox.layers.{i}.mlp" for i in range(num_layers)
        ] + [
            f"gpt_neox.layers.{i}" for i in range(num_layers)
        ]

        cfg = TrainConfig(
            SaeConfig(multi_topk=True),
            batch_size=1,
            run_name=str(run_name),
            log_to_wandb=True,
            hookpoints=hookpoints,
        )
        trainer = SaeTrainer(cfg, dataset, model)
        trainer.fit()
            
        # Can't finetune at the first checkpoint
        if i == 0:
            continue

        # Finetune starting from weights in previous checkpoint
        cfg = TrainConfig(
            SaeConfig(multi_topk=True),
            batch_size=1,
            run_name=str(run_name) + ".finetune",
            log_to_wandb=True,
            hookpoints=hookpoints,
            lr = 1e-5 # Lower learning rate
        )
        trainer = SaeTrainer(cfg, dataset, model)

        prev_run_name = Path(
            f"sae/{args.model_name.replace('/', '--')}/{args.steps[i - 1]}{('-' + args.tag) if args.tag else ''}"
        )
        for name, sae in trainer.saes.items():
            load_model(sae, f"{prev_run_name}/{name}/sae.safetensors", device=str(device))
    
        trainer.fit()


if __name__ == "__main__":
    main()