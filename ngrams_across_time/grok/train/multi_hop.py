import math
import argparse
import random
import glob
import os
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset
from datasets import concatenate_datasets, load_dataset, DatasetDict, Dataset as HfDataset
from sae import Sae, SaeTrainer, TrainConfig, SaeConfig
from nnsight import NNsight

from ngrams_across_time.utils.utils import assert_type, set_seeds
from ngrams_across_time.feature_circuits.circuit import get_circuit


set_seeds(42)



def create_name_based_splits(
        dataset: HfDataset, split_sizes: dict[str, int]
    ) -> DatasetDict:
    """
    Creates dataset splits with specified number of rows per name.
    Assumes dataset is sorted by name with exactly 1000 items per name.
    
    Args:
        dataset: Input dataset (must be sorted by name, 1000 items per name)
        split_sizes: Dictionary mapping split names to rows per name
                    e.g., {'train': 800, 'val': 100, 'test': 100}
    
    Returns:
        DatasetDict containing splits with specified rows per name
    """
    num_rows_per_name = 1000
    num_names = len(dataset) // num_rows_per_name
    
    total_size = sum(split_sizes.values())
    if total_size > num_rows_per_name:
        raise ValueError(f"Total split sizes ({total_size}) exceed number of rows per name ({num_rows_per_name})")

    splits = {}
    offset = 0
    for split_name, size in split_sizes.items():
        group_starts = np.arange(num_names) * num_rows_per_name + offset
        
        indices = group_starts[:, None] + np.arange(size)
        indices = indices.flatten()
        
        splits[split_name] = dataset.select(indices)
        offset += size
    
    return DatasetDict(splits)

def count_first_name(dataset):
    name_counter = Counter()
    
    batch_size = 10_000
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        
        name_counter.update(batch['name'])
        if len(name_counter) > 1:
            break
    
    print('number of occurrences of first names assuming dataset sorted by name', name_counter.most_common(10))
    
    return name_counter

def calculate_model_size(num_params):
    return num_params * 4 / (1024 * 1024)  # Size in MB

def calculate_architecture(num_params):
    n_layers = int(math.log(num_params / 1e6, 2)) + 4
    hidden_size = int(math.sqrt(num_params / (n_layers * 4)))
    hidden_size = (hidden_size // 64) * 64  # Round to nearest multiple of 64
    return n_layers, hidden_size

def create_model_and_tokenizer(num_params):
    n_layers, hidden_size = calculate_architecture(num_params)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=n_layers,
        num_attention_heads=hidden_size // 64,
        max_position_embeddings=2048,
    )
    
    model = LlamaForCausalLM(config)
    
    return model, tokenizer


class OnTheFlyTokenizationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoded = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in encoded.items()}


def load_and_prepare_datasets(tokenizer, N=250000):
    bios_dataset = load_dataset(f"EleutherAI/transformer-reasoning-bios-dataset-{N}")
    bios_dataset = assert_type(DatasetDict, bios_dataset)
    
    # count_first_name(bios_dataset['train']) # 1_000 bios per person for N=250_000 in sorted order
    bios_train = bios_dataset['train'].select_columns(['bio']).rename_column('bio', 'text')

    # Create datasets with a log-spaced number of rewrites of each bio
    chunks = []
    for rewrites in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        splits = create_name_based_splits(
            bios_train,
            split_sizes={'train': rewrites, 'val': 100, 'test': 100},
        )
        train_dataset = OnTheFlyTokenizationDataset(splits['train'], tokenizer)
        val_dataset = OnTheFlyTokenizationDataset(splits['val'], tokenizer)
        heldout_dataset = OnTheFlyTokenizationDataset(splits['test'], tokenizer)
        chunks.append({
            'train': train_dataset,
            'val': val_dataset,
            'test': heldout_dataset,
        })


    return chunks


def find_question_end(text, tokenizer):
    # Find the last occurrence of "? "
    question_end = text.rfind(": ")
    if question_end == -1:
        return None
    
    # Tokenize up to the question mark, including special tokens
    question_tokens = tokenizer.encode(text[:question_end+1], add_special_tokens=True)
    return len(question_tokens)


def main(args):
    model_size_mb = calculate_model_size(args.num_params)
    print(f"Estimated model size: {model_size_mb:.2f} MB")

    # Load and prepare datasets
    if args.resume_from:
        print(f"Loading model from checkpoint: {args.resume_from}")
        model = LlamaForCausalLM.from_pretrained(args.resume_from)
        tokenizer = AutoTokenizer.from_pretrained(args.resume_from)
    else:
        model, tokenizer = create_model_and_tokenizer(args.num_params)

    chunks = load_and_prepare_datasets( tokenizer, args.N)

    if args.resume_from:
        base_checkpoint_dir = f"./results/n{args.N}_p{args.num_params}"
        latest_checkpoint = max(glob.glob(os.path.join(base_checkpoint_dir, "checkpoint-*")), key=os.path.getctime)

    def preprocess_logits_for_metrics(logits, labels):
        batch_size, seq_length, vocab_size = logits.shape
        selected_logits = []
        
        for i in range(batch_size):
            # Decode the full sequence
            text = tokenizer.decode(labels[i], skip_special_tokens=True)
            question_end = find_question_end(text, tokenizer)
            
            if question_end is not None:
                selected_logits.append(logits[i, question_end:, :])
            else:
                selected_logits.append(logits[i, -1:, :])  # Fallback if no question end found
        
        return torch.cat(selected_logits, dim=0)

    for i, data in enumerate(chunks):
        train_dataset, val_dataset, heldout_dataset = data['train'], data['val'], data['test']
        epochs = 50_000_000/len(train_dataset)
        print(f"Epochs: {epochs}")
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/n{args.N}_p{args.num_params}",
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_accumulation_steps=2,
            warmup_steps=500,
            weight_decay=0.1,
            logging_dir=f"./logs/n{args.N}_p{args.num_params}",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=10_000,
            save_steps=10_000,
            load_best_model_at_end=True,
            dataloader_num_workers=8,
            fp16=True,
            tf32=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=heldout_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if args.resume_from:
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            trainer.train()

        # Evaluate on validation set
        val_results = trainer.evaluate(val_dataset)
        print("Validation Results:", val_results)

        # Evaluate on heldout profiles
        heldout_results = trainer.evaluate()
        print("Heldout Profiles Results:", heldout_results)

        # Save the model
        model.save_pretrained(f"./final_model_n{args.N}_p{args.num_params}")
        tokenizer.save_pretrained(f"./final_model_n{args.N}_p{args.num_params}")

        # Train SAEs on the final model
        run_name = Path(f"sae/llama/final_model_n{args.N}_p{args.num_params}_c{i}")
        if run_name.exists():
            continue
        
        hookpoints = ["model.embed_tokens"]
        for i in range(len(model.layers)):
            hookpoints.append(f"model.layers.{i}.self_attn")
            hookpoints.append(f"model.layers.{i}.mlp")
        
        cfg = TrainConfig(
            SaeConfig(multi_topk=True),
            batch_size=64,
            run_name=str(run_name),
            log_to_wandb=True,
            hookpoints=hookpoints,
        )
        sae_trainer = SaeTrainer(cfg, train_dataset, model)
        sae_trainer.fit()

        # # TODO fix path Load dictionaries from disk
        # dictionaries = {
        #     hookpoint: torch.load(f"{run_name}/{hookpoint}/sae.dictionaries.pth")
        #     for hookpoint in hookpoints
        # }

        # # Calculate circuit entropy over training set using get_circuit
        # # 1. Get torch tensor of input
        # def metric_fn(inputs, outputs, labels):
        #     pass
        
        # nnsight_model = NNsight(model)
        # ig_circuit = get_circuit(
        #     train_dataset, None, nnsight_model, hookpoints, dictionaries,
        #     metric_fn=metric_fn
        # )


        # # Get SAE activations



        # # Save to pth


if __name__ == "__main__":
    # Expect training loss to remain stable or even go up as the network runs out of storage space)

    # if over small sets syntactical and semantic features are combined into one syntax-semantic pair feature, it will be
    # more unique to a given task and less generalizable. If the model is trained over a larger set, it will be more likely to
    # separate the two types of features, making the model more generalizable.

    # What would happen if we also trained on the QA dataset then calculated circuit entropy at the answer position?
    # We'd use the dispatch feature for one hop questions, and hopefully both circuit entropy and loss would drop as 
    # rewrites increase

    # Where do we measure circuit entropy without the QAs? I guess at each position where a fact is placed, over just averaging over everything
    # although that will increase circuit noise

    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters")
    parser.add_argument("--num_params", type=int, default=1_000_000, help="Number of parameters for the model")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of examples to use for training (for testing purposes)")
    parser.add_argument("--N", type=int, default=25_000, help="Number of profiles to use for dataset", choices=[10_000, 25_000, 250_000])
    parser.add_argument("--resume_from", action="store_true", help="Resume training from most recent checkpoint")
    args = parser.parse_args()
    
    main(args)
