import argparse
import glob
import os
from pathlib import Path

import numpy as np
from datasets import Dataset
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset

from ngrams_across_time.transformer_reasoning.src.transformer_reasoning.generate_dataset.generate_qa_dataset import generate_question
from ngrams_across_time.transformer_reasoning.src.transformer_reasoning.train.train_utils import calculate_model_size, InfiniteBiosDataset
from sae import Sae, SaeTrainer, TrainConfig, SaeConfig
from nnsight import NNsight

from ngrams_across_time.utils.utils import assert_type, set_seeds
from ngrams_across_time.feature_circuits.circuit import get_mean_sae_entropy
from ngrams_across_time.grok.conc_grok import gini, all_node_scores, hoyer, hoyer_square, abs_score_entropy


def load_and_prepare_datasets(tokenizer, subset_size=None, N=250000, qa_ratio=0.1, orders=None):
    profiles = load_dataset(f"EleutherAI/profiles_dataset_{N}")['train'] # type: ignore
    
    shuffled_indices = torch.randperm(len(profiles)).tolist()
    heldout_indices = shuffled_indices[:1000]
    retained_indices = shuffled_indices[1000:]

    heldout_profiles = profiles.select(heldout_indices)
    eval_questions = []
    
    for profile in heldout_profiles:
        qa_result = generate_question(profile, profiles, max(orders) if orders else 3, {}, {})
        if qa_result:
            question, _ = qa_result
            eval_questions.append(question)
    
    eval_dataset = Dataset.from_list(eval_questions)
    
    # Tokenize the evaluation dataset
    def tokenize_qa(example):
        text = f"<|endoftext|>Question: {example['question']} Answer: {example['answer']}"
        return tokenizer(text, padding=True, truncation=True, max_length=512)
    
    eval_dataset = eval_dataset.map(
        tokenize_qa,
        remove_columns=eval_dataset.column_names
    )

    # Create infinite training dataset with consistent retained indices
    train_dataset = InfiniteBiosDataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=orders or [1,2],
        qa_prob=qa_ratio,
        qa_indices=retained_indices
    )

    return train_dataset, eval_dataset


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
    base_dir = args.ckpts
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    # extract the list of checkpoint numbers
    checkpoints = [checkpoint.split("-")[-1] for checkpoint in checkpoints]
    latest_checkpoint = checkpoints[-1]
    latest_checkpoint_name = Path(base_dir) / f"checkpoint-{latest_checkpoint}"
    
    print(f"Loading model from checkpoint: {latest_checkpoint_name}")
    model = LlamaForCausalLM.from_pretrained(latest_checkpoint_name)
    tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint_name)
    train_dataset, heldout_dataset = load_and_prepare_datasets(
        tokenizer, args.subset_size, args.N, args.qa_ratio, args.orders
    )

    # Running inference on log-spaced checkpoints
    print(f"Found {len(checkpoints)} checkpoints")
    checkpoint_data = {}
    for checkpoint in checkpoints:
        checkpoint_name = Path(base_dir) / f"checkpoint-{checkpoint}"
        model = LlamaForCausalLM.from_pretrained(checkpoint_name)

        # Train SAEs if necessary
        sae_path = Path(f"sae/llama/{checkpoint_name.name}")
        if not sae_path.exists():
            hookpoints = ["model.embed_tokens"]
            for i in range(len(model.model.layers)):
                hookpoints.append(f"model.layers.{i}.self_attn")
                hookpoints.append(f"model.layers.{i}.mlp")
            
            cfg = TrainConfig(
                SaeConfig(multi_topk=True),
                batch_size=64,
                run_name=str(sae_path),
                log_to_wandb=True,
                hookpoints=hookpoints,
            )

            # TODO convert iterable dataset to dataset if this breaks
            model.cuda()
            sae_trainer = SaeTrainer(cfg, train_dataset, model)
            sae_trainer.fit()

        nnsight_model = NNsight(model)

        # Load saes from disk
        attns = [layer.self_attn for layer in nnsight_model.model]
        mlps = [layer.mlp for layer in nnsight_model.model]

        dictionaries = {}
        for i in range(len(nnsight_model.model.layers)): # type: ignore
            dictionaries[attns[i]] = Sae.load_from_disk(
                os.path.join(sae_path, f'model.layers.{i}.self_attn'),
                device=torch.device("cuda")
            )
            dictionaries[mlps[i]] = Sae.load_from_disk(
                os.path.join(sae_path, f'model.layers.{i}.mlp'),
                device=torch.device("cuda")
            )
        all_submods = [submod for layer_submods in zip(attns, mlps) for submod in layer_submods]

        nodes, fvu, multi_topk_fvu = get_mean_sae_entropy(
            nnsight_model, all_submods, dictionaries, 
            train_dataset, aggregate=True, batch_size=64
        )
        mean_fvu = np.mean([v.item() for v in fvu.values()])
        mean_multi_topk_fvu = np.mean([v.item() for v in multi_topk_fvu.values()])

        checkpoint_data[checkpoint][f'sae_fvu'] = mean_fvu
        checkpoint_data[checkpoint][f'sae_multi_topk_fvu'] = mean_multi_topk_fvu
        checkpoint_data[checkpoint][f'sae_entropy_nodes'] = {'nodes': nodes}   

        checkpoint_data[checkpoint][f'sae_entropy'] = abs_score_entropy(nodes)

        node_scores = all_node_scores(nodes)
        checkpoint_data[checkpoint][f'hoyer'] = hoyer(node_scores)
        checkpoint_data[checkpoint][f'hoyer_square'] = hoyer_square(node_scores)
        checkpoint_data[checkpoint][f'gini'] = gini(node_scores)


    workspace_path = Path(f"workspace/n{args.N}_p{args.num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_infinite")
    workspace_path.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_data, workspace_path / "checkpoint_data.pt")        

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

    # This is the number of times we step through each profiles; the "dataset size" is infinite
    epochs = 4_500 * 25000 / len(train_dataset)
    print(f"Epochs: {epochs}")
    output_dir = f"./results/n{args.N}_p{args.num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_infinite"

    # Load model checkpoint from disk

    model_path_template = f"./final_model/n{args.N}_p{args.num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}"
    tokenizer_path_template = f"./final_model/n{args.N}_p{args.num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters")
    parser.add_argument("--num_params", type=int, default=1_000_000, help="Number of parameters for the model")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of examples to use for training (for testing purposes)")
    parser.add_argument("--N", type=int, default=25000, help="Number of profiles to use for QA dataset")
    parser.add_argument("--orders", type=int, nargs="+", default=None, help="Orders to use for QA dataset")
    parser.add_argument("--ckpts", type=str, default='/mnt/ssd-1/david/transformer-reasoning/results/n25000_p2000000_o2_wd0.1_infinite', help="Path to model checkpoints")
    parser.add_argument("--qa_ratio", type=float, default=0.5,
                       help="Ratio of QA examples to bios examples")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--push_to_hub", action="store_true", help="Push trained model to hf hub")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()
    
    main(args)
