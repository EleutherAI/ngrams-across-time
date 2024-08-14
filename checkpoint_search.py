from argparse import ArgumentParser
import os
import time

from src.language.get_revisions import get_model_checkpoints
from language.pythia import get_basic_pythia_model_names, get_model_size, load_with_retries
from src.utils.utils import assert_type
from src.utils.divergences import kl_divergence_log_space, js_divergence_log_space
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict, Dataset
                    

def get_divergences(revisions: list, model_name: str, data: dict[str, DataLoader], vocab_size: int, key='input_ids'):
    sample_data = next(iter(list(data.values())[0]))[key]
    print(sample_data)

    output_data = torch.zeros(
        len(revisions) - 1, # num checkpoints
        len(sample_data), # num samples
        len(sample_data[0]), # num tokens
    )
    results = {
        ds_name: {
            'js': output_data.clone(),
            'kl': output_data.clone(),
        }
        for ds_name in data.keys()
    }

    for i, revision in enumerate(revisions):
        model_size = get_model_size(model_name)
        model = load_with_retries(model_name, revision, model_size)
        if not model:
            continue

        # Run data through the models and record KL and JS divergences 
        # I want to build a dataset like HF with sequences just given as lists
        for dataset_name, dataloader in data.items():
            for j, batch in enumerate(dataloader):
                input_ids = batch[key].cuda()
                
                outputs = model(input_ids[:, :-1])
                logits = outputs.logits[:, :, :vocab_size]

                results[dataset_name]['kl'][i, j * batch : (j * batch) + batch] = kl_divergence_log_space(
                    logits, batch['labels'].cuda(), dim=-1
                )
                results[dataset_name]['js'][i, j * batch : (j * batch) + batch] = js_divergence_log_space(
                    logits, batch['labels'].cuda(), dim=-1
                )
    return results

def parse_args():
    parser = ArgumentParser(description='Search for interesting data points between contiguous checkpoints')
    
    group = parser.add_argument_group("Model arguments")
    group.add_argument('--model', type=str, nargs='+', help='Model names')
    group.add_argument('--tokenizer', type=str, help='Tokenizer name')
    group.add_argument('--start', type=int, required=True, help='Start checkpoint')
    group.add_argument('--end', type=int, required=True, help='End checkpoint')
    group.add_argument('--batch_size', type=int, default=16)
    
    group = parser.add_argument_group("Data arguments")
    group.add_argument('--data_path', type=str, nargs='+', help='HF datasets on disk')
    group.add_argument('--data', type=str, nargs='+', help='HF datasets on the hub')
    group.add_argument('--out', type=str, default="output", help='Location to save data in CSV format')
    
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load n-gram model distributions
    data = {}
    for data_path in args.data_path:
        ds = load_from_disk(data_path)
        ds.set_format("torch", columns=["input_ids"])
        if isinstance(ds, DatasetDict):
            ds = ds[args.split]
        ds = assert_type(Dataset, ds)
        data[data_path.split('/')[-1]] = DataLoader(ds, batch_size=args.batch_size, shuffle=False) # type: ignore

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Inference over checkpoints
    model_names = args.model or get_basic_pythia_model_names()
    for model_name in model_names:
        checkpoints = get_model_checkpoints(model_name)
        if not checkpoints:
            raise ValueError('No checkpoints found')
        ranged_checkpoints = [v for k, v in checkpoints.items() if args.start <= k <= args.end]
        
        results = get_divergences(ranged_checkpoints, model_name, data, vocab_size=tokenizer.vocab_size)
        torch.save(results, os.path.join(args.out, f"{model_name}.pt"))

if __name__ == '__main__':
    main()
