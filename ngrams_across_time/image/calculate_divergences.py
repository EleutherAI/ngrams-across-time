from pathlib import Path
import argparse
import pdb

import torch
from torch.utils.data import DataLoader
from transformers import ConvNextV2ForImageClassification

from auto_circuit.utils.graph_utils import patchable_model

from ngrams_across_time.image.load_model_data import load_models_and_dataset
from ngrams_across_time.utils.tensor_db import TensorDatabase
from ngrams_across_time.utils.divergences import kl_divergence_log_space, js_divergence_log_space
from ngrams_across_time.utils.utils import assert_type

def calculate_divergences(model, dataloader, device):
    kl_divs = []
    js_divs = []
    
    model.eval()

    ds_index = 0
    with torch.no_grad():
        for norm_batch, edited_batch in dataloader:
            batch_size = norm_batch['pixel_values'].shape[0]

            x_normal, y_normal = norm_batch['pixel_values'].to(device), norm_batch['label'].to(device)
            x_edited, y_edited = edited_batch['pixel_values'].to(device), edited_batch['label'].to(device)
            
            # Original classification
            logits_orig = model(x_normal).logits
            logits_edited = model(x_edited).logits
            
            kl_div = kl_divergence_log_space(logits_orig, logits_edited)
            js_div = js_divergence_log_space(logits_orig, logits_edited)

            kl_divs.append(torch.stack([torch.arange(ds_index, ds_index + batch_size), kl_div.cpu()], dim=1))
            js_divs.append(torch.stack([torch.arange(ds_index, ds_index + batch_size), js_div.cpu()], dim=1))
            ds_index += batch_size
    
    kl_divs = torch.cat(kl_divs, dim=0)
    js_divs = torch.cat(js_divs, dim=0)

    return kl_divs, js_divs

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and dataset
    model_iterator, dataset = load_models_and_dataset(args.model, args.dataset, args.ce_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize TensorDatabase
    db = TensorDatabase(str(args.data_path / "tensor_db.sqlite"), str(args.data_path / "tensors"))
    
    # Iterate through steps
    for step, model in model_iterator:
        print(f"Processing step {step}")

        # Calculate divergences
        kl_divs, js_divs = calculate_divergences(model, dataloader, device)
        
        # Save results in TensorDatabase
        tags = {
            'model': args.model,
            'dataset': args.dataset,
            'step': step,
            'metric': 'kl',
            'dataset': args.dataset,
            'ce_type': args.ce_type
        }
        db.add_tensor(kl_divs, tags)
        
        tags['metric'] = 'js'
        db.add_tensor(js_divs, tags)
    
    db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate divergences for concept editing")
    parser.add_argument("--model", type=str, default='convnext-nano', help="Model name")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset name")
    parser.add_argument("--data_path", type=Path, default=Path("test_image"), help="Path to save TensorDatabase")
    parser.add_argument("--start_step", type=int, default=0, help="Starting step")
    parser.add_argument("--end_step", type=int, default=1024, help="Ending step")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--ce_type", type=str, default="qn", help="Concept editing type")
    
    args = parser.parse_args()
    main(args)