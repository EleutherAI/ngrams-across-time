from pathlib import Path
import argparse
import pdb
import hashlib

import torch
from torch.utils.data import DataLoader
from transformers import ConvNextV2ForImageClassification

from auto_circuit.utils.graph_utils import patchable_model

from ngrams_across_time.image.load_model_data import load_models_and_dataset
from ngrams_across_time.utils.tensor_db import TensorDatabase
from ngrams_across_time.utils.divergences import kl_divergence_log_space, js_divergence_log_space
from ngrams_across_time.utils.utils import assert_type

def image_hash(image_tensor, num_bits=32):
    hash_hex = hashlib.md5(image_tensor.numpy().tobytes()).hexdigest()
    return int(hash_hex[:num_bits//6], 16)

def calculate_divergences(model, dataloader, device, db, step):
    kl_divs_qn = []
    js_divs_qn = []
    ce_losses = []
    kl_divs_got = []
    js_divs_got = []
    
    logit_diff_clean_correct_qnts = []
    logit_diff_clean_correct_gots = []
    logit_diff_qn_correct_qns = []
    logit_diff_got_correct_gots = []

    norm_labels = []
    qn_labels = []
    got_labels = []

    model.eval()

    ds_index = 0
    image_hashes = []
    
    with torch.no_grad():
        for norm_batch, qn_batch, got_batch in dataloader:
            batch_size = norm_batch['pixel_values'].shape[0]

            x_normal, y_normal = norm_batch['pixel_values'].to(device), norm_batch['label'].to(device)
            x_qn, y_qn = qn_batch['pixel_values'].to(device), qn_batch['label'].to(device)
            x_got, y_got = got_batch['pixel_values'].to(device), got_batch['label'].to(device)
            
            logits_orig = model(x_normal).logits
            logits_qn = model(x_qn).logits
            logits_got = model(x_got).logits
            
            kl_div_qn = kl_divergence_log_space(logits_orig, logits_qn)
            js_div_qn = js_divergence_log_space(logits_orig, logits_qn)

            kl_div_got = kl_divergence_log_space(logits_orig, logits_got)
            js_div_got = js_divergence_log_space(logits_orig, logits_got)

            ce_loss = torch.nn.functional.cross_entropy(logits_orig, y_normal, reduction='none')

            kl_divs_qn.append(torch.stack([torch.arange(ds_index, ds_index + batch_size), kl_div_qn.cpu()], dim=1))
            js_divs_qn.append(torch.stack([torch.arange(ds_index, ds_index + batch_size), js_div_qn.cpu()], dim=1))
            kl_divs_got.append(torch.stack([torch.arange(ds_index, ds_index + batch_size), kl_div_got.cpu()], dim=1))
            js_divs_got.append(torch.stack([torch.arange(ds_index, ds_index + batch_size), js_div_got.cpu()], dim=1))

            logit_diff_clean_correct_qnt = logits_orig[torch.arange(batch_size), y_normal] - logits_orig[torch.arange(batch_size), y_qn]
            logit_diff_clean_correct_got = logits_orig[torch.arange(batch_size), y_normal] - logits_orig[torch.arange(batch_size), y_got]
            logit_diff_qn_correct_qn = logits_qn[torch.arange(batch_size), y_normal] - logits_qn[torch.arange(batch_size), y_qn]
            logit_diff_got_correct_got = logits_got[torch.arange(batch_size), y_normal] - logits_got[torch.arange(batch_size), y_got]

            logit_diff_clean_correct_qnts.append(torch.stack([torch.arange(ds_index, ds_index + batch_size), logit_diff_clean_correct_qnt.cpu()], dim=1))
            logit_diff_clean_correct_gots.append(torch.stack([torch.arange(ds_index, ds_index + batch_size), logit_diff_clean_correct_got.cpu()], dim=1))
            logit_diff_qn_correct_qns.append(torch.stack([torch.arange(ds_index, ds_index + batch_size), logit_diff_qn_correct_qn.cpu()], dim=1))
            logit_diff_got_correct_gots.append(torch.stack([torch.arange(ds_index, ds_index + batch_size), logit_diff_got_correct_got.cpu()], dim=1))

            ce_losses.append(torch.stack([torch.arange(ds_index, ds_index + batch_size), ce_loss.cpu()], dim=1))
            ds_index += batch_size
            
            batch_hashes = [image_hash(img) for img in norm_batch['pixel_values']]
            image_hashes.extend(batch_hashes)
        
    kl_divs_qn = torch.cat(kl_divs_qn, dim=0)
    js_divs_qn = torch.cat(js_divs_qn, dim=0)
    kl_divs_got = torch.cat(kl_divs_got, dim=0)
    js_divs_got = torch.cat(js_divs_got, dim=0)
    ce_losses = torch.cat(ce_losses, dim=0)
    logit_diff_clean_correct_qnts = torch.cat(logit_diff_clean_correct_qnts, dim=0)
    logit_diff_clean_correct_gots = torch.cat(logit_diff_clean_correct_gots, dim=0)
    logit_diff_qn_correct_qnts = torch.cat(logit_diff_qn_correct_qns, dim=0)
    logit_diff_got_correct_gots = torch.cat(logit_diff_got_correct_gots, dim=0)

    # Save image_hashes along with other metrics
    tags = {
        'model': args.model,
        'dataset': args.dataset,
        'step': step,
        'metric': 'kl_qn',
        'dataset': args.dataset,
        'return_type': 'edited'
    }
    db.add_tensor(kl_divs_qn, tags)
    
    tags['metric'] = 'js_qn'
    db.add_tensor(js_divs_qn, tags)

    tags['metric'] = 'kl_got'
    db.add_tensor(kl_divs_got, tags)

    tags['metric'] = 'js_got'
    db.add_tensor(js_divs_got, tags)

    tags['metric'] = 'logit_diff_clean_correct_qn'
    db.add_tensor(logit_diff_clean_correct_qnts, tags)

    tags['metric'] = 'logit_diff_clean_correct_got'
    db.add_tensor(logit_diff_clean_correct_gots, tags)

    tags['metric'] = 'logit_diff_qn_correct_qn'
    db.add_tensor(logit_diff_qn_correct_qnts, tags)

    tags['metric'] = 'logit_diff_got_correct_got'
    db.add_tensor(logit_diff_got_correct_gots, tags)

    tags['metric'] = 'ce'
    db.add_tensor(ce_losses, tags)
    
    tags['metric'] = 'image_hashes'
    db.add_tensor(torch.tensor(image_hashes, dtype=torch.int32), tags)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and dataset
    model_iterator, dataset = load_models_and_dataset(args.model, args.dataset, "edited")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize TensorDatabase
    db = TensorDatabase(str(args.data_path / "tensor_db.sqlite"), str(args.data_path / "tensors"))
    
    # Iterate through steps
    for step, model in model_iterator:
        print(f"Processing step {step}")

        # Calculate divergences
        calculate_divergences(model, dataloader, device, db, step)
    db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate divergences for concept editing")
    parser.add_argument("--model", type=str, default='convnext-nano', help="Model name")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset name")
    parser.add_argument("--data_path", type=Path, default=Path("test_image"), help="Path to save TensorDatabase")
    parser.add_argument("--start_step", type=int, default=0, help="Starting step")
    parser.add_argument("--end_step", type=int, default=1024, help="Ending step")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    
    args = parser.parse_args()
    main(args)