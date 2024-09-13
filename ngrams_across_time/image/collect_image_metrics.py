from pathlib import Path
from typing import List, Literal

import torch

from ngrams_across_time.image.image_data_types import image_hash
from ngrams_across_time.utils.divergences import kl_divergence_log_space, js_divergence_log_space

def collect_image_losses(model, dataloader, order_index, metrics: Literal['loss'] = 'loss'):
    model.eval()
    device = model.device
    losses = {metric: [] for metric in metrics}
    with torch.no_grad():
        for batch in dataloader:
                batch = batch[order_index]
                x_target, y_target = batch['pixel_values'].to(device), batch['label'].to(device)
                logits_target = model(x_target).logits
                losses['loss'].append(torch.nn.functional.cross_entropy(logits_target, y_target, reduction='none'))
            
        return {metric: torch.cat(loss_list, dim=0) for metric, loss_list in losses.items()}
            

            
def calculate_logit_diffs(args, model, dataloader, device, db, step):
    ce_losses = []
    
    logit_diff_clean_correct_qnts = []
    logit_diff_clean_correct_gots = []
    logit_diff_qn_correct_qns = []
    logit_diff_got_correct_gots = []

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

            ce_loss = torch.nn.functional.cross_entropy(logits_orig, y_normal, reduction='none')

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
        'metric': 'logit_diff_clean_correct_qn',
        'dataset': args.dataset,
        'return_type': 'edited'
    }
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

