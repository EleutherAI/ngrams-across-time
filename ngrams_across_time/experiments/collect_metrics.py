from pathlib import Path
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
from ngrams_across_time.utils.tensor_db import TensorDatabase
from ngrams_across_time.utils.data import ZippedDataset
from ngrams_across_time.image.collect_image_metrics import collect_image_losses
from ngrams_across_time.language.collect_language_metrics import collect_language_divergences

def get_metric_function(
    model_name: str,
    db_path: Path,
    dataset: ZippedDataset,
    models: Dict[int, torch.nn.Module] | Tuple[Dict[int, torch.nn.Module], int],
    modality: str,
    batch_size: int = 32,
    target_order: int = 2,
):
    orders = {
        'low_order': target_order - 1,
        'target_order': target_order,
        'high_order': target_order + 1,
        'base': None
    }
    if modality == 'language':
        models, vocab_size = models

    def get_or_compute_metrics(checkpoint: int):
        db = TensorDatabase(str(db_path / "tensor_db"), str(db_path / "tensors"))

        db.clean_invalid_tensors()

        metric_results = {}
        missing_metrics = []
        tags = {
            'model': model_name,
            'step': checkpoint,
        }

        if modality == 'language':
            metrics = ['kl', 'js', 'loss']
        else:
            metrics = ['loss']

        for order_name, order in orders.items():
            for metric in metrics:
                tags.update({
                    'metric': metric,
                    'order': order,
                })
                if modality == 'language':
                    tags['num-shards'] = 2
                    if metric == 'loss':
                        tags['order'] = 'baseline'
                
                db_results = db.query_last(**tags)
                if db_results is None:
                    missing_metrics.append(metric)
                else:
                    metric_results[metric] = db_results['tensor']
                
            if len(missing_metrics) > 0:
                model = models[checkpoint]

                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                if modality == 'language':
                    metric_results.update(collect_language_divergences(model, dataloader, order_name, missing_metrics, vocab_size))
                else:
                    metric_results.update(collect_image_losses(model, dataloader, order_name, missing_metrics))
                for metric, tensor in metric_results.items():
                    if metric in missing_metrics:
                        tags.update({
                            'metric': metric,
                            'order': order,
                            })
                        if metric == 'loss' and modality == 'language':
                            tags['order'] = 'baseline'
                        
                        assert isinstance(tensor, torch.Tensor)
                        db.add_tensor(tensor, tags)
            

        db.close()

        return metric_results
    
    return get_or_compute_metrics

