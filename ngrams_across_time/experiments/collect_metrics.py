from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import DataLoader
from ngrams_across_time.utils.tensor_db import TensorDatabase
from ngrams_across_time.utils.data import MultiOrderDataset
from ngrams_across_time.image.collect_image_metrics import collect_image_losses
from ngrams_across_time.language.collect_language_metrics import collect_language_divergences

metric_fn = {
    'language': collect_language_divergences,
    'image': collect_image_losses,
}

def get_metric_function(
    model_name: str,
    db_path: Path,
    dataset: MultiOrderDataset,
    models: Dict[int, torch.nn.Module],
    modality: str,
    batch_size: int = 32,
    target_order: int = 2,
):
    orders = [target_order - 1, target_order, target_order + 1]

    def get_or_compute_metrics(checkpoint: int):
        db = TensorDatabase(str(db_path / "tensor_db"), str(db_path / "tensors"))

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

        for order_index, order in enumerate(orders):
            for metric in metrics:
                tags.update({
                    'metric': metric,
                    'order': order,
                })
                if modality == 'language':
                    tags['num-shards'] = 2
                    if metric == 'loss':
                        tags['order'] = 'baseline'
                
                metric_results[metric] = db.query_last(**tags)
                if metric_results[metric] is None:
                    missing_metrics.append(metric)
                
            if len(missing_metrics) > 0:
                model = models[checkpoint]

                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                metric_results.update(metric_fn[modality](model, dataloader, order_index, missing_metrics))
                for metric, tensor in metric_results.items():
                    tags['metric'] = metric
                    db.add_tensor(tensor, tags)
            

        db.close()

        return metric_results
    
    return get_or_compute_metrics