from typing import List, Literal, Dict, Any, Optional
from pathlib import Path
import pickle
import os
import pandas as pd


import torch
import torchvision.transforms as T
from torch import Tensor
from transformers import ConvNextV2ForImageClassification
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset, DataLoader

from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.data import PromptDataset

from concept_erasure import QuadraticFitter, QuadraticEditor
from concept_erasure.quantile import QuantileNormalizer
from concept_erasure.utils import assert_type

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODULE_DIR))

from ngrams_across_time.image.image_data_types import (
    ConceptEditedDataset,
    QuantileNormalizedDataset,
    IndependentCoordinateSampler,
    GaussianMixture,
)
from ngrams_across_time.utils.data import ZippedDataset

def get_available_image_models() -> List[str]:
    base_path = Path('/mnt/ssd-1/lucia/features-across-time/img-ckpts')
    models = []
    for dataset in os.listdir(base_path):
        dataset_path = base_path / dataset
        if os.path.isdir(dataset_path):
            for model in os.listdir(dataset_path):
                models.append(f"{model} ({dataset})")
    return sorted(models)

def get_model_path(model_name: str, dataset_name: str, chkpt: int) -> str:
    base_path = Path(f'/mnt/ssd-1/lucia/features-across-time/img-ckpts/{dataset_name}/{model_name}')
    return base_path / f'checkpoint-{chkpt}'

def get_available_checkpoints(model_name: str, dataset_name: str) -> List[int]:
    base_path = Path(f'/mnt/ssd-1/lucia/features-across-time/img-ckpts/{dataset_name}/{model_name}')
    return [int(x.name.split('-')[-1]) for x in base_path.iterdir()]

def get_image_models(
    model_name: str,
    dataset_name: str,
    start: int,
    end: int,
    patchable: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Dict[int, Any]:
    checkpoints = get_available_checkpoints(model_name, dataset_name)
    checkpoints = [ckpt for ckpt in checkpoints if start <= ckpt <= end]
    
    models = {}
    for ckpt in checkpoints:
        model_path = get_model_path(model_name, dataset_name, ckpt)
        model = ConvNextV2ForImageClassification.from_pretrained(model_path).cuda()
        if patchable:
            model = patchable_model(model, factorized=True, device=device)
        models[ckpt] = model
        del model
        torch.cuda.empty_cache()
    
    return models

# TODO: enable specification of order
def get_image_dataset(
        dataset_name: str, 
        return_type: Literal['edited', 'synthetic'], 
        patchable: bool = False,
        model_name: Optional[str] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        order: Optional[int] = None
) -> ZippedDataset | PromptDataset:
    ds = load_dataset(dataset_name)
    assert isinstance(ds, DatasetDict)

    if return_type == 'edited':
        ds = get_ce_datasets(ds, dataset_name)
    elif return_type == 'synthetic':
        ds = get_synthetic_datasets(ds, dataset_name)
    else:
        raise ValueError(f"Invalid return_type: {return_type}")
    if patchable:
        assert order is not None
        data_index_path = Path(f"{PROJECT_ROOT}/data/filtered/filtered-{order}-data-{model_name.replace('/', '--')}-{start_step}-{end_step}.csv")
        ds = get_patchable_image_dataset(ds, data_index_path, filter = return_type == 'edited')
    return ds


def get_patchable_image_dataset(ds: ZippedDataset, data_index_path: Path, filter: bool = False) -> Dict[int, PromptDataset]:
    if filter:
        data_indices = pd.read_csv(data_index_path)
        ds = ds.select(data_indices['sample_idx'].tolist())

    result = {}
    for target, low in zip(ds.target_dataset, ds.low_order_dataset):
        cls = int(target['label'])
        if cls not in result:
            result[cls] = PromptDataset([], [], [], [])
        
        result[cls].clean_prompts.append(target['pixel_values'])
        result[cls].corrupt_prompts.append(low['pixel_values'])
        result[cls].answers.append(target['label'])
        result[cls].wrong_answers.append(low['label'])

    return result

def infer_columns(feats):
    from datasets import Image, ClassLabel
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]
    assert len(img_cols) == 1 and len(label_cols) == 1
    return img_cols[0], label_cols[0]

def prepare_common_data(dataset: DatasetDict, dataset_name: str):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    img_col, label_col = infer_columns(dataset['train'].features)
    labels = dataset['train'].features[label_col].names
    print(f"Classes in dataset: {labels}")

    dataset = dataset.map(lambda example, idx: {'sample_idx': idx}, with_indices=True)

    def preprocess(batch):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
        ])
        return {
            'pixel_values': [transform(x) for x in batch[img_col]],
            'label': torch.tensor(batch[label_col]),
            'sample_idx': torch.tensor(batch['sample_idx']),
        }

    dataset = dataset.with_transform(preprocess)

    example = dataset["train"][0]['pixel_values']
    c, h, w = example.size()
    print(f"Image size: {h} x {w}")

    train = dataset["train"].with_format("torch")
    X = assert_type(Tensor, train[img_col]).div(255)
    Y = assert_type(Tensor, train[label_col])

    print("Computing statistics...")
    fitter = QuadraticFitter.fit(X.flatten(1).cuda(), Y.cuda())
    normalizer = QuantileNormalizer(X, Y)
    print("Done.")

    if val := dataset.get("validation"):
        test = dataset["test"].with_transform(preprocess) if "test" in dataset else None
        val = val.with_transform(preprocess)
    else:
        nontrain = dataset["test"].train_test_split(train_size=1024, seed=seed)
        val, test = nontrain["train"].with_transform(preprocess), nontrain["test"].with_transform(preprocess)

    class_probs = torch.bincount(Y).float()
    
    with val.formatted_as("torch"):
        X_val = assert_type(Tensor, val[img_col]).div(255)
        Y_val = assert_type(Tensor, val[label_col])

    return X, Y, X_val, Y_val, fitter, normalizer, class_probs, val, c, h, w, seed

def get_ce_datasets(dataset: DatasetDict, dataset_name: str, return_type: Literal['edited','synthetic']):
    X, Y, X_val, Y_val, fitter, normalizer, class_probs, val, c, h, w, seed = prepare_common_data(dataset, dataset_name)

    cache = Path.cwd() / "editor-cache" / f"{dataset_name}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            editor = pickle.load(f)
    else:
        print("Computing optimal transport maps...")
        editor = fitter.editor("cpu")
        cache.parent.mkdir(exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(editor, f)

    val_sets = {
        "normal_dataset": val,
        "qn_dataset": QuantileNormalizedDataset(class_probs, normalizer, X_val, Y_val, seed),
        "got_dataset": ConceptEditedDataset(class_probs, editor, X_val, Y_val, seed),
    }

    return ZippedDataset(
        low_order_dataset=val_sets['qn_dataset'], 
        target_dataset=val_sets['got_dataset'], 
        high_order_dataset=val_sets['normal_dataset'], 
        base_dataset=val_sets['normal_dataset']
    )

def get_synthetic_datasets(dataset: DatasetDict, dataset_name: str):
    X, Y, X_val, Y_val, fitter, normalizer, class_probs, val, c, h, w, seed = prepare_common_data(dataset, dataset_name)

    gaussian = GaussianMixture(
        class_probs, len(val), means=fitter.mean_x.cpu(), covs=fitter.sigma_xx.cpu(), shape=(c, h, w), seed=seed
    )

    val_sets = {
        "ics_dataset": IndependentCoordinateSampler(class_probs, normalizer, len(val), seed=seed),
        "gauss_dataset": gaussian,
        "normal_dataset": val,
    }

    return ZippedDataset(
        low_order_dataset=val_sets['ics_dataset'], 
        target_dataset=val_sets['gauss_dataset'], 
        high_order_dataset=val_sets['normal_dataset'], 
        base_dataset=val_sets['normal_dataset']
    )

def prepare_dataset(dataset_str: str, return_type: Literal['edited','synthetic']):

    path, _, name = dataset_str.partition(":")
    ds = load_dataset(path, name or None)
    assert isinstance(ds, DatasetDict)

    if return_type == 'edited':
        return get_ce_datasets(ds, dataset_str)
    elif return_type == 'synthetic':
        return get_synthetic_datasets(ds, dataset_str)
