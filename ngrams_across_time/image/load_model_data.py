from typing import Tuple, List, Literal
from pathlib import Path
import pickle

import torch
import torchvision.transforms as T
from torch import Tensor
from transformers import ConvNextV2ForImageClassification
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset, DataLoader

from concept_erasure import QuadraticFitter, QuadraticEditor
from concept_erasure.quantile import QuantileNormalizer
from concept_erasure.utils import assert_type

from ngrams_across_time.image.data_types import (
    ConceptEditedDataset,
    QuantileNormalizedDataset,
    MatchedEditedDataset
)

def get_model_path(model_name: str, dataset_name: str, chkpt: int) -> str:
    base_path = Path(f'/mnt/ssd-1/lucia/features-across-time/img-ckpts/{dataset_name}/{model_name}')
    return base_path / f'checkpoint-{chkpt}'

def get_available_checkpoints(model_name: str, dataset_name: str) -> List[int]:
    base_path = Path(f'/mnt/ssd-1/lucia/features-across-time/img-ckpts/{dataset_name}/{model_name}')
    return [int(x.name.split('-')[-1]) for x in base_path.iterdir()]

def load_models_and_dataset(model_name: str, dataset_name: str, ce_type: Literal['got','qn']):
    checkpoints = get_available_checkpoints(model_name, dataset_name)
    dataset = prepare_dataset(dataset_name, ce_type)
    
    def model_loader():
        for ckpt in checkpoints:
            model_path = get_model_path(model_name, dataset_name, ckpt)
            model = ConvNextV2ForImageClassification.from_pretrained(model_path).cuda()
            yield ckpt, model
            del model
            torch.cuda.empty_cache()
    
    return model_loader(), dataset

def infer_columns(feats):
    from datasets import Image, ClassLabel
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]
    assert len(img_cols) == 1 and len(label_cols) == 1
    return img_cols[0], label_cols[0]


def prepare_dataset(dataset_str: str, ce_type: Literal['got','qn']):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    path, _, name = dataset_str.partition(":")
    ds = load_dataset(path, name or None)
    assert isinstance(ds, DatasetDict)

    img_col, label_col = infer_columns(ds["train"].features)
    labels = ds["train"].features[label_col].names
    print(f"Classes in '{dataset_str}': {labels}")

    def preprocess(batch):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
        ])
        return {
            'pixel_values': [transform(x) for x in batch[img_col]],
            'label': torch.tensor(batch[label_col]),
        }

    ds = ds.with_transform(preprocess)

    # Infer the image size from the first image
    example = ds["train"][0]['pixel_values']
    c, h, w = example.size()
    print(f"Image size: {h} x {w}")

    train = ds["train"].with_format("torch")
    X = assert_type(Tensor, train[img_col]).div(255)
    Y = assert_type(Tensor, train[label_col])

    print("Computing statistics...")
    fitter = QuadraticFitter.fit(X.flatten(1).cuda(), Y.cuda())
    normalizer = QuantileNormalizer(X, Y)
    print("Done.")

    # Use the validation set if available, otherwise split the test set
    if val := ds.get("validation"):
        test = ds["test"].with_transform(preprocess) if "test" in ds else None
        val = val.with_transform(preprocess)
    else:
        nontrain = ds["test"].train_test_split(train_size=1024, seed=seed)
        val = nontrain["train"].with_transform(preprocess)
        test = nontrain["test"].with_transform(preprocess)

    class_probs = torch.bincount(Y).float()

    cache = Path.cwd() / "editor-cache" / f"{dataset_str}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            editor = pickle.load(f)
    else:
        print("Computing optimal transport maps...")

        editor = fitter.editor("cpu")
        cache.parent.mkdir(exist_ok=True)

        with open(cache, "wb") as f:
            pickle.dump(editor, f)

    with val.formatted_as("torch"):
        X = assert_type(Tensor, val[img_col]).div(255)
        Y = assert_type(Tensor, val[label_col])

    val_sets = {
        "got_dataset": ConceptEditedDataset(class_probs, editor, X, Y),
        "normal_dataset": val,
        "qn_dataset": QuantileNormalizedDataset(class_probs, normalizer, X, Y),
    }

    return MatchedEditedDataset(**val_sets, ce_type=ce_type)
