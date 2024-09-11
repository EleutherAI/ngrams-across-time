from typing import Literal, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset

from ngrams_across_time.language.hf_client import get_basic_pythia_model_names, get_model_checkpoints as get_language_checkpoints
from ngrams_across_time.language.load_language_model_data import get_models as get_language_models, get_ngram_dataset
from ngrams_across_time.image.load_image_model_data import get_available_image_models, get_image_models, get_image_dataset, get_available_checkpoints as get_image_checkpoints

def load_models(
    modality: Literal['language', 'image'],
    model_name: str,
    order: int,
    dataset_name: str = '',
    start: int = 0,
    end: int = 16384,
    patchable: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    max_seq_len: int = 10,
) -> Tuple[Dict[int, Any], Dataset]:

    if modality == 'language':
        if model_name not in get_basic_pythia_model_names():
            print("Unknown model.")
            if modality == "language":
                models = get_basic_pythia_model_names()
                print("Available language models:")
                for model in models:
                    print(f"- {model}")
                checkpoints = get_language_checkpoints(model)
                print(f"Available checkpoints for {model}:")
                for step, revision in sorted(checkpoints.items()):
                    print(f"Step: {step}, Revision: {revision}")
            return

        models, vocab_size = get_language_models(
            model_name,
            start=start,
            end=end,
            patchable=patchable,
            device=device,
            max_seq_len=max_seq_len
        )
        dataset = get_ngram_dataset(
            model_name,
            start,
            end,
            order,
            vocab_size,
            patchable=patchable,
        )

        # Pack models and vocab size into a tuple for later use
        models = (models, vocab_size)
    elif modality == 'image':
        if model_name not in [s.split(" (")[0] for s in get_available_image_models()]:
            models = get_available_image_models()
            print("Available image models:")
            for model in models:
                print(f"- {model}")
            model_name, dataset = model.split(" (")
            dataset = dataset.rstrip(")")
            checkpoints = get_image_checkpoints(model_name, dataset)
            print(f"Available checkpoints for {model_name} on {dataset}:")
            for checkpoint in sorted(checkpoints):
                print(f"Checkpoint: {checkpoint}")
            return
        
        models = get_image_models(
            model_name,
            dataset_name,
            start,
            end,
            patchable=patchable,
            device=device
        )
        dataset = get_image_dataset(
            dataset_name, 
            patchable=patchable, 
            return_type='synthetic',
            model_name=model_name,
            start_step=start,
            end_step=end,
            order=order
        )
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return models, dataset