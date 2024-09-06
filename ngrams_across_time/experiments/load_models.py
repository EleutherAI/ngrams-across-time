from ngrams_across_time.language.hf_client import get_basic_pythia_model_names, get_model_checkpoints as get_language_checkpoints
from ngrams_across_time.image.load_image_model_data import get_available_image_models, get_available_checkpoints as get_image_checkpoints
from ngrams_across_time.language.load_language_model_data import get_models, get_ngram_dataset, get_ngram_examples
from ngrams_across_time.image.load_image_model_data import load_models_and_synthetic_images

from auto_circuit.utils.graph_utils import patchable_model

import torch


def load_models(
        modality: str, 
        model_name: str, 
        order: int, 
        dataset: str = None, 
        max_ds_len: int = 1024, 
        start: int = 0, 
        end: int = 16384,
        patchable: bool = False,
        max_seq_len: int = 10,
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if modality == 'language':
            models, vocab_size = get_models(
                model_name, 
                order, 
                start,
                end,
                max_ds_len,
                patchable,
            )
            if patchable:
                dataset = get_ngram_dataset(
                    vocab_size,  
                    order, 
                    max_ds_len
                )
            else:
                dataset = get_ngram_examples(
                    model_name, 
                    max_ds_len
                )
        else:
            models, dataset = load_models_and_synthetic_images(
                model_name, 
                order,
                dataset,
                patchable
            )

    except Exception as e:
        print(f"Error loading model and dataset: {e}")
        print("This is likely because an unavailable model or dataset was specified.")
        if modality == "language":
            models = get_basic_pythia_model_names()
            print("Available language models:")
            for model in models:
                print(f"- {model}")
            checkpoints = get_language_checkpoints(model)
            print(f"Available checkpoints for {model}:")
            for step, revision in sorted(checkpoints.items()):
                print(f"Step: {step}, Revision: {revision}")
        else:  # image
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

    return models, dataset