import re
import time

from huggingface_hub import HfApi
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def get_model_size(model_name: str): 
    match = re.search(r'(\d+(?:\.\d+)?[mb]?)', model_name.split("-")[-1], re.IGNORECASE)
    if not match:
        return 0  # If no size found, treat it as the smallest
    
    size = match.group(1)
    if size.lower().endswith('b'):
        return int(float(size[:-1]) * 1e9)
    elif size.lower().endswith('m'):
        return int(float(size[:-1]) * 1e6)
    else:
        return int(size)

def filter_basic_pythia_models(models):
    basic_models = []
    pattern = r'^EleutherAI/pythia-(\d+(?:\.\d+)?[mb])$'
    
    for model in models:
        if re.match(pattern, model, re.IGNORECASE):
            basic_models.append(model)
    
    return sorted(basic_models, key=lambda x: float(re.search(r'(\d+(?:\.\d+)?)', x).group(1))) # type: ignore


def get_basic_pythia_model_names():
    api = HfApi()
    models = api.list_models(author="EleutherAI", search="pythia")
    basic_models = filter_basic_pythia_models([model.modelId for model in models]) # type: ignore

    # Sort models by size
    def model_size(model_id):
        # Extract the size part from the model name
        size_match = re.search(r'(\d+(?:\.\d+)?[mb]?)', model_id.split("-")[-1], re.IGNORECASE)
        if not size_match:
            return 0  # If no size found, treat it as the smallest
        
        size = size_match.group(1)
        if size.lower().endswith('b'):
            return float(size[:-1]) * 1000
        elif size.lower().endswith('m'):
            return float(size[:-1])
        else:
            return float(size)
    
    return sorted(basic_models, key=model_size)


def load_with_retries(model_name: str, revision: str, model_size: int):
    for retry in range(3):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                revision=revision,
                cache_dir=".cache",
                quantization_config=BitsAndBytesConfig(load_in_4bit=True) if model_size > 6e9 else None
            ).cuda()
            return model
        except Exception as e:
            if retry < 2:
                print(f"Attempt {retry + 1} failed, retrying in 2 seconds...", e)
                time.sleep(2)
            else:
                return None