import re
import time
import requests
from typing import Callable

from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def get_model_checkpoints(model_id):
    api_url = f"https://huggingface.co/api/models/{model_id}/refs"
    response = requests.get(api_url)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    refs = response.json()

    revisions = {}
    for branch in refs['branches']:
        if not branch['ref'].startswith('refs/heads/'):
            continue

        ref_name = branch['ref'].split('/', 2)[-1]
        match = re.match(r'(?:step[_]?|revision)?(\d+)', ref_name)
        if match:
            step = int(match.group(1))
            revisions[step] = ref_name

    return revisions


def get_closest_checkpoint(revisions: dict[int, str], target_step: int):
    steps = sorted(revisions.keys())
    if not steps:
        return None
    
    closest = steps[0]
    min_diff = abs(target_step - closest)
    
    for step in steps:
        diff = abs(target_step - step)
        if diff < min_diff:
            min_diff = diff
            closest = step
    
    return closest


def get_pythia_model_size(model_name: str): 
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


def get_basic_pythia_model_names() -> list[str]:
    api = HfApi()
    models = api.list_models(author="EleutherAI", search="pythia")
    
    basic_models = []
    pattern = r'^EleutherAI/pythia-(\d+(?:\.\d+)?[mb])$'
    for model in models:
        model_name = model.modelId # type: ignore
        if re.match(pattern, model_name, re.IGNORECASE):
            basic_models.append(model_name)
    
    return sorted(basic_models, key=get_pythia_model_size)


def with_retries(fn: Callable, retries: int = 3):
    for retry in range(retries):
        try:
            return fn()
        except Exception as e:
            if retry < 2:
                print(f"Attempt {retry + 1} failed, retrying in 2 seconds...", e)
                time.sleep(2)
            else:
                return None


def load_with_retries(model_name: str, revision: str, model_size: int, retries: int = 3):
    def load():
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            revision=revision,
            cache_dir=".cache",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True) if model_size > 6e9 else None
        ).cuda()

    return with_retries(load, retries)