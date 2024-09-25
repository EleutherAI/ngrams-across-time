from typing import Any, Type, TypeVar, cast, Tuple
from transformers.modeling_outputs import ModelOutput

import torch

T = TypeVar("T")

def set_seeds(seed=42):
    import random
    import numpy as np
    import torch
    
    random.seed(seed) # unused
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def get_logit_diff(model, input_tensor, correct_class, target_class):
    with torch.no_grad():
        output = model(input_tensor)
    logits = get_logits(output)
    if logits.ndim == 3:
        # Take only the last token's logits
        logits = logits[:, -1]
    if correct_class.ndim == 2:
        correct_class = correct_class.squeeze(1)
        target_class = target_class.squeeze(1)
    return ((logits[range(len(logits)), correct_class] - logits[range(len(logits)), target_class]).cpu().numpy(), 
            torch.nn.functional.cross_entropy(logits, correct_class.to(logits.device), reduction='none').cpu().numpy(), 
            torch.nn.functional.cross_entropy(logits, target_class.to(logits.device), reduction='none').cpu().numpy())


def get_loss(model, input_tensor, correct_class):
    with torch.no_grad():
        output = model(input_tensor)
    logits = get_logits(output)
    if logits.ndim == 3:
        logits = logits[:, -1]
    if correct_class.ndim == 2:
        correct_class = correct_class.squeeze(1)
    return (torch.nn.functional.cross_entropy(logits, correct_class.to(logits.device), reduction='none').cpu().numpy(),)


def get_logits(model_output: torch.Tensor | ModelOutput, out_slice: Tuple[slice | int, ...] = slice(None)) -> torch.Tensor:
    if isinstance(model_output, ModelOutput):
        assert hasattr(model_output, "logits")
        return model_output.logits[out_slice]
    return model_output[out_slice]