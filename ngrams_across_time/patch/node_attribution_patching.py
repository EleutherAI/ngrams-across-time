from typing import Callable, Dict, Tuple, Any, Union
from einops import rearrange
import torch
from torch import nn
from contextlib import contextmanager
import pdb

class _Finished(Exception):
    pass

def process_backward_conv(noise: torch.Tensor, clean: torch.Tensor, grad_output: torch.Tensor, *args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonally project clean activations to noise activations and reshape grad_output correctly for conv nets
    """
    # We treat the feature maps analogously to the sequence dimension in tranformers
    direction = noise - clean
    return direction, grad_output

def get_tensor_process_fn() -> Union[
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]],
    Callable[[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]],
    Callable[[None, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
]:
    return process_backward_conv

@contextmanager
def patch_model(
    model: nn.Module,
    noise_acts: Dict[str, torch.Tensor],
    patch_scores: Dict[str, torch.Tensor],
    quantile: float = 0.9
):
    handles = []
    mod_to_noise = {}
    mod_to_name = {}

    flat_scores = torch.cat([v.flatten() for v in patch_scores.values()])
    threshold = torch.quantile(flat_scores, quantile)

    def fwd_hook(module: nn.Module, input: tuple[torch.Tensor, ...] | torch.Tensor, output: tuple[torch.Tensor, ...] | torch.Tensor):
        output[patch_scores[mod_to_name[module]] > threshold] = mod_to_noise[module][patch_scores[mod_to_name[module]] > threshold]

    for name, module in model.named_modules():
        mod_to_name[module] = name
        for path, noise in noise_acts.items():
            if not name == path.rstrip('.output'):
                continue
            handles.append(module.register_forward_hook(fwd_hook))
            mod_to_noise[module] = noise

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad()

@contextmanager
def prepare_model_for_effects(
    model: nn.Module,
    noise_acts: Dict[str, torch.Tensor] | Dict[str, Tuple[torch.Tensor, torch.Tensor]] | None,
    head_dim: int = 0,
):
    effects = {}
    handles = []
    mod_to_clean = {}
    mod_to_noise = {}
    mod_to_name = {}
    names = list(noise_acts.keys())

    def bwd_hook(module: nn.Module, grad_input: tuple[torch.Tensor, ...] | torch.Tensor, grad_output: tuple[torch.Tensor, ...] | torch.Tensor):
        grad_output = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        clean = mod_to_clean[module]
        direction, grad_output = get_tensor_process_fn()(mod_to_noise[module], clean, grad_output, head_dim)
        effect = torch.linalg.vecdot(direction, grad_output.type_as(direction))
        name = mod_to_name[module]
        effects[name] = effect.clone().detach()

    def fwd_hook(module: nn.Module, input: tuple[torch.Tensor, ...] | torch.Tensor, output: tuple[torch.Tensor, ...] | torch.Tensor):
        output = output[0] if isinstance(output, tuple) else output
        mod_to_clean[module] = output.clone().detach()
    
    for name, module in model.named_modules():
        mod_to_name[module] = name
        for path, noise in noise_acts.items():
            if not name == path.rstrip('.output'):
                continue
            handles.append(module.register_full_backward_hook(bwd_hook))
            handles.append(module.register_forward_hook(fwd_hook))
            mod_to_noise[module] = noise

    try:
        yield effects
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad()

def get_effects(
    *args,
    model: nn.Module,
    noise_acts: Dict[str, torch.Tensor] | Dict[str, Tuple[torch.Tensor, torch.Tensor]] | None,
    output_func: Callable[[torch.Tensor], torch.Tensor],
    head_dim: int = 0,
    **kwargs
) -> dict[str, torch.Tensor]:
    """Get the approximate effects of ablating model activations with noise_acts for the given inputs
    using attribution patching.

    Args:
        model: The model to get the effects from.
        names: The names of the modules to get the effects from.
        noise_acts: A dictionary of noise activations for attribution patching.
        output_func: A function that takes the output of the model and reduces
            to a (batch_size, ) shaped tensor.
        tensor_process_fns: A dictionary of functions that transform tensors in a manner appropriate for the particular model
        head_dim: The size of attention heads (if applicable).
        *args: Arguments to pass to the model.
        **kwargs: Keyword arguments to pass to the model.

    Returns:
        A dictionary mapping the names of the modules to the attribution effects.
    """
    with prepare_model_for_effects(model, noise_acts, head_dim) as effects:
        with torch.enable_grad():
                out = model(*args, **kwargs)
                out = output_func(out, *args, 'out')
                assert out.ndim == 1, "output_func should reduce to a 1D tensor"
                out.backward(torch.ones_like(out))
    return effects


def get_activations(
    *args, model: torch.nn.Module, names: list[str], enable_grad: bool = False, **kwargs
) -> dict[str, torch.Tensor]:
    """Get the activations of the model for the given inputs.

    Args:
        model: The model to get the activations from.
        names: The names of the modules to get the activations from. Should be a list
            of strings corresponding to pytorch module names, with ".input" or ".output"
            appended to the end of the name to specify whether to get the input
            or output of the module.
        *args: Arguments to pass to the model.
        **kwargs: Keyword arguments to pass to the model.

    Returns:
        A dictionary mapping the names of the modules to the activations of the model
        at that module. Keys contain ".input" or ".output" just like `names`.
    """
    activations = {}
    hooks = []

    try:
        all_module_names = [name for name, _ in model.named_modules()]

        for name in names:
            assert name.endswith(".input") or name.endswith(
                ".output"
            ), f"Invalid name {name}, names should end with '.input' or '.output'"
            base_name = ".".join(name.split(".")[:-1])
            assert (
                base_name in all_module_names
            ), f"{base_name} is not a submodule of the model"

        def make_hook(name, is_input):
            def hook(module, input, output):
                if is_input:
                    if isinstance(input, torch.Tensor):
                        activations[name] = input
                    elif isinstance(input[0], torch.Tensor):
                        activations[name] = input[0]
                    else:
                        raise ValueError(
                            "Expected input to be a tensor or tuple with tensor as "
                            f"first element, got {type(input)}"
                        )
                else:
                    activations[name] = (
                        output if isinstance(output, torch.Tensor) else output[0]
                        )

                if set(names).issubset(activations.keys()):
                    # HACK: stop the forward pass to save time
                    raise _Finished()

            return hook

        for name, module in model.named_modules():
            if name + ".input" in names:
                hooks.append(
                    module.register_forward_hook(make_hook(name + ".input", True))
                )
            if name + ".output" in names:
                hooks.append(
                    module.register_forward_hook(make_hook(name + ".output", False))
                )
        try:
            if enable_grad:
                with torch.enable_grad():
                    model(*args, **kwargs)
            else:
                model(*args, **kwargs)
        except _Finished:
            pass
    finally:
        # Make sure we always remove hooks even if an exception is raised
        for hook in hooks:
            hook.remove()

    return activations