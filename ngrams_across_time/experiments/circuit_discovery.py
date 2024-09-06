


from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.utils.graph_utils import patch_mode, patchable_model

from ngrams_across_time.language.patch_language import get_patchable_model as get_patchable_model_language


def get_patchable_model(model_name: str, revision: str, device: torch.device, seq_len: int, slice_output: OutputSlice = "last_seq"):
    if modality == 'language':
        return get_patchable_model_language(model_name, revision, device, seq_len, slice_output)
    else:
        return get_patchable_model_image(model_name, revision, device, seq_len, slice_output)