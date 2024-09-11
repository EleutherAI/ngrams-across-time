from einops import rearrange

from auto_circuit.utils.misc import tqdm
from auto_circuit.types import PruneScores, Edge
from auto_circuit.utils.patchable_model import PatchableModel

def compile_edge_scores(edge_scores: PruneScores, threshold: float, rec_model: PatchableModel) -> list[Edge]:
    keep_edges = []
    top_edges_with_scores = []
    for module, scores in tqdm(edge_scores.items(), 'compiling edge scores'):
        scores = rearrange(scores, 'dest_idx src_idx -> (dest_idx src_idx)')
        high_score_indices = (scores > threshold).nonzero().squeeze(1)
        if high_score_indices.numel() == 0:
            continue
        keep_edges.extend([rec_model.edge_dict[None][i] for i in high_score_indices])
        top_edges_with_scores.extend([(rec_model.edge_dict[None][i].name, scores[i].item()) for i in high_score_indices])
    return keep_edges, top_edges_with_scores