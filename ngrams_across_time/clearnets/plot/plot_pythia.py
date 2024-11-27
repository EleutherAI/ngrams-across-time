import torch
from torch import Tensor
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
import plotly.io as pio

from ngrams_across_time.language.hf_client import get_model_checkpoints

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


def plot_pythia(stop: int | None = None):
    model_name = "pythia-160m"
    out_path = Path(f"workspace/inference/{model_name}.pth")

    checkpoints = get_model_checkpoints(f"EleutherAI/{model_name}")    
    log2_keys = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16_000, 33_000, 66_000, 131_000, 143_000]
    log_checkpoints = {key: checkpoints[key] for key in log2_keys}

    stop = stop or len(log_checkpoints)

    checkpoint_data = torch.load(out_path)

    def add_scores_if_exists(
            fig, key: str, name: str, normalize: bool = False,
            use_secondary_y=False
        ):
        if key not in checkpoint_data[list(log_checkpoints.keys())[0]].keys():
            print(f"Skipping {name} because it does not exist in the checkpoint")
            return
        
        scores = [scores[key] for scores in list(checkpoint_data.values())[:stop]]
        if isinstance(scores[0], Tensor):
            scores = [score.item() for score in scores]

        # Normalize scores to be between 0 and 1
        if normalize:
            max_score = max(scores)
            min_score = min(scores)
            scores = [(score - min_score) / (max_score - min_score) for score in scores]
        
        fig.add_trace(
            go.Scatter(x=list(log_checkpoints.keys()), y=scores, mode='lines', name=name), 
            secondary_y=use_secondary_y
        )

    images_path = Path("images")
    images_path.mkdir(exist_ok=True)
    
    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', checkpoint_data)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', checkpoint_data)
    add_scores_if_exists(fig, 'sae_entropy', f'H(SAE Nodes)', checkpoint_data, use_secondary_y=True
    )
    fig.update_layout(
        title=f"Loss and SAE Node Entropy over Steps (Pythia-160M)", 
        xaxis_title="Step", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'pythia_feature_sharing_entropy.pdf', format='pdf')

    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', checkpoint_data)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', checkpoint_data)

    add_scores_if_exists(fig, 'hoyer', 'Hoyer', checkpoint_data, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and SAE Hoyer Norm over Steps (Pythia-160M)", 
        xaxis_title="Step", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'pythia_feature_sharing_hoyer.pdf', format='pdf')


    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', checkpoint_data)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', checkpoint_data)

    add_scores_if_exists(fig, 'gini', 'Gini coefficient', checkpoint_data, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and Gini Coefficient of Mean SAE Activations over Steps (Pythia-160M)", 
        xaxis_title="Step", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'pythia_feature_sharing_gini.pdf', format='pdf')


    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', checkpoint_data)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', checkpoint_data)
    add_scores_if_exists(fig, 'parameter_norm', 'Parameter norm', checkpoint_data)
    fig.update_layout(
        title=f"Loss and Parameter Norm over Steps (Pythia-160M)", 
        xaxis_title="Step", 
        yaxis_title="Loss",
        width=1000
    )                                           
    fig.write_image(images_path / f'pythia_parameter_norm.pdf', format='pdf')
