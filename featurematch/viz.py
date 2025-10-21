import matplotlib.pyplot as plt
import torch
from typing import Optional

def plot_heatmap(S: torch.Tensor, title: str = "FeatureMatch: Cosine", outpath: Optional[str] = None) -> None:
    """
    Single matplotlib heatmap of [K_a, K_b] cosine matrix (CPU tensor).
    """
    if S.is_cuda:
        S = S.detach().cpu()
    plt.figure(figsize=(7, 5))
    plt.imshow(S.numpy(), aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Features B")
    plt.ylabel("Features A")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    plt.close()
