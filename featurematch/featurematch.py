from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch

EPS = 1e-9

@dataclass
class AlignmentResult:
    """
    Minimal v0.1 alignment result for cosine-based cross-representation matching.
    - cosine: [K_a, K_b] cosine matrix (CPU).
    - top_matches: for each feature in A: list of (idx_b, score) pairs (length = topk).
    - stats: {'mean_best','median_best','pct_above_threshold'}.
    """
    cosine: torch.Tensor
    top_matches: List[List[Tuple[int, float]]]
    stats: Dict[str, float]

def _to_device(x: torch.Tensor, device: str) -> torch.Tensor:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device)

def _center_and_normalize(Z: torch.Tensor) -> torch.Tensor:
    """
    Center columns (remove baseline activations) and L2-normalize.
    Centering removes constant baseline activations, focusing cosine on relative patterns.
    """
    Z = Z - Z.mean(dim=0, keepdim=True)
    norms = torch.norm(Z, dim=0, keepdim=True)
    return Z / (norms + EPS)

def _cosine_blocked(A: torch.Tensor, B: torch.Tensor, block: int = 4096) -> torch.Tensor:
    """
    Compute A^T @ B in blocks along B's columns. A,B are [N, K] (centered+normalized).
    Returns [K_a, K_b] cosine matrix. Memory-safe for large K.
    """
    K_a = A.shape[1]
    K_b = B.shape[1]
    out = A.new_zeros((K_a, K_b))
    for j0 in range(0, K_b, block):
        j1 = min(j0 + block, K_b)
        out[:, j0:j1] = A.T @ B[:, j0:j1]
    return out

def align_features(
    Z_a: torch.Tensor,  # [N, K_a]
    Z_b: torch.Tensor,  # [N, K_b]
    topk: int = 5,
    threshold: float = 0.8,
    device: str = "auto",
    block: int = 4096,
) -> AlignmentResult:
    """
    Compare features between two representations (e.g., two SAEs' codes) using cosine similarity.

    Assumptions
    ----------
    - Z_a, Z_b: dense float tensors with shape [N, K] (same N), collected at the SAME hook/layer on the SAME eval dataset.
    - v0.1 is codes-only; callers who want weight-based comparison should compute codes first.

    Steps
    -----
    1) Center columns (remove baseline activations).
    2) L2-normalize columns.
    3) Cosine matrix via blocked matmul.
    4) For each feature in A, take top-k matches in B.
    5) Stats on best matches: mean, median, pct_above_threshold.

    Returns
    -------
    AlignmentResult with CPU tensors and Python lists (easy to serialize).
    """
    assert Z_a.dim() == 2 and Z_b.dim() == 2, "Z_a and Z_b must be 2D [N,K]"
    assert Z_a.shape[0] == Z_b.shape[0], "Z_a and Z_b must share the same N (rows/samples)"

    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else (device if device != "auto" else "cpu")
    Za = _center_and_normalize(_to_device(Z_a, dev))
    Zb = _center_and_normalize(_to_device(Z_b, dev))

    S = _cosine_blocked(Za, Zb, block=block)  # [K_a, K_b]
    S_cpu = S.detach().to("cpu")

    # top-k per feature in A
    k = min(topk, S_cpu.shape[1])
    vals, idxs = torch.topk(S_cpu, k=k, dim=1)  # [K_a, k]
    top_matches: List[List[Tuple[int, float]]] = []
    for i in range(S_cpu.shape[0]):
        row = [(int(idxs[i, j]), float(vals[i, j])) for j in range(k)]
        top_matches.append(row)

    # stats on best-match scores
    best = vals[:, 0]
    mean_best = float(best.mean().item())
    median_best = float(best.median().item())
    pct_above = float((best >= threshold).float().mean().item() * 100.0)

    stats = {
        "mean_best": mean_best,
        "median_best": median_best,
        "pct_above_threshold": pct_above,
    }

    return AlignmentResult(cosine=S_cpu, top_matches=top_matches, stats=stats)
