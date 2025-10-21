# FeatureMatch — Cross-Representation Feature Alignment (v0.1)

**FeatureMatch** quantifies correspondence between features learned by two models (e.g., two SAEs) using **cosine similarity**.
It reports top-k matches per feature, summary stats, and a simple heatmap.

> v0.1 is intentionally minimal: cosine-only, top-k matches, basic stats, one heatmap, strict tests.

## Requirements
- Python **3.9+**
- PyTorch, NumPy, Matplotlib, PyTest (dev)

## Installation
```bash
pip install "git+https://github.com/Course-Correct-Labs/featurematch.git"
```

## Why Centering?
Centering removes constant baseline activations, focusing cosine on **relative** feature patterns rather than absolute magnitudes.

## Usage (5 lines)
```python
import torch
from featurematch.featurematch import align_features

res = align_features(Z_a, Z_b, topk=5, threshold=0.8)   # Z_* are [N, K] codes on same layer & dataset
print(res.stats)                                        # {'mean_best':..., 'median_best':..., 'pct_above_threshold':...}
```

## Assumptions

Z_a, Z_b are codes with shape [N, K] collected at the same hook/layer on the same eval dataset (dense float tensors).

## Interpretation Guide (heuristics)

- **mean_best ≥ 0.85**: strong reproducibility (dictionaries mostly aligned)
- **0.70–0.85**: partial alignment; seeds/hparams differ
- **< 0.70**: low alignment; likely different learned dictionaries
- **pct_above_threshold** (default thr = 0.8): quick sanity check; >60% is typically "good" for very similar runs

## Demo

See `notebooks/featurematch_demo.ipynb` for a working example with:
- Permutation recovery case (perfect alignment)
- Random dictionary case (baseline)
- Heatmap visualization

## Roadmap (not in v0.1)

- v0.2: Jaccard for binarized codes, permutation baseline
- v0.3: Hungarian matching, (optional) global CKA

## License

MIT © Course Correct Labs
