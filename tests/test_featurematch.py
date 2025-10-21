import torch
from featurematch.featurematch import align_features

def test_permutation_recovery_strict():
    torch.manual_seed(0)
    N, K = 200, 64
    Z_a = torch.randn(N, K)
    perm = torch.randperm(K)
    P = torch.zeros(K, K)
    P[torch.arange(K), perm] = 1.0
    Z_b = Z_a @ P  # exact permutation

    res = align_features(Z_a, Z_b, topk=1, threshold=0.8, device="cpu", block=64)
    best_scores = torch.tensor([res.top_matches[i][0][1] for i in range(K)])
    # Close to 1.0 within strict tolerances
    torch.testing.assert_close(best_scores, torch.ones_like(best_scores), rtol=1e-4, atol=1e-6)

    # ≥95% exact index recovery
    recovered = sum(int(res.top_matches[i][0][0] == perm[i].item()) for i in range(K))
    assert recovered >= int(0.95 * K)

def test_mismatched_K():
    torch.manual_seed(1)
    N, Ka, Kb = 128, 64, 80
    Z_a = torch.randn(N, Ka)
    Z_b = torch.randn(N, Kb)
    res = align_features(Z_a, Z_b, topk=5, threshold=0.8, device="cpu", block=32)
    assert res.cosine.shape == (Ka, Kb)
    assert len(res.top_matches) == Ka
    assert all(len(row) == min(5, Kb) for row in res.top_matches)

def test_blocking_consistency():
    torch.manual_seed(2)
    N, Ka, Kb = 160, 48, 60
    Z_a = torch.randn(N, Ka)
    Z_b = torch.randn(N, Kb)
    res_big = align_features(Z_a, Z_b, topk=3, device="cpu", block=4096)
    res_small = align_features(Z_a, Z_b, topk=3, device="cpu", block=8)
    torch.testing.assert_close(res_small.cosine, res_big.cosine, rtol=1e-5, atol=1e-7)

def test_centering_invariance_to_offsets():
    torch.manual_seed(3)
    N, K = 120, 32
    Z_a = torch.randn(N, K)
    Z_b = Z_a.clone()
    Z_b = Z_b + torch.linspace(-2.0, 2.0, steps=K).view(1, K)
    res = align_features(Z_a, Z_b, topk=1, device="cpu")
    best_scores = torch.tensor([res.top_matches[i][0][1] for i in range(K)])
    assert float(best_scores.mean().item()) > 0.9
