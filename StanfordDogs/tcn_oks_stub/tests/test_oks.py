import numpy as np
from metrics.oks import oks_single, oks_map

def test_oks_single_simple():
    gt = np.zeros((3,2), dtype=np.float32)
    pred = np.zeros((3,2), dtype=np.float32)
    vis = np.array([1,1,1], dtype=np.int32)
    val = oks_single(pred, gt, vis, scale=100.0)
    assert 0.9 <= val <= 1.0

def test_oks_map_shapes():
    g = [np.zeros((3,2), dtype=np.float32) for _ in range(5)]
    p = [np.zeros((3,2), dtype=np.float32) for _ in range(5)]
    v = [np.ones((3,), dtype=np.int32) for _ in range(5)]
    s = [100.0 for _ in range(5)]
    res = oks_map(p, g, v, s)
    assert "mean_oks" in res and "oks_map" in res
