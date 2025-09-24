# duration_validation_helper.py
# Minimal helpers to validate HSMM durations against bout statistics.

from typing import List, Dict
import numpy as np

def bout_lengths_from_states(seq_ids: np.ndarray) -> List[int]:
    """
    Convert a sequence of integer states [T] into a list of bout lengths.
    """
    seq_ids = np.asarray(seq_ids).astype(int)
    if seq_ids.size == 0: return []
    lengths, cur, n = [], int(seq_ids[0]), 1
    for s in seq_ids[1:]:
        s = int(s)
        if s == cur: n += 1
        else:
            lengths.append(n); cur = s; n = 1
    lengths.append(n)
    return lengths

def duration_summary_per_state(state_seq_2d: np.ndarray, n_states: int) -> Dict[int, Dict[str, float]]:
    """
    state_seq_2d: (B, T) integer states. Returns per-state mean/median.
    """
    summary = {}
    for sid in range(n_states):
        lens = []
        for row in state_seq_2d:
            lens.extend([l for s, l in _segments(row) if s == sid])
        if len(lens) == 0:
            summary[sid] = {"count": 0, "mean": 0.0, "median": 0.0}
        else:
            arr = np.array(lens)
            summary[sid] = {"count": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr))}
    return summary

def _segments(seq: np.ndarray):
    out = []
    cur, n = seq[0], 1
    for s in seq[1:]:
        if s == cur: n += 1
        else:
            out.append((int(cur), n)); cur, n = s, 1
    out.append((int(cur), n))
    return out