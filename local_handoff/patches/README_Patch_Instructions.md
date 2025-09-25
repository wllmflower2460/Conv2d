
# Synchrony Ablation - Patch & Instructions
Updated: 2025-09-21

This pack fixes issues from your M1 ablation:
1) Conformal prediction sets can be empty -> guarantee non-empty sets.
2) VQ perplexity is a placeholder -> proper instrumentation helpers.
3) HSMM durations -> quick validation helpers to compare bout statistics.

What is inside
- patches/Ablation_Framework_Upgraded.patch - unified diff to apply to your current Ablation_Framework_Upgraded.py.
- patches/vq_perplexity_helper.py - functions to compute perplexity and usage from VQ assignments.
- patches/duration_validation_helper.py - functions to extract bout lengths and summaries for HSMM validation.

Apply the diff (recommended)
From your repo root (where Ablation_Framework_Upgraded.py lives):
    git apply patches/Ablation_Framework_Upgraded.patch
or, if not a git repo:
    patch -p0 < patches/Ablation_Framework_Upgraded.patch

The patch makes two changes:
- Replaces prediction_set_mask() to force-include top-1 label (non-empty sets).
- Changes calibration split from 0.30 -> 0.50 for better ECE stability on small val sets.

If the patch fails to apply due to drift, open the file and manually replace the function with the one in the diff.

Wire in VQ perplexity
Open your VectorQuantizerEMA2D and after computing assignments add:
    from patches.vq_perplexity_helper import vq_perplexity_from_indices
    # code_indices: LongTensor (...,) in [0, K-1]
    perplexity, usage = vq_perplexity_from_indices(code_indices, K=self.num_codes)
    vq_info = {**vq_info, "perplexity": perplexity, "usage": usage}
If you already produce one-hot encodings, use vq_perplexity_from_onehot(one_hot_assign) instead.

Targets: perplexity in [50, 200] and usage > 0.3 for healthy codebooks on your data.

Validate HSMM durations
After decoding state_seq to integer states (B, T):
    from patches.duration_validation_helper import duration_summary_per_state
    summary = duration_summary_per_state(state_seq_ids, n_states=HSMM_STATES)
    print(summary)

Compare means/medians per state with your ethogram bout stats. If mismatched:
- try Negative Binomial duration model,
- reduce max_duration,
- or remove input-dependent transitions and re-check calibration.

Re-run and check the decision line
1. Refit temperature scaling (now with 50 percent calibration split).
2. Ensure 90 percent coverage in [0.88, 0.92] and Avg Set ~ 1-2.
3. Verify ECE <= 0.03 (should fix hdp_only).

Troubleshooting
- Avg Set > 5 and Coverage ~1.0: probabilities are too flat; check temperature fitting and model logits.
- Empty sets (Avg Set < 1): you forgot to apply the conformal non-empty patch.
- Perplexity ~1.0: VQ not wired; confirm code indices are real and updated during training.
