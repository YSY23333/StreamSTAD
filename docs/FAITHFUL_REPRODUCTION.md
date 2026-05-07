# Faithful Baseline Reproduction Policy

This project distinguishes two experiment tracks.

## Track A: Faithful Reproduction

Use this track for paper numbers.

Rules:

- Use each baseline's official training, inference, loss, proposal generation, NMS, and evaluation code.
- Only change the minimum adapter needed for sensor input shape and output export.
- Do not replace a paper's decoder/post-processing with our unified decoder for the main faithful table.
- Record the exact upstream commit or local code bundle path.

Current source status:

| Baseline | Source status | Faithful policy |
|---|---|---|
| DPWiT / WiFiTAD | Official code: `https://github.com/AVC2-UESTC/WiFiTAD` | Use official train/test/eval and Soft-NMS. Only adapt data paths or input channel shape if needed. |
| SlimSTAD | Local bundle exists at `../SlimSTAD/STAD-main`; no public GitHub found in the current search | Use the local original train/test/eval scripts and original post-processing. Only adapt paths/input shape. |
| MATR | Official code cloned from `https://github.com/skhcjh231/MATR_codebase` | Use official training/inference protocol. Only replace the video feature loader/projection with sensor feature adapter. |
| MOAD / Backtrace Mamba | No official code found in current public search | Do not report as faithful reproduction until official code is available or authors provide code. |

## Track B: Unified Adapter Experiments

Use this track for engineering smoke tests and early transfer experiments.

Files under `src/stadstream/models/*_signal.py` and `scripts/run_experiment.py` implement a unified input/output interface:

```text
input:  [B, C, T]
output: {"loc", "conf", "priors"}
```

These models are useful for checking data flow, online protocol, and metric code. They are not faithful reproductions of the original papers unless explicitly stated.

## Online Comparison Recommendation

For a fair online transfer table, there are two acceptable variants:

1. **Paper-faithful online table**: each method keeps its own official decoder/post-processing. This is faithful but post-processing differs.
2. **Unified decoder ablation table**: all methods export common intermediate predictions and use the same online decoder. This isolates encoder quality but is not the main faithful reproduction table.

The main paper should clearly label which table is being reported.

