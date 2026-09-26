# CM-S6: Cross-Modal Selective Scan via Parameter-Level Modulation for Multisource Remote Sensing

Official implementation of **CM-S6**, a cross-modal selective state-space mechanism that incorporates auxiliary-modality information into the generation of selective SSM parameters (Δ, B, C), together with input injection and output calibration controlled by five bounded learnable coefficients.

## Overview

- **Parameter-level cross-modal modulation**: auxiliary modalities modulate Δ (state retention), B (input writing), and C (state readout) within a single shared selective scan.
- **Five tanh-bounded coefficients**: η (CII), α (Δ), β (B), γ (C), ζ (output calibration) coordinate the input–parameter–output pathway.
- **Lightweight variant CM-S6(L)**: retains all modulation paths while simplifying scan branches for resource-constrained deployment.

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets

| Dataset | Modalities | Source |
|---|---|---|
| Houston 2013 | HSI + LiDAR DSM | 2013 IEEE GRSS Data Fusion Contest |
| Augsburg | HSI + SAR + DSM | Figshare |
| MUUFL Gulfport | HSI + LiDAR DSM | GatorSense, University of Florida |

Place the preprocessed data under `data/` following the structure in `configs/*.yaml`.

## Training

```bash
# Full CM-S6
cd Train
bash run.sh

# Ablation experiments
cd ../Ablation
bash run.sh

# Baseline comparisons
cd ../Comparison
bash run.sh
```

## Analysis

```bash
cd Analysis
bash run.sh
```

## Citation

If you find this code useful, please cite:

```
Zhou, J.; Qi, S.; Zhang, D.; Jiang, Z. CM-S6: Cross-Modal Selective Scan via Parameter-Level Modulation for Multisource Remote Sensing. Remote Sens. 2026.
```
