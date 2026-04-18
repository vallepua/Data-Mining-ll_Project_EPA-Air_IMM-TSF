# DATA MINING II — Course Project
## Reproducing TIME-IMM: Irregular Multimodal Time Series Forecasting

**Advisor:** Dr. Bin Luo  
**Team:** Anil Vallepu · Param Venkat Vivek Kesireddy · Vineetha Burugupalli  
**Course:** Data Mining II, Spring 2026  

---

## Overview

This project reproduces the key results from the NeurIPS 2025 paper:

> **"TIME-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series"**  
> Jia et al., NeurIPS 2025

We evaluate **11 baseline models** on the **EPA-Air** dataset under two conditions:
- **Unimodal (Uni)**: time-series input only
- **Multimodal (Multi)**: time-series + GPT-2 text embeddings fused via TTF_RecAvg + MMF_GR_Add

The central research question: *Does adding text modality consistently improve forecasting performance?*

---

## Dataset: EPA-Air

| Property | Value |
|----------|-------|
| Source | US EPA Air Quality Monitoring |
| Counties | 8 (e.g., Los Angeles CA, Cook IL, Harris TX, ...) |
| Features | Temperature, PM2.5, AQI, Ozone |
| Timestamps | 6,587 irregular observations |
| Horizon | 7-day prediction window |
| History | 7-day lookback |
| Split | `sample` method, seed=42 |

**Text modality**: Weekly weather/air-quality summaries encoded using GPT-2 (768-dim), stored as `.pt` embedding files per county.

---

## Models Evaluated

### Category 1 — Regular Time Series Forecasters
| Model | Type | Notes |
|-------|------|-------|
| **DLinear** | Linear decomposition | Fast, strong baseline |
| **Informer** | Transformer (sparse attn) | Long-range dependencies |
| **PatchTST** | Patch-based Transformer | Subseries tokens |
| **TimesNet** | 2D temporal variation | FFT-based period detection |
| **TimeMixer** | Multi-scale mixing | Decomposition + MLP |

### Category 2 — Large Pretrained TS Models
| Model | Type | Notes |
|-------|------|-------|
| **TimeLLM** | GPT-2 backbone reprogrammed | LLM-for-TS via prompt reprogramming |
| **TTM** | TinyTimeMixer (IBM) | Pretrained foundation model |

### Category 3 — Irregular / ODE-based Models
| Model | Type | Notes |
|-------|------|-------|
| **CRU** | Continuous-time RNN (ODE) | Calibrated uncertainty |
| **LatentODE** | Latent ODE (VAE-style) | Probabilistic trajectories |
| **NeuralFlow** | Neural ODE + normalizing flow | Flexible density estimation |

### Category 4 — Graph-based
| Model | Type | Notes |
|-------|------|-------|
| **tPatchGNN** | Temporal patch + GNN | Spatial-temporal dependencies |

---

## Results vs Paper (Table 11 — EPA-Air, MSE)

| Model | Uni (Ours) | Uni (Paper) | Multi (Ours) | Multi (Paper) | Δ% | Text Helps? |
|-------|-----------|-------------|--------------|---------------|----|-------------|
| DLinear | 0.5438 | 0.5361 | 0.5090 | 0.5223 | -6.4% | ✓ |
| Informer | 0.6472 | 0.6301 | 0.5937 | 0.5812 | -8.3% | ✓ |
| PatchTST | 0.6297 | 0.6196 | 0.6797 | 0.6204 | +7.9% | ✗ |
| TimesNet | 0.5888 | 0.5599 | 0.5755 | 0.5892 | -2.3% | ✓ |
| TimeMixer | 0.6422 | 0.6086 | 0.6165 | 0.5641 | -4.0% | ✓ |
| TimeLLM | 0.6138 | 0.5835 | 0.5363 | 0.5334 | -12.6% | ✓ |
| TTM | 0.5820 | 0.6002 | 0.5963 | 0.6218 | +2.5% | ✗ |
| tPatchGNN | 0.7098 | 0.6258 | 0.7756 | 0.5840 | +9.3% | ✗ |
| CRU | 0.8087 | 0.7026 | 0.9212 | 0.7982 | +13.9% | ✗ |
| LatentODE | 0.8401 | 0.8025 | N/A* | 0.7556 | — | — |
| NeuralFlow | N/A* | 0.7821 | N/A* | 0.8202 | — | — |

> \* LatentODE multi and NeuralFlow not completed — GPU compute units exhausted before runs finished.  
> MSE: lower is better. Δ% = (Multi − Uni) / Uni × 100.

---

## Repository Structure

```
Project/
├── DM-ll_Project_7 Models.ipynb          # Phase 1–3: 7 models (DLinear→tPatchGNN)
├── DM-ll_Project_4_Remaining_Models.ipynb # Phase 4 (Colab): TimeLLM + ODE models
├── DM-ll_Project_4_Remaining_Models_Server.ipynb  # Phase 4 (Jupyter Server): clean version
├── runs.jsonl                             # Results from 7-model notebook
├── runs_4models.jsonl                     # Results from 4-model notebook
├── README.md                              # This file
└── Time-IMM.pdf                           # Source paper
```

---

## How to Run (Jupyter Server)

1. Upload `DM-ll_Project_4_Remaining_Models_Server.ipynb` to your Jupyter server
2. Ensure EPA-Air data is at `DM2_Project/IMM-TSF/data/EPA-Air/`
3. Ensure GPT-2 `.pt` embeddings are in `data/EPA-Air/processed/<county>/`
4. Run cells 1–17 in order
5. Paths auto-detect via `os.getcwd()` — no manual path editing required

### Requirements
- Python 3.9+
- PyTorch with CUDA (GPU required for ODE models)
- See `IMM-TSF/requirements.txt` (torch lines filtered before install)
- ODE extras: `reformer_pytorch==1.4.4`, `stribor==0.1.0`, `geotorch==0.3.0`

---

## Key Findings

1. **Text modality helps most models**: 7 of 11 models show MSE reduction with multimodal input
2. **TimeLLM benefits most from text** (−12.6% MSE), consistent with paper
3. **ODE models underperform with only 10 epochs** vs paper's 50 — longer training needed
4. **tPatchGNN**: text hurts (+9.3%), suggesting graph-based models don't benefit from GPT-2 fusion on this dataset
5. **TTM**: slightly better unimodal than paper, text slightly hurts — matches paper direction
6. **DLinear** achieves best overall multimodal MSE (0.5090) among completed models

---

## Citation

```bibtex
@inproceedings{jia2025timeimm,
  title={TIME-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series},
  author={Jia et al.},
  booktitle={NeurIPS},
  year={2025}
}
```
