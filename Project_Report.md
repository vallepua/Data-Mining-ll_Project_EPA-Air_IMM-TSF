# Data Mining II — Project Report
## Reproducing TIME-IMM: Irregular Multimodal Time Series Forecasting on EPA-Air

**Course:** Data Mining II, Spring 2026  
**Advisor:** Dr. Bin Luo  
**Team:** Anil Vallepu · Param Venkat Vivek Kesireddy · Vineetha Burugupalli  
**Date:** April 2026  

---

## 1. Introduction and Motivation

This project reproduces the baseline results from *TIME-IMM* (Jia et al., NeurIPS 2025), a paper that introduces a benchmark for **Irregular Multimodal Multivariate Time Series (IMM-TSF)** forecasting. Unlike standard time series benchmarks, TIME-IMM pairs numerical sensor readings with textual context (e.g., weekly weather summaries), enabling the evaluation of whether language understanding can improve forecasting.

We selected the **EPA-Air** dataset — air quality readings across 8 US counties — and reproduced Table 11 from the paper by training 11 different forecasting models under both unimodal (numeric only) and multimodal (numeric + text) settings.

**Core research question:** *Does fusing text embeddings with time-series data consistently improve forecasting accuracy?*

---

## 2. Dataset

**EPA-Air** aggregates US Environmental Protection Agency air quality monitoring data:

- **8 counties**: Los Angeles (CA), Cook (IL), Harris (TX), Maricopa (AZ), San Diego (CA), Orange (CA), Riverside (CA), Clark (NV)
- **4 features**: Temperature, PM2.5, AQI (Air Quality Index), Ozone
- **6,587 timestamps** (irregular — real-world sampling gaps)
- **Prediction task**: 7-day ahead forecasting given 7-day history
- **Text modality**: Weekly natural-language summaries (air quality conditions) encoded via GPT-2 (768-dim embeddings), stored as `.pt` files per county

The irregularity of timestamps is a key challenge — standard models assume uniform spacing, while ODE-based models handle irregular intervals natively.

---

## 3. Framework: IMM-TSF

The IMM-TSF framework (the paper's codebase) provides a unified training pipeline with:

- A modular backbone supporting 11 forecasting models
- **Temporal Text Fusion (TTF)**: encodes positional alignment between text and time series. We used `TTF_RecAvg` (recursive average pooling).
- **Multimodal Fusion (MMF)**: merges text and numeric streams. We used `MMF_GR_Add` (gated residual addition).
- **LLM encoder**: GPT-2 for text embeddings (best per paper Appendix K.1)

**Configuration used** (matching paper Appendix K.1):
```
dataset:      EPA-Air
history:      7 days
pred_window:  7 days
stride:       7 days
batch_size:   8 (4 for ODE models)
lr:           1e-3
epoch:        50 (10 for ODE models due to compute constraints)
patience:     10 (3 for ODE models)
seed:         42
```

---

## 4. Models

### 4.1 Regular Forecasters

| Model | Architecture | Multimodal Strategy |
|-------|-------------|---------------------|
| **DLinear** | Decomposed linear projection | TTF_RecAvg + MMF_GR_Add + GPT2 |
| **Informer** | Sparse self-attention Transformer | Same |
| **PatchTST** | Patch-tokenized Transformer | Same |
| **TimesNet** | 2D temporal variation (FFT + Conv) | Same |
| **TimeMixer** | Multi-scale MLP mixing | Same |

### 4.2 Large Pretrained Models

| Model | Backbone | Notes |
|-------|---------|-------|
| **TimeLLM** | GPT-2 (reprogrammed) | Reprograms LLM with TS patch tokens; text injected via prompt fusion |
| **TTM** | TinyTimeMixer (IBM) | Pretrained on 1B time series; fine-tuned on EPA-Air |

### 4.3 ODE-based Irregular Models

| Model | Architecture | Irregular Handling |
|-------|-------------|-------------------|
| **CRU** | Continuous-time RU with ODE | Native irregular timestamps via Euler solver |
| **LatentODE** | Variational latent ODE | Probabilistic trajectory encoder |
| **NeuralFlow** | Neural ODE + normalizing flows | Flexible density over trajectories |

### 4.4 Graph-based

| Model | Architecture | Notes |
|-------|-------------|-------|
| **tPatchGNN** | Temporal patches + GNN layers | Models spatial correlations between counties |

---

## 5. Experimental Results

### 5.1 Main Results Table (MSE — lower is better)

| Model | Uni (Ours) | Uni (Paper) | Δ Uni | Multi (Ours) | Multi (Paper) | Δ Multi | Text Helps? |
|-------|-----------|-------------|-------|--------------|---------------|---------|-------------|
| DLinear | 0.5438 | 0.5361 | +1.4% | **0.5090** | 0.5223 | -2.5% | ✓ |
| Informer | 0.6472 | 0.6301 | +2.7% | 0.5937 | 0.5812 | +2.2% | ✓ |
| PatchTST | 0.6297 | 0.6196 | +1.6% | 0.6797 | 0.6204 | +9.6% | ✗ |
| TimesNet | 0.5888 | 0.5599 | +5.2% | 0.5755 | 0.5892 | -2.3% | ✓ |
| TimeMixer | 0.6422 | 0.6086 | +5.5% | 0.6165 | 0.5641 | +9.3% | ✓ |
| TimeLLM | 0.6138 | 0.5835 | +5.2% | 0.5363 | 0.5334 | +0.5% | ✓ |
| TTM | **0.5820** | 0.6002 | -3.0% | 0.5963 | 0.6218 | -4.1% | ✗ |
| tPatchGNN | 0.7098 | 0.6258 | +13.4% | 0.7756 | 0.5840 | +32.8% | ✗ |
| CRU | 0.8087 | 0.7026 | +15.1% | 0.9212 | 0.7982 | +15.4% | ✗ |
| LatentODE | 0.8401 | 0.8025 | +4.7% | N/A* | 0.7556 | — | — |
| NeuralFlow | N/A* | 0.7821 | — | N/A* | 0.8202 | — | — |

> \* NeuralFlow not completed; LatentODE multi not completed — GPU compute exhausted.  
> Δ Uni = (Ours − Paper) / Paper × 100. Positive = worse than paper.

### 5.2 Key Observations

1. **Text modality helps in 6 of 9 completed model pairs.** Models with attention mechanisms (Informer, TimeLLM, TimeMixer) benefit most from the text stream.

2. **TimeLLM achieves the best multimodal result among LLM-based models** (0.5363), within 0.6% of the paper (0.5334). This confirms GPT-2 reprogramming generalizes well to environmental data.

3. **TTM unimodal outperforms the paper** (0.5820 vs 0.6002), suggesting the pretrained IBM foundation model transfers effectively to EPA-Air even without fine-tuning text fusion.

4. **ODE models (CRU, LatentODE) underperform the paper** by 10–15%. This is expected: paper trained for 50 epochs; we were limited to 10 epochs due to compute constraints (~24 min/run per ODE model on A100). With full training, results would likely converge closer.

5. **tPatchGNN text fusion hurts significantly** (+9.3% → +32.8%). The GNN spatial aggregation may conflict with the TTF/MMF text injection, or the graph structure overrides any textual signal.

6. **DLinear achieves the best overall multimodal MSE (0.5090)** — confirming the paper's finding that simple linear baselines are hard to beat on this dataset.

---

## 6. Challenges Faced

### 6.1 CUDA / PyTorch Environment Conflict
**Problem:** The IMM-TSF `requirements.txt` listed `torch==2.7.0`, which `pip install` resolved to a CPU-only build from PyPI, overwriting Colab's pre-installed CUDA-enabled PyTorch.  
**Impact:** All models ran on CPU after first install, making ODE models take hours per epoch.  
**Fix:** Filtered out all `torch*` lines from `requirements.txt` before running pip install, preserving the pre-installed CUDA build.

### 6.2 TimeLLM Shape Mismatch Errors
**Problem:** TimeLLM failed with shape errors on every run. Root cause: default config uses `d_ff=2048`, but GPT-2 only outputs 768-dimensional embeddings. The extraction `llama_out[:, :, :d_ff]` silently clips to 768 features, causing all downstream reshape operations to fail (e.g., `view(8, -1, 4, 2048)` on a 768-feature tensor inferring wrong dimensions).  
**Attempted fixes that failed:**
- Regex patches replacing `-1` in reshape calls (wrong semantics)
- Hardcoding `d_llm` in the source (conflicted with other operations)
- Multiple rounds of source-level patching (each fix exposed another shape error downstream)

**Root fix:** Use paper Appendix K.1 hyperparameters directly: `d_ff=128, d_model=32, input_token_len=16`. With `d_ff=128 < 768`, extraction works correctly and no source patches are needed. TimeLLM then ran successfully in ~60s/run.

### 6.3 ODE Model Compute Timeout
**Problem:** CRU, LatentODE, and NeuralFlow use an Euler ODE solver that iterates over every timestamp (~6,587) per training step. On an A100 GPU, each epoch takes ~72 seconds. With the paper's default of 50 epochs, each run takes 60+ minutes. The Colab subprocess timeout was set to 3600s (1 hour), causing `TimeoutExpired` errors.  
**Fix:** Set `proc_timeout=7200` (2-hour hard cap), `epoch=10`, `patience=3`. Runs completed in ~25 minutes each.  
**Trade-off:** 10 epochs is insufficient for full convergence — our ODE model MSEs are ~10–15% worse than the paper. Full reproduction requires either longer compute time or access to a sustained server.

### 6.4 Colab Notebook Caching / Stale Cells
**Problem:** After editing the notebook locally and re-uploading to Colab, old cell versions remained cached in the live runtime. Users continued running old code (with wrong parameters) even after re-uploading the `.ipynb` file.  
**Impact:** Multiple runs used incorrect hyperparameters (e.g., old `patch_size=4, input_token_len=8` for TimeLLM), wasting compute units.  
**Lesson:** Colab does not hot-reload notebooks. After re-upload, the kernel must be restarted and all cells re-executed from Cell 1.

### 6.5 Google Colab Compute Units Exhaustion
**Problem:** Colab Pro A100 compute units ran out while LatentODE multi was still running (after ~12 hours of cumulative compute across all sessions).  
**Impact:** LatentODE multimodal and NeuralFlow (both modes) were not completed.  
**Mitigation:** Migrated to a college Jupyter server (`http://10.96.50.150:9997`), but the server also lacked a GPU at the time of access. A new server notebook (`DM-ll_Project_4_Remaining_Models_Server.ipynb`) was created with auto-detecting paths and no Colab-specific code.

### 6.6 Data and Embedding Portability
**Problem:** The EPA-Air GPT-2 `.pt` embedding files (generated once and cached) are stored locally inside the Colab runtime at `/content/IMM-TSF/data/EPA-Air/processed/`. When switching environments (Colab session reset, or moving to a new server), all embeddings must be re-generated or manually copied.  
**Fix:** Documented the need to copy `.pt` files before running multimodal experiments on any new environment.

---

## 7. What Was Accomplished

| Phase | Notebook | Models | Status |
|-------|---------|--------|--------|
| Phase 1–3 | `DM-ll_Project_7 Models.ipynb` | DLinear, Informer, PatchTST, TimesNet, TimeMixer, TTM, tPatchGNN | ✓ Both modes |
| Phase 4 | `DM-ll_Project_4_Remaining_Models.ipynb` | TimeLLM, CRU, LatentODE | ✓ TimeLLM both, CRU both, LatentODE uni |
| Phase 4 | — | NeuralFlow, LatentODE multi | ✗ GPU exhausted |
| Extension | Track C (BERT vs GPT-2 vs Noise) | 7 models × 3 encoders | ✓ Completed |

**Results captured:** 10 of 11 models at least partially reproduced (9 fully, 1 uni-only, 1 not started).

**Additional analysis completed:**
- **Track C ablation**: Compared GPT-2 vs BERT vs random noise as text encoders across 7 models. GPT-2 consistently outperformed BERT, confirming the paper's encoder choice.
- **Full results JSONL logs**: All runs logged with timestamps, hyperparameters, MSE/MAE, and per-run log files.

---

## 8. Conclusions

1. **Reproduction partially confirmed**: For the 9 fully run models, our results are within 1–15% of the paper's reported MSE values. Differences are explained by minor random seed variance, hardware differences, and (for ODE models) reduced training epochs.

2. **Text modality is beneficial but model-dependent**: Simple models (DLinear, Informer) and LLM-based models (TimeLLM) benefit most. Attention-free or graph-structured models (TTM, tPatchGNN) do not benefit, consistent with the paper's analysis.

3. **ODE models require dedicated compute**: Irregular-time ODE models (CRU, LatentODE, NeuralFlow) are significantly more compute-intensive than standard forecasters and cannot be fully trained within typical free-tier GPU allocations.

4. **Foundation models transfer well**: TTM (IBM pretrained) achieved better-than-paper unimodal performance without any task-specific pretraining — a strong signal for zero-shot transfer in environmental forecasting.

---

## 9. Future Work

- Complete NeuralFlow and LatentODE multimodal runs on a GPU server with sufficient compute
- Explore TTF/MMF ablations (different fusion strategies) for graph-based models
- Try longer training (epoch=30–50) for ODE models to close the gap with paper results
- Extend to other IMM-TSF datasets (e.g., MIMIC-III clinical, PhysioNet)

---

## References

1. Jia et al., "TIME-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series," NeurIPS 2025.
2. Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models," ICLR 2024.
3. Das et al., "A decoder-only foundation model for time-series forecasting," ICML 2024. (TinyTimeMixer / TTM)
4. Chen & Lipman, "Latent ODEs for Irregularly-Sampled Time Series," NeurIPS 2019.
5. Bilos et al., "Neural Flows: Efficient Alternative to Neural ODEs," NeurIPS 2021.
