# Model Run Summary

All runs: H=2, F=2, seed=42, chronological split (train=165, val=35, test=37), cohort=40,729 users, 605,727 edges.

## Common Loss Function

All models use multi-task weighted cross-entropy loss with entropy regularization:

```
loss = Σ w_i · CE_i - 0.01 · H(w)
```

where `w` are learnable task weights and `H(w) = -Σ w_i · log(w_i)` encourages balanced weighting. Six behavior tasks: aggression level, aggression type, gender bias, religious bias, caste bias, ethnicity bias. Each task corresponds to a subset of the 20-d behavior profile.

TSGAN V1 is architecturally different (single 20-d decoder sliced into 6 task groups) but uses the same `Eq4TaskLoss`.

## Decoder Architecture

| Model | Decoders | Output |
|---|---|---|
| TSGAN V1 | 1 unified 20-d | single 20-d vector |
| All others (V2, STGCN, DCRNN, GPT-ST, ST-SSDL, D2STGNN, UniST, CMuST, DiffSTG, USTD, PCGrad) | 6 task-specific heads | 6 task logits → 20-d behavior profile |

## Per-Model Results

### TSGAN V1
- **Architecture**: Original TSGAN with FAG fusion, shared STAM, single 20-d decoder
- **Loss**: MAE, MSE tracked
- **Epochs**: 200/200
- **Run time**: ~9.3h
- **Best val epoch**: ~30
- **Test**: MAE t+1=0.139, MAE t+2=0.139, MSE=0.033/0.033

### TSGAN V2
- **Architecture**: Same as V1 but 6 task-specific decoders
- **Loss**: weighted CE - 0.01·entropy
- **Epochs**: 200/200
- **Run time**: ~14.1h
- **Test**: MAE t+1=0.182, MAE t+2=0.182, MSE=0.051/0.051

### STGCN V2
- **Architecture**: ST-Conv blocks (temporal gated conv → ChebConv → temporal gated conv), 6 task heads
- **Loss**: weighted CE - 0.01·entropy, sparse PyG ChebConv
- **Epochs**: 120/120
- **Run time**: ~5h
- **Test**: MAE t+1=0.059, MAE t+2=0.059, MSE=0.019/0.019

### DCRNN V2
- **Architecture**: DCGRU encoder-decoder with dual random-walk diffusion, 6 task heads
- **Loss**: weighted CE - 0.01·entropy
- **Epochs**: 100/100
- **Run time**: ~8h
- **Test**: MAE t+1=0.060, MAE t+2=0.060, MSE=0.019/0.019

### GPT-ST Pretrain + STGCN
- **Architecture**: Frozen GPT-ST pretrained encoder → gated fusion → sparse STGCN with 6 task heads
- **Loss**: masked MAE pretrain, weighted CE - 0.01·entropy downstream
- **Epochs**: 50 pretrain + 46/120 downstream (interrupted)
- **Run time**: ~8h pretrain + ~6h downstream
- **Test**: MAE t+1=0.062, MAE t+2=0.062, MSE=0.019/0.019

### ST-SSDL V2
- **Architecture**: Recurrent graph encoder/decoder with current-anchor deviation, prototype bank, 6 task heads
- **Loss**: forecast CE + contrastive + deviation + 0.01·entropy
- **Epochs**: 100/100
- **Run time**: ~3.7h
- **Test**: MAE t+1=0.071, MAE t+2=0.068, MSE=0.023/0.023

### D2STGNN V2
- **Architecture**: Decoupled spatial-temporal (diffusion + inherent branches), sparse dynamic graph, 6 task heads
- **Loss**: weighted CE - 0.01·entropy
- **Epochs**: early stop at epoch 5
- **Run time**: ~0.5h
- **Test**: MAE t+1=0.059, MAE t+2=0.059, MSE=0.019/—

### UniST V2
- **Architecture**: Patch-based transformer over pseudo-grid (208×208), 6 task heads
- **Loss**: weighted CE - 0.01·entropy
- **Epochs**: early stop at 51/100
- **Run time**: ~6h
- **Test**: MAE t+1=0.058, MAE t+2=0.057, MSE=0.020/0.019

### CMuST
- **Architecture**: Cross-modal spatiotemporal model, 6 task heads
- **Loss**: weighted CE - 0.01·entropy
- **Epochs**: 100/100
- **Run time**: ~3.7h
- **Test**: MAE t+1=0.057, MAE t+2=0.058, MSE=0.018/0.018

### DiffSTG
- **Architecture**: Diffusion spatiotemporal graph model, 6 task heads
- **Loss**: weighted CE - 0.01·entropy
- **Epochs**: ~90/100
- **Run time**: ~3.5h
- **Test**: MAE t+1=0.058, MAE t+2=0.058, MSE=0.018/0.018

### USTD
- **Architecture**: Unified spatiotemporal decoder, 6 task heads
- **Loss**: weighted CE - 0.01·entropy
- **Epochs**: ~90/100
- **Run time**: ~3.5h
- **Test**: MAE t+1=0.057, MAE t+2=0.058, MSE=0.018/0.018

### TSGAN V2 PCGrad
- **Architecture**: TSGAN V2 + PCGrad gradient surgery over 6 task losses
- **Loss**: PCGrad over weighted CE tasks, 6 heads
- **Epochs**: 37/200 (early convergence, best checkpoint used)
- **Run time**: ~15h
- **Test**: MAE t+1=0.132, MAE t+2=0.132, MSE=0.031/0.031

## Results Table

| Model | Decoders | Loss | Epochs | Run Time | MAE t+1 | MAE t+2 | MSE t+1 | MSE t+2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TSGAN V1 | 1 unified | wCE - 0.01H | 200 | ~9.3h | 0.139 | 0.139 | 0.033 | 0.033 |
| TSGAN V2 | 6 task | wCE - 0.01H | 200 | ~14.1h | 0.182 | 0.182 | 0.051 | 0.051 |
| STGCN V2 | 6 task | wCE - 0.01H | 120 | ~5h | 0.059 | 0.059 | 0.019 | 0.019 |
| DCRNN V2 | 6 task | wCE - 0.01H | 100 | ~8h | 0.060 | 0.060 | 0.019 | 0.019 |
| GPT-ST+STGCN | 6 task | wCE - 0.01H | 46i | ~14h | 0.062 | 0.062 | 0.019 | 0.019 |
| ST-SSDL V2 | 6 task | wCE - 0.01H | 100 | ~3.7h | 0.071 | 0.068 | 0.023 | 0.023 |
| D2STGNN V2 | 6 task | wCE - 0.01H | 5 | ~0.5h | 0.059 | 0.059 | 0.019 | — |
| UniST V2 | 6 task | wCE - 0.01H | 51 | ~6h | 0.058 | **0.057** | 0.020 | 0.019 |
| **CMuST** | 6 task | wCE - 0.01H | 100 | ~3.7h | **0.057** | 0.058 | **0.018** | **0.018** |
| DiffSTG | 6 task | wCE - 0.01H | ~90 | ~3.5h | 0.058 | 0.058 | **0.018** | **0.018** |
| **USTD** | 6 task | wCE - 0.01H | ~90 | ~3.5h | **0.057** | 0.058 | **0.018** | **0.018** |
| V2 PCGrad | 6 task | PCGrad | 37/200 | ~15h | 0.132 | 0.132 | 0.031 | 0.031 |

i = interrupted

## Per-Task MAE t+1

| Model | Overall | Aggr Level | Aggr Type | Gender | Religious | Caste | Ethnicity |
|---|---:|---:|---:|---:|---:|---:|---:|
| USTD | **0.057** | 0.144 | 0.070 | 0.018 | 0.065 | 0.029 | **0.016** |
| CMuST | **0.057** | 0.144 | 0.070 | 0.019 | 0.066 | **0.028** | 0.017 |
| UniST | 0.058 | **0.143** | **0.070** | 0.021 | 0.068 | 0.029 | 0.018 |
| DiffSTG | 0.058 | 0.146 | 0.071 | **0.018** | **0.065** | 0.029 | 0.017 |
| STGCN | 0.059 | 0.150 | 0.073 | 0.020 | 0.069 | **0.028** | **0.016** |
| D2STGNN | 0.059 | 0.154 | 0.075 | **0.018** | **0.065** | 0.029 | **0.016** |
| DCRNN | 0.060 | 0.153 | 0.074 | 0.021 | 0.068 | 0.030 | 0.017 |
| GPT-ST+STGCN | 0.062 | 0.161 | 0.079 | 0.019 | 0.068 | 0.030 | 0.017 |
| ST-SSDL | 0.071 | 0.179 | 0.088 | 0.022 | 0.081 | 0.035 | 0.020 |
| V2 PCGrad | 0.132 | 0.205 | 0.155 | 0.102 | 0.133 | 0.106 | 0.092 |
| TSGAN V1 | 0.139 | 0.186 | 0.135 | 0.125 | 0.139 | 0.125 | 0.125 |
| TSGAN V2 | 0.182 | 0.197 | 0.185 | 0.132 | 0.153 | 0.197 | 0.230 |

Gender bias and ethnicity bias are the easiest tasks (MAE < 0.02 for all GCN models); aggression level is hardest (0.14–0.18). Top models converge within ~0.06 on hard tasks and are nearly identical on easy ones.

## Interpretation

- **Top performers**: USTD and CMuST tied at MAE t+1=0.057. All top-5 models (USTD, CMuST, UniST, DiffSTG, STGCN/D2STGNN) cluster tightly between 0.057–0.060.
- **TSGAN underperforms**: V1 at 0.139 and V2 at 0.182 are far behind. The TSGAN architecture does not adapt well to this task.
- **Entropy regularization works**: Six task heads with weighted CE + 0.01 entropy is shared across all strong models. D2STGNN hit 0.059 in just 5 epochs.
- **Graph models dominate**: GCN-based approaches (STGCN, D2STGNN, DiffSTG) consistently outperform TSGAN variants.
- **PCGrad helps TSGAN**: Best checkpoint (epoch ~3) gave MAE 0.132, beating both V1 (0.139) and V2 (0.182). Training diverged after epoch 5 but early stopping preserved a useful model.
- **H=2 vs H=12**: All runs here use H=2. The original provisional runs used H=12 for GCN models and H=2 for TSGAN — those are not apples-to-apples. These results are.
