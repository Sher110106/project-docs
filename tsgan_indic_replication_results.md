# TSGAN Indic: Faithful Replication of TSGAN for 20-Dimensional Behavioral Forecasting

This report documents the successful replication of the **Temporal Social Graph Attention Network (TSGAN)** architecture from the paper *"TSGAN: Temporal Social Graph Attention Network for Aggressive Behaviour Forecasting"*, adapted for Indic social media data with 40,729 users and a 20-dimensional behavioral profile target.

## Architectural Fidelity

The TSGAN Indic implementation faithfully reproduces the original paper's core architecture:

| Component | Original TSGAN | TSGAN Indic | Status |
|-----------|---------------|-------------|--------|
| **DenseLayer** | Reshape→Linear→BN→ReLU→Reshape | Identical | same |
| **GraphNodeContextEncoder** | 2-layer DenseLayer over static node embeddings | Identical | same |
| **SIA (Social Influence Attention)** | Chunked per-node attention with edge-weighted scores | `permute→unsqueeze→multiply→sum(dim=3)` pattern preserved. Gradient checkpointing added for 40K-node scalability (FP-identical) | same |
| **TDA (Temporal Dynamics Attention)** | Multi-head causal attention + exponential time-decay kernel | Identical multi-head chunking, decay kernel, `tril` mask | same |
| **FAG (Fusion Attention Gate)** | DenseLayer spatial/temporal projections + sigmoid gating + 2-layer fusion | Identical | same |
| **STAM** | SIA + TDA + FAG with residual `X + H` | Shared stack across encode/decode phases | same |
| **CTA (Cross-Temporal Attention)** | Multi-head attention between history and future embeddings | Identical `split→permute→matmul→softmax→merge` | same |
| **Barlow Twins GCL** | GCNConv + edge dropping (50%) + feature masking (10%) + cross-correlation loss | Identical augmentations and loss, 4000 epochs, 256-dim embeddings | same |
| **Temporal window** | H=2 history, F=2 forecast | Identical | same |

### Scalability adaptations (no functional change)
- **Gradient checkpointing** in SIA: `torch.utils.checkpoint.checkpoint()` wraps per-chunk attention to avoid storing activations for all 40,729 nodes
- **Edge encoder**: 3D edge features (SPD + NPD + topic similarity) encoded to scalar attention weight; original's 1D edge feature maps trivially

### Task adaptations (required for 20d forecasting)
- Input: 25-dim (5 static context + 20 dynamic behavior) vs original 6-dim
- Output: 20-dim behavioral profile vs original 1-dim aggression score

## Model Variants

### V1: Single Decoder
A single shared MLP decoder maps the latent representation to the full 20-dimensional output. Trained with masked MAE loss over both horizons.

### V2: Six Task-Specific Decoders
Six independent MLP decoders specialized per behavioral category:
- `aggression_level` (3 dims), `aggression_type` (5 dims)
- `gender_bias` (3 dims), `religious_bias` (3 dims)
- `caste_bias` (3 dims), `ethnicity_bias` (3 dims)

Outputs concatenated to 20-dim. Trained with per-task masked MAE.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Cohort | 40,729 nodes, 605,727 edges |
| Temporal window | 240 days, chronological split |
| Split | 165 train / 35 val / 37 test |
| Input dim | 25 (5 static + 20 dynamic) |
| Output dim | 20 |
| Hidden dim (R) | 64 |
| Attention heads (AH) | 8 |
| Chunk size | 1000 |
| Optimizer | Adam, lr=1e-3, wd=0.18 |
| Scheduler | CosineAnnealingLR |
| Batch size | 2 |
| Early stopping | patience=30 |

## Test Set Results

### V1 — Single 20-dim Decoder (early stop at epoch 95)

| Horizon | MAE | MSE |
|---------|-----|-----|
| **t+1** | 0.2366 | 0.1635 |
| **t+2** | 0.2279 | 0.1490 |

### V2 — Six Task-Specific Decoders (early stop at epoch 72)

| Horizon | MAE | MSE |
|---------|-----|-----|
| **t+1** | 0.2349 | 0.1736 |
| **t+2** | 0.2272 | 0.1656 |

### V2 Per-Task Breakdown

| Task | Dims | MAE t+1 | MAE t+2 |
|------|------|---------|---------|
| aggression_level | 3 | 0.317 | 0.318 |
| aggression_type | 5 | 0.176 | 0.167 |
| gender_bias | 3 | 0.197 | 0.190 |
| religious_bias | 3 | 0.275 | 0.271 |
| caste_bias | 3 | 0.216 | 0.205 |
| ethnicity_bias | 3 | 0.203 | 0.197 |

### Interpretation
- **aggression_type** is the easiest to forecast (MAE ~0.17), suggesting strong temporal diffusion patterns
- **aggression_level** is the hardest (MAE ~0.32), consistent with high day-to-day volatility
- **religious_bias** shows elevated error (~0.27), indicating complex social dynamics
- V1 and V2 converge to similar overall MAE (~0.23), with V2 providing useful per-task diagnostic granularity
