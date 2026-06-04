# Model Run Summary

This file summarizes each run against the actual paper/codebase it is trying to replicate. The model sections are not framed as comparisons against our earlier baselines; those comparisons are kept only in the results table and interpretation.

## TSGAN

Source being replicated:

```text
Reference code: TSGAN/models/TSGAN_model.py
Local base used for edits: TSGAN_indic
Run output: outputs_tsgan_fixed/
```

### Similarities With TSGAN Paper/Code

- Restored the original-style FAG fusion module instead of the simplified IAG fusion used in the earlier local variant.
- Restored shared STAM layers between encoder and decoder, matching the reference code structure.
- Restored the original multi-channel SIA edge-weight multiplication pattern.
- Preserved the main TSGAN components used in the reference implementation:
  - DenseLayer
  - GraphNodeContextEncoder
  - TDA
  - CTA
  - STAM
  - SIA-style graph attention
- Preserved the original-style short temporal window setup used in this replication run:
  - history `H=2`
  - forecast `F=2`
- Kept the strict real-profile cohort and full social graph for the Indic task:
  - `40,729` users
  - `605,727` edges

### Differences From TSGAN Paper/Code

- The supervised output was adapted to our task: behavior dimensions `5:25`, a `20`-dim vector.
- V1 used one unified `20`-dim decoder.
- V2 used six task-specific decoders for:
  - aggression level
  - aggression type
  - gender bias
  - religious bias
  - caste bias
  - ethnicity bias
- The edge encoder output stayed scalar because the original full `R`-dim edge encoding was too memory-heavy on the `605k`-edge graph.
- Added explicit MSE tracking beside MAE for reporting; this is evaluation instrumentation, not a TSGAN architecture change.
- The dataset, labels, and graph are our Indic social-behavior artifacts, not the original paper dataset.

### Outcome

- Fixed TSGAN V1 completed with test MAE `0.236584 / 0.227865` for `t+1 / t+2`.
- Fixed TSGAN V2 completed with test MAE `0.234852 / 0.227157` for `t+1 / t+2`.
- V2 was slightly better than V1 on MAE, while V1 had lower MSE.

## PCGrad TSGAN

Source method being added:

```text
Method: PCGrad, gradient surgery for multi-task learning
Base model: fixed TSGAN V2
Run: fixed TSGAN V2 + PCGrad
```

### Similarities With PCGrad Method

- Treated the six behavior groups as separate task losses.
- Computed gradients per task before the optimizer update.
- Projected conflicting gradients when task gradients had negative dot products.
- Applied the merged projected gradient through the base optimizer.

### Differences From PCGrad Method/Original TSGAN

- PCGrad is an optimizer-level multi-task method, not part of the original TSGAN architecture.
- It was applied on top of fixed TSGAN V2 because V2 already has six task-specific decoders.
- V1 support exists by slicing one `20`-dim output into six task losses, but the evaluated PCGrad run was V2 only.
- The run was interrupted and evaluated from the best saved checkpoint rather than a full clean completion.

### Outcome

- PCGrad functioned technically.
- It did not improve held-out test MAE over fixed TSGAN V2.
- Test MAE was `0.246170 / 0.239650` for `t+1 / t+2`.

## STGCN

Source being replicated:

```text
Paper: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
Authors: Bing Yu, Haoteng Yin, Zhanxing Zhu
Venue/year: IJCAI 2018
Local paper: papers/0505.pdf
Reference repo: https://github.com/VeritasYin/STGCN_IJCAI-18.git
Tyrone clone: /home/deep/projects/indic/external_repos/STGCN_IJCAI-18
Inspected commit: 747007ab01d56762c1a6bad89b255700cedf204d
Run output: outputs_stgcn/v2_20d_sparse
```

### Similarities With STGCN Paper/Code

- Preserved the core ST-Conv block order:
  - temporal gated convolution
  - graph convolution
  - temporal gated convolution
- Preserved temporal GLU-style gating.
- Preserved Chebyshev graph convolution as the graph operation concept.
- Used paper-style temporal and graph kernel settings:
  - temporal kernel `Kt=3`
  - graph Chebyshev order `Ks=3`
- Used the paper-style history length:
  - history `H=12`
- Used the paper-style STGCN channel pattern:
  - temporal `64`
  - spatial bottleneck `16`
  - temporal `64`
- Kept RMSProp as the default optimizer.
- Kept MSE/L2-style training loss.

### Differences From STGCN Paper/Code

- The original repo is TensorFlow 1.x; our runnable implementation is standalone PyTorch.
- The original code expects a dense weighted traffic adjacency matrix. Our graph is sparse and much larger:
  - `40,729` nodes
  - `605,727` edges
- The dense Chebyshev kernel in the original code is not feasible for the full Indic graph, so the graph layer was implemented with sparse PyG `ChebConv`.
- The original task predicts traffic speed at road sensors. Our task predicts a `20`-dim user behavior profile.
- The original output is effectively single-channel traffic forecasting. Our output uses six task heads covering the full `20` dimensions.
- The original traffic data loader was replaced with a loader for:
  - `profiles_exact.npy`
  - `edge_index.npy`
  - `edge_attr_static.npy`
  - `mask_next1.npy`
  - `mask_next2.npy`
- Our forecast horizon for this run was `F=2`, not the original traffic benchmark's longer recursive horizon.

### Outcome

- STGCN trained cleanly and early-stopped at epoch `68`.
- Best validation was epoch `43`.
- Test MAE was `0.240788 / 0.242641` for `t+1 / t+2`.
- Test MSE was `0.118587 / 0.120166` for `t+1 / t+2`.

## DCRNN

Source being replicated:

```text
Paper: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
Authors: Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu
Venue/year: ICLR 2018
Local paper: papers/1707.01926v3.pdf
Reference repo: https://github.com/liyaguang/DCRNN.git
Tyrone clone: /home/deep/projects/indic/external_repos/DCRNN
Inspected commit: 602afd9d767d3aa1c9b3eac51710d6aeee12c227
Run output: outputs_dcrnn/v2_20d_dual_random_walk
```

### Similarities With DCRNN Paper/Code

- Preserved directed diffusion over the graph.
- Preserved dual random-walk supports.
- Used max diffusion step `2`.
- Implemented DCGRU reset, update, and candidate gates.
- Used encoder-decoder sequence modeling.
- Used a zero decoder input at the first decoder step.
- Used scheduled sampling with inverse-sigmoid decay.
- Used two recurrent layers.
- Used hidden size `64`.
- Used Adam with:
  - base learning rate `0.01`
  - epsilon `1e-3`
  - gradient clipping at `5`
  - learning-rate drops at epochs `20,30,40,50`
- Used masked MAE as the training loss.
- Used z-score normalization from training data.

### Differences From DCRNN Paper/Code

- The official repo is TensorFlow 1.x and uses `tensorflow.contrib`; our runnable implementation is standalone PyTorch.
- The original task predicts traffic speed. Our task predicts the `20`-dim social-behavior profile.
- The original graph is a traffic sensor graph. Our graph is the directed social graph:
  - `40,729` users
  - `605,727` directed edges
- The original data format is traffic HDF5/sensor data. Our loader reads numpy artifacts from `data/tsgan_inputs`.
- The original target is single-channel speed. Our target is behavior dimensions `5:25`.
- The first full run used:
  - history `H=12`
  - forecast `F=2`
  - input dim `25`
  - output dim `20`

### Outcome

- DCRNN completed all `100` epochs cleanly.
- Best validation was epoch `100`.
- Test MAE was `0.105701 / 0.110732` for `t+1 / t+2`.
- Test MSE was `0.082055 / 0.087803` for `t+1 / t+2`.
- This is currently the strongest completed paper-backed baseline on both MAE and MSE.

## Results Table

| Run | Paper/code matched | History | Decoder / Heads | Loss Used | Status | Test MAE t+1 | Test MAE t+2 | Test MSE t+1 | Test MSE t+2 |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| Fixed TSGAN V1 | TSGAN | 2 | single 20-d decoder | MAE train, MSE tracked | completed | 0.236584 | 0.227865 | 0.163480 | 0.149031 |
| Fixed TSGAN V2 | TSGAN | 2 | six task decoders | MAE train, MSE tracked | completed | 0.234852 | 0.227157 | 0.173603 | 0.165608 |
| Fixed TSGAN V2 + PCGrad | TSGAN + PCGrad | 2 | six task decoders | PCGrad over six task MAE losses | interrupted, best checkpoint tested | 0.246170 | 0.239650 | 0.184860 | 0.178771 |
| Sparse STGCN 20-d | STGCN | 12 | six task heads | MSE/L2 | completed, early stop epoch 68 | 0.240788 | 0.242641 | 0.118587 | 0.120166 |
| Sparse DCRNN 20-d | DCRNN | 12 | 20-d DCGRU seq2seq output | masked MAE | completed, epoch 100 | **0.105701** | **0.110732** | **0.082055** | **0.087803** |

## Interpretation

- The model sections above should be read as replication notes: each one compares our implementation to that model's own paper/code.
- Best completed run so far: sparse DCRNN 20-d.
- DCRNN is strongest on both MAE and MSE, and it is a good fit because the original method explicitly models directed diffusion.
- STGCN is a valid paper-backed baseline, but its original dense graph implementation had to be replaced with a sparse graph layer to run on the full cohort.
- PCGrad was technically valid but did not improve TSGAN V2 in the evaluated run.
- Fairness caveat: TSGAN and PCGrad used `H=2`, while STGCN and DCRNN used `H=12`.
- Next fair ablation should either rerun TSGAN with `H=12` or rerun STGCN/DCRNN with `H=2`.
