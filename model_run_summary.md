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

## GPT-ST

Source being replicated:

```text
Paper: GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks
Authors: Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang
Venue/year: NeurIPS 2023
Local paper: papers/2311.04245v1.pdf
Reference repo: https://github.com/HKUDS/GPT-ST.git
Local clone: GPT-ST
Inspected commit: b50c64d347ab9d565189523bf621ecb4e4a1ab8c
Pretrain output: outputs_gptst/pretrain_20d
Downstream output: outputs_gptst/stgcn_20d
```

### Similarities With GPT-ST Paper/Code

- Preserved GPT-ST as a pretraining framework, not as a standalone forecaster.
- Preserved the paper's two-stage workflow:
  - masked autoencoder pretraining
  - frozen pretrained encoder used during downstream forecasting
- Preserved random masking before the adaptive masking phase.
- Preserved adaptive cluster-aware masking after `change_epoch`.
- Preserved masked reconstruction MAE during pretraining.
- Preserved KL cluster guidance loss during adaptive masking.
- Preserved the customized temporal encoder with time-conditioned and node-specific parameters.
- Preserved the hierarchical spatial encoder:
  - hypergraph capsule clustering
  - dynamic routing
  - cross-cluster relation learning
- Preserved gated fusion between GPT-ST embeddings and raw projected input.
- Used the paper-style history length:
  - history `H=12`
- Used the paper/code hidden and hypergraph settings where feasible:
  - hidden dim `64`
  - embed dim `16`
  - spatial embed dim `4`
  - `HS=10`
  - `HT=16`
  - `HT_Tem=8`
  - `num_route=2`
  - mask ratio `0.25`

### Differences From GPT-ST Paper/Code

- The original task predicts traffic flow/speed/demand. Our task predicts the `20`-dim social-behavior profile.
- The original data loaders expect traffic `.npz` files. Our loader reads:
  - `profiles_exact.npy`
  - `edge_index.npy`
  - `edge_attr_static.npy`
  - `mask_next1.npy`
  - `mask_next2.npy`
- The original traffic time features were replaced with deterministic daily/coarse time features derived from the 240 profile windows.
- The original repo baselines are not designed for the full `40,729` node graph, so the downstream stage uses our sparse STGCN backend.
- The original code contained hard-coded `cuda:0` tensor placement; the Indic port uses device-aware tensor placement.
- An unused multi-GB guide tensor allocation was removed because it does not affect the forward computation and is too costly on the full cohort.
- The output target was changed from single-channel traffic values to behavior dimensions `5:25`.

### Outcome

- GPT-ST pretraining completed `50` epochs.
- GPT-ST + sparse STGCN downstream training completed `50` epochs and early-stopped.
- Test MAE was `0.119965 / 0.123982` for `t+1 / t+2`.
- Test MSE was `0.060330 / 0.063691` for `t+1 / t+2`.
- GPT-ST strongly improved the sparse STGCN baseline.
- GPT-ST + STGCN is worse than DCRNN on MAE but better than DCRNN on MSE.

## ST-SSDL

Source being replicated:

```text
Paper: How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning
Authors: Haotian Gao, Zheng Dong, Jiawei Yong, Shintaro Fukushima, Kenjiro Taura, Renhe Jiang
Venue/year: NeurIPS 2025
Local paper: papers/2510.04908v1.pdf
Reference repo: https://github.com/Jimmy-7664/ST-SSDL
Local clone: ST-SSDL
Inspected commit: a98331d
Run output: outputs_stssdl/v2_20d_sparse
```

### Similarities With ST-SSDL Paper/Code

- Preserved current-window and historical-anchor modeling.
- Preserved the self-supervised deviation learning objective.
- Preserved the learned prototype bank.
- Preserved query-to-prototype attention.
- Preserved contrastive prototype learning.
- Preserved current-vs-anchor deviation loss.
- Preserved recurrent graph encoder/decoder structure.
- Used paper-style history length:
  - history `H=12`
- Used the same strict full cohort:
  - `40,729` users
  - `605,727` edges

### Differences From ST-SSDL Paper/Code

- The original code predicts single-channel traffic series. Our target is the `20`-dim behavior profile.
- The original code builds dense static and adaptive `N x N` graph supports. On our graph, one dense float32 support is about `6.6 GB` before gradients and recurrent intermediates.
- Dense adaptive graph construction was replaced with sparse adaptive edge weighting over observed social edges.
- The traffic weekly historical average was adapted to same-day-of-week profile anchors.
- 5-minute time-of-day embeddings were replaced with daily-period embeddings.
- The first full run used reduced width for the full 40k-user graph:
  - hidden dim `32`
  - prototype dim `32`
  - input embedding dim `32`
  - node embedding dim `16`
  - adaptive embedding dim `16`

### Outcome

- ST-SSDL completed and stopped after `39` epochs.
- Best validation was epoch `9`.
- Test MAE was `0.118757 / 0.122973` for `t+1 / t+2`.
- Test MSE was `0.083344 / 0.088382` for `t+1 / t+2`.

## D2STGNN

Source being replicated:

```text
Paper: Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting
Authors: Zezhi Shao, Zhao Zhang, Wei Wei, Fei Wang, Yongjun Xu, Xin Cao, Christian S. Jensen
Venue/year: PVLDB 2022
Local paper: papers/2206.09112v4.pdf
Reference repo: https://github.com/zezhishao/D2STGNN
Local clone: D2STGNN
Inspected commit: 82c2d38
Run output: outputs_d2stgnn/v2_20d_sparse
```

### Similarities With D2STGNN Paper/Code

- Preserved the Decoupled Spatial-Temporal Framework.
- Preserved diffusion-first and inherent-second branch ordering.
- Preserved estimation gate with node and time embeddings.
- Preserved residual decomposition after diffusion and inherent blocks.
- Preserved dynamic graph learning from current hidden states.
- Preserved separate diffusion and inherent forecast hidden states before final regression.
- Preserved localized temporal diffusion with:
  - temporal kernel `Kt=3`
  - spatial order `Ks=2`
- Used paper-style history length:
  - history `H=12`

### Differences From D2STGNN Paper/Code

- The original code predicts single-channel traffic values. Our target is the `20`-dim behavior profile.
- The original implementation creates dense static adaptive and dynamic `N x N` graph tensors:
  - `Softmax(ReLU(E_d E_u^T))`
  - `torch.bmm(Q, K^T)`
- Dense all-pairs graphs are infeasible for the full `40,729`-user graph, so the Indic port uses sparse message passing over observed social edges.
- The dynamic graph learner was adapted to compute edge weights only for existing edges.
- The original traffic loader was replaced with the common `data/tsgan_inputs` loader.
- 5-minute traffic time embeddings were replaced with daily-period embeddings.
- The first full run used reduced width:
  - hidden dim `32`
  - node dim `16`
  - forecast dim `64`
  - layers `2`

### Outcome

- D2STGNN queued correctly after ST-SSDL and completed.
- Training early-stopped at epoch `99`.
- Best validation was epoch `69`.
- Test MAE was `0.103274 / 0.111521` for `t+1 / t+2`.
- Test MSE was `0.080714 / 0.089098` for `t+1 / t+2`.

## UniST

Source being replicated:

```text
Paper: UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction
Authors: Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, Yong Li
Venue/year: KDD 2024
Local paper: papers/2402.11838
Reference repo: https://github.com/tsinghua-fib-lab/UniST
Local clone: UniST
Inspected commit: 6ee69db00d8f67e550f9bc3c06f004f9f7885703
Run output: outputs_unist/v2_20d_pseudogrid
```

### Similarities With UniST Paper/Code

- Preserved UniST's patch-based spatial tokenization.
- Preserved temporal and spatial positional conditioning.
- Preserved encoder/decoder transformer structure.
- Preserved future mask-token decoding.
- Preserved prompt-style future conditioning.
- Used the same `H=12`, `F=2` comparison setup as STGCN, DCRNN, GPT-ST, ST-SSDL, and D2STGNN.

### Differences From UniST Paper/Code

- The official UniST implementation is grid-first. Our data is a user graph, not an urban grid.
- To keep minimal code changes and preserve the released UniST mechanics, users were packed into a deterministic padded pseudo-grid:
  - `40,729` users
  - `208 x 208` padded cells
  - `patch_size=16`
  - `169` spatial patches per time step
- The social graph edges are not used by this first UniST run.
- The original scalar urban target was replaced with behavior dimensions `5:25`, giving the same `20`-dim target used by the other baselines.
- Loss and metrics use the same `mask_next1.npy` and `mask_next2.npy` masks as the other runs.

### Outcome

- UniST completed and early-stopped at epoch `38`.
- Best validation was epoch `13`.
- Test MAE was `0.147103 / 0.148617` for `t+1 / t+2`.
- Test MSE was `0.085215 / 0.086991` for `t+1 / t+2`.
- UniST improves substantially over sparse STGCN on MAE, but it does not beat DCRNN or D2STGNN on MAE.

## Baseline Code Availability

| Requested baseline | Paper / method identified | Code availability found | Our status |
|---|---|---|---|
| `https://github.com/VeritasYin/STGCN_IJCAI-18` | STGCN | Available | Completed |
| `https://github.com/liyaguang/dcrnn` | DCRNN | Available | Completed |
| `https://arxiv.org/pdf/2402.11838` | UniST | Available: `https://github.com/tsinghua-fib-lab/UniST` | Completed |
| `https://arxiv.org/pdf/2311.04245v1` | GPT-ST | Available: `https://github.com/HKUDS/GPT-ST` | Completed |
| `https://arxiv.org/abs/2510.04908` | ST-SSDL | Available: `https://github.com/Jimmy-7664/ST-SSDL` | Completed |
| `https://arxiv.org/pdf/2206.09112` | D2STGNN | Available: `https://github.com/GestaltCogTeam/D2STGNN` | Completed |
| NeurIPS 2020 hash `3fe78a8acf5fda99de95303940a2420c` | PCGrad | Available: `https://github.com/tianheyu927/PCGrad` | Implemented; interrupted best checkpoint tested |
| `https://arxiv.org/pdf/2504.00721` | MinGRE / zero-inflated adversarial spatiotemporal graph learning | No clear official code repo found in search | Not run |

## Results Table

| Run | Paper/code matched | History | Decoder / Heads | Loss Used | Status | Test MAE t+1 | Test MAE t+2 | Test MSE t+1 | Test MSE t+2 |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| Fixed TSGAN V1 | TSGAN | 2 | single 20-d decoder | MAE train, MSE tracked | completed | 0.236584 | 0.227865 | 0.163480 | 0.149031 |
| Fixed TSGAN V2 | TSGAN | 2 | six task decoders | MAE train, MSE tracked | completed | 0.234852 | 0.227157 | 0.173603 | 0.165608 |
| Fixed TSGAN V2 + PCGrad | TSGAN + PCGrad | 2 | six task decoders | PCGrad over six task MAE losses | interrupted, best checkpoint tested | 0.246170 | 0.239650 | 0.184860 | 0.178771 |
| Sparse STGCN 20-d | STGCN | 12 | six task heads | MSE/L2 | completed, early stop epoch 68 | 0.240788 | 0.242641 | 0.118587 | 0.120166 |
| Sparse DCRNN 20-d | DCRNN | 12 | 20-d DCGRU seq2seq output | masked MAE | completed, epoch 100 | 0.105701 | **0.110732** | 0.082055 | 0.087803 |
| GPT-ST + Sparse STGCN 20-d | GPT-ST + STGCN | 12 | frozen GPT-ST encoder + gated fusion + six STGCN task heads | GPT-ST masked MAE pretrain, downstream MSE/L2 | completed, early stop epoch 50 | 0.119965 | 0.123982 | **0.060330** | **0.063691** |
| Sparse ST-SSDL 20-d | ST-SSDL | 12 | sparse adaptive recurrent decoder, 20-d output | forecast MAE + contrastive + deviation losses | completed, stopped epoch 39 | 0.118757 | 0.122973 | 0.083344 | 0.088382 |
| Sparse D2STGNN 20-d | D2STGNN | 12 | decoupled diffusion/inherent sparse branches, 20-d output | masked MAE | completed, early stop epoch 99 | **0.103274** | 0.111521 | 0.080714 | 0.089098 |
| UniST 20-d pseudo-grid | UniST | 12 | patch transformer over padded user pseudo-grid, 20-d output | masked MSE | completed, early stop epoch 38 | 0.147103 | 0.148617 | 0.085215 | 0.086991 |

## Interpretation

- The model sections above should be read as replication notes: each one compares our implementation to that model's own paper/code.
- Best test MAE t+1 so far: sparse D2STGNN 20-d.
- Best test MAE t+2 so far: sparse DCRNN 20-d by a narrow margin.
- Best test MSE so far: GPT-ST + sparse STGCN 20-d.
- DCRNN remains the most consistently strong diffusion baseline and is a good fit because the original method explicitly models directed diffusion.
- D2STGNN is now the strongest t+1 MAE model and is close to DCRNN at t+2 MAE.
- UniST is a valid completed foundation-model baseline; it improves over sparse STGCN but is weaker than DCRNN, D2STGNN, GPT-ST, and ST-SSDL on MAE.
- ST-SSDL is competitive with GPT-ST on MAE but does not improve on DCRNN or D2STGNN.
- GPT-ST + STGCN substantially improves the plain sparse STGCN baseline and is the strongest MSE model so far.
- STGCN is a valid paper-backed baseline, but its original dense graph implementation had to be replaced with a sparse graph layer to run on the full cohort.
- ST-SSDL and D2STGNN required the same dense-to-sparse graph adaptation because their original dynamic graph modules materialize all-pairs `N x N` tensors.
- UniST did not require dense graph adaptation because it is grid/patch based; the main adaptation was packing graph users into a pseudo-grid.
- PCGrad was technically valid but did not improve TSGAN V2 in the evaluated run.
- The only requested baseline where I did not find a clear official code repo is `arXiv:2504.00721` / MinGRE.
- Fairness caveat: TSGAN and PCGrad used `H=2`, while STGCN, DCRNN, GPT-ST, ST-SSDL, D2STGNN, and UniST used `H=12`.
- Next fair ablation should either rerun TSGAN with `H=12` or rerun the later baselines with `H=2`.
