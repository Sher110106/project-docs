# Model Run Summary

## Fixed TSGAN Baseline

### Kept Same

- Cohort: strict real-profile cohort, `40,729` users.
- Graph: same social graph, `605,727` edges.
- Inputs: `profiles_exact.npy`, `25` input dims.
- Targets: behavior dimensions `5:25`, `20` output dims.
- Forecast target: `t+1` and `t+2`.
- Split: chronological train/validation/test split.
- V1/V2 distinction:
  - V1: single `20`-dim decoder.
  - V2: six task-specific decoders.

### Changed

- Corrected TSGAN architectural deviations to better match the original TSGAN implementation.
- Replaced simplified fusion gate with original-style FAG.
- Shared STAM layers between encode/decode instead of separate stacks.
- Restored original multi-channel SIA edge-weight pattern.
- Kept scalar edge encoder output because full `R`-dim edge encoding was too memory-heavy on `605k` edges.
- Added explicit MSE tracking beside MAE.

### Main Outcome

- Fixed V2 was the strongest tested model on test MAE.
- V1 was close to V2 on MAE but had lower MSE.

## PCGrad TSGAN

### Kept Same

- Same strict `40,729` user cohort.
- Same `25`-dim input and `20`-dim target schema.
- Same chronological split.
- Same fixed TSGAN architecture as base.
- Same six task ranges:
  - aggression level
  - aggression type
  - gender bias
  - religious bias
  - caste bias
  - ethnicity bias

### Changed

- Added PCGrad optimizer wrapper.
- For V2, each task decoder produced a separate task loss.
- PCGrad projected conflicting task gradients before optimizer step.
- For V1, implementation supports slicing the single `20`-dim output into six task losses.
- The launched/evaluated PCGrad run was V2 only; it was interrupted and evaluated from best saved checkpoint.

### Main Outcome

- PCGrad functioned technically.
- PCGrad did not improve held-out test MAE over vanilla fixed V2.
- It added optimizer complexity without a clear gain in this run.

## STGCN

### Kept Same

- Same strict profile-backed user population concept.
- Same target: behavior dimensions `5:25`, `20` dims.
- Same t+1/t+2 forecast target.
- Same full social graph scale: `40,729` nodes, `605,727` edges.
- Same output reporting style: MAE/MSE for `t+1` and `t+2`.

### Changed

- Built standalone `STGCN_indic`; no TSGAN imports.
- Cloned and inspected `VeritasYin/STGCN_IJCAI-18`.
- Did not run original TensorFlow code because it requires dense graph kernels.
- Replaced dense TensorFlow Chebyshev kernel with sparse PyG `ChebConv`.
- Used paper-style ST-Conv block:
  - temporal GLU
  - graph convolution
  - temporal GLU
- Used paper-style settings:
  - `H=12`
  - `Kt=3`
  - `Ks=3`
  - RMSProp
  - MSE/L2 training loss
  - temporal/spatial/temporal bottleneck pattern `64/16/64`
- Changed output from single traffic-speed value to six heads producing the full `20`-dim profile.
- Added feature-file support so future feature extraction can be tested without changing the target file.

### Main Outcome

- STGCN trained cleanly and early-stopped at epoch `68`.
- Best validation was epoch `43`.
- Test MAE was worse than fixed TSGAN V2, especially at `t+2`.
- Test MSE was lower than fixed TSGAN V2, but this is not directly apples-to-apples because STGCN used `H=12` while fixed TSGAN used `H=2`.

## DCRNN

### Kept Same

- Same strict `40,729` user cohort.
- Same social graph scale: `40,729` nodes and `605,727` directed edges.
- Same target: behavior dimensions `5:25`, full `20`-dim profile.
- Same forecast target: `t+1` and `t+2`.
- Same chronological train/validation/test split style.
- Same final reporting style: MAE/MSE for `t+1` and `t+2`, plus per-task metrics.

### Changed

- Cloned and inspected the original `liyaguang/DCRNN` implementation.
- Did not use the original TensorFlow 1.x code path because it depends on old `tensorflow.contrib` APIs.
- Built standalone `DCRNN_indic` in PyTorch, keeping the paper mechanics:
  - directed graph diffusion
  - dual random-walk supports
  - max diffusion step `2`
  - DCGRU encoder-decoder
  - scheduled sampling with inverse-sigmoid decay
  - two recurrent layers
  - hidden size `64`
  - Adam with learning-rate drops at epochs `20,30,40,50`
  - masked MAE training loss
  - z-score normalization
- Used the paper-style longer history setting `H=12`, forecasting `F=2`.
- Adapted the output from traffic speed to six behavior tasks covering the full `20`-dim profile.

### Main Outcome

- DCRNN completed all `100` epochs cleanly.
- Best validation was the final epoch, epoch `100`.
- Test metrics were the best so far across both MAE and MSE.
- This is the strongest paper-backed baseline currently tested, but it is still not a strict apples-to-apples comparison with fixed TSGAN because DCRNN used `H=12` while the fixed TSGAN runs used `H=2`.

## Results Table

| Run | History | Decoder / Heads | Loss Used | Status | Test MAE t+1 | Test MAE t+2 | Test MSE t+1 | Test MSE t+2 |
|---|---:|---|---|---|---:|---:|---:|---:|
| Fixed TSGAN V1 | 2 | single 20-d decoder | MAE train, MSE tracked | completed | 0.236584 | 0.227865 | 0.163480 | 0.149031 |
| Fixed TSGAN V2 | 2 | six task decoders | MAE train, MSE tracked | completed | **0.234852** | **0.227157** | 0.173603 | 0.165608 |
| Fixed TSGAN V2 + PCGrad | 2 | six task decoders | PCGrad over six task MAE losses | interrupted, best checkpoint tested | 0.246170 | 0.239650 | 0.184860 | 0.178771 |
| Sparse STGCN 20-d | 12 | six task heads | MSE/L2 | completed, early stop epoch 68 | 0.240788 | 0.242641 | 0.118587 | 0.120166 |
| Sparse DCRNN 20-d | 12 | six task outputs from DCGRU seq2seq | masked MAE | completed, epoch 100 | **0.105701** | **0.110732** | **0.082055** | **0.087803** |

## Interpretation

- Best test MAE: sparse DCRNN 20-d.
- Best test MSE: sparse DCRNN 20-d.
- PCGrad was technically valid but not beneficial in this run.
- STGCN is a genuine paper-backed baseline, but DCRNN is much stronger on this first run.
- DCRNN is the current best candidate for the next paper-backed comparison, especially because it models directed diffusion explicitly.
- The main caveat is comparison fairness: STGCN and DCRNN used `H=12`, while fixed TSGAN and PCGrad used `H=2`.
- Next fair comparison would rerun TSGAN with `H=12`, or rerun STGCN/DCRNN with `H=2` as ablations.
