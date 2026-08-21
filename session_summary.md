# ASSET Session Summary

Multi-day session hardening and extending the ASSET backdoor-detection research codebase: fixing real bugs in the original pipeline, generalizing it to support all poisoning methods, integrating 7 additional attacks from ATAF, tuning ASSET's detection hyperparameters across all 14 methods, and building resumable automation for all of it.

## 1. Understanding and fixing the original pipeline

Validated `train_backdoored_model.py` (BadNets-style poisoning + ResNet18 training) and `ASSET_demo.ipynb` (ASSET's active-separation detection: a frozen reference model `o_model` + a freshly-trained "detective" model that suppresses confidence everywhere except statistically-outlying batches, creating a loss gap between poisoned and clean samples that a threshold can separate).

Bugs found and fixed in `new_poi_util.py`:
- **Poison-index sampling bug**: `size=int(len(Dataset)*poi_rates)` exceeded the available pool at `poi_rates=1.0`. Fixed with `min(size, len(current_label))`.
- **`posion_image_all2all` aliasing bug**: `self.data = dataset.data` (and `.targets`) shared the underlying array instead of copying it, silently corrupting the shared test dataset and producing a "Clean ACC 0.50%" result (worse than random). Fixed with `copy.deepcopy()`.
- **`get_result()` eval-mode inconsistency**: `model.eval()` was called between the poison/clean loader loops instead of at the top of the function, causing different AUC readings for the *same* weights depending on whether the model was freshly trained or reloaded from disk. Fixed by moving `model.eval()` to the top.
- **`poi_methond` → `poi_method`**: fixed a spelling typo from the original author, everywhere it was referenced.

## 2. Generalizing to support any poisoning method

`train_backdoored_model.py` and `poi_dataset()` were generalized to take `POI_METHOD` as a parameter (env var / function arg) instead of being hardcoded, with per-method logging (`logs/train_<method>.log`) via a `_Tee` class that duplicates stdout/stderr to both console and file.

## 3. ATAF integration (7 additional attacks)

Per explicit instruction — **import from `~/ATAF/ataf-lib` read-only, never write or edit anything there** — integrated 7 poisoning generators from the ATAF library into `new_poi_util.py` via a bridge (`AtafPoisonedDataset` + `_ataf_poison()`), operating on ATAF's whole-array `PoisonGenerator` convention and adapting it to ASSET's lazy per-item `Dataset` convention:

- **wanet** (spatial warp trigger)
- **blended** (alpha-blend trigger)
- **badnets** (visible patch trigger)
- **low_frequency** (frequency-domain additive trigger)
- **ssba** (sample-specific invisible/steganographic trigger)
- **inputaware** (sample-specific soft-mask blend trigger)
- **ctrl** (clean-label, frequency-domain trigger)

Total supported methods: **14** (7 original + 7 ATAF), covering both attacks that require training a poisoned model (all 14) and no-training label-noise-only attacks (subset of the original 7).

## 4. Infrastructure and path migration

- All checkpoint/results paths migrated to `/home/HDD/ATAF/siddharth/ASSET/trained_models/` after the user moved the folder there (absolute paths used throughout to avoid relative-path fragility across different working directories).
- `train_all_poisons.sh`: checks which methods already have a trained checkpoint and only trains the missing ones.
- All result CSVs later consolidated into a `results_csv/` subfolder, with every script (`retrain_weak_poisons.py`, `asset_threshold_sweep.py`, `asset_eps_sweep.py`) and the notebook's logging cell updated to write there.

## 5. Upstream evaluation: threshold tuning

`asset_threshold_sweep.py` sweeps the training-time `THRESHOLD` (the `adjusted_outlyingness` cutoff that decides which batches get reinforced) from 1.0–2.0 in steps of 0.1, keeping only the best-AUC model per method (saved to `asset_best_models/`), with full resume support (skips any `(method, threshold)` pair already logged).

Ran across all 14 methods (`backdoor`/`noisy_all2one` were pre-seeded with dummy zero-AUC rows to skip them, since their good thresholds — 1.1 and 1.3 — were already found manually earlier). Final best thresholds:

| Method | Threshold | AUC |
|---|---|---|
| backdoor | 1.1 | 0.981 |
| low_frequency | 1.6 | 0.955 |
| blended | 1.3 | 0.956 |
| wanet | 1.8 | 0.941 |
| inputaware | 1.3 | 0.920 |
| badnets | 1.0 | 0.906 |
| noisy_all2one | 1.3 | 0.854 |
| flipping_label | 1.8 | 0.823 |
| clean_label_narcissus | 1.5 | 0.795 |
| noisy_all2all | 1.1 | 0.749 |
| ssba | 1.3 | 0.713 |
| ctrl | 1.8 | 0.694 |
| backdoor_all2all | 1.2 | 0.620 |
| noisy_label | 1.8 | 0.500 (chance level — no real separation possible for this method) |

## 6. Upstream evaluation: epsilon tuning

`get_t(data, eps)` fits a 1-component Gaussian mixture to the loss distribution and extrapolates a decision threshold `t(ε) = μ + C·√(−ln ε)` — stricter (smaller) eps raises the cutoff, trading recall for precision.

Initial directional search (`find_best_eps.py`, since removed) tried a coarse exponent search per method. This was superseded by **`asset_eps_sweep.py`**, which loads each method's saved detective-model checkpoint once and exhaustively sweeps every exponent from `1e0` down to `1e-100`, picking the eps that maximizes **F1** (chosen because it's the harmonic mean of precision/recall — it collapses toward zero if either side is neglected, so it can't be gamed by flagging everything or nothing, unlike an arithmetic mean).

Final best-F1 operating point per method (saved to `results_csv/asset_upstream_eval_summary.csv`):

| Method | Threshold | Epsilon | AUC | TP | FP | FN | F1 |
|---|---|---|---|---|---|---|---|
| Backdoor | 1.1 | 1e-94 | 0.981 | 1948 | 300 | 552 | 0.821 |
| Blended | 1.3 | 1e-40 | 0.956 | 2016 | 3035 | 484 | 0.534 |
| Low_frequency | 1.6 | 1e-53 | 0.955 | 2260 | 4182 | 240 | 0.505 |
| Wanet | 1.8 | 1e-56 | 0.941 | 1930 | 4020 | 570 | 0.457 |
| Badnets | 1.0 | 1e-29 | 0.906 | 1363 | 2353 | 1137 | 0.439 |
| Inputaware | 1.3 | 1e-39 | 0.920 | 1510 | 3283 | 990 | 0.414 |
| Noisy_all2all | 1.1 | 1e-34 | 0.749 | 665 | 882 | 1835 | 0.329 |
| Noisy_all2one | 1.3 | 1e-13 | 0.854 | 1735 | 9159 | 765 | 0.259 |
| SSBA | 1.3 | 1e-21 | 0.713 | 1060 | 8287 | 1440 | 0.179 |
| Backdoor_all2all | 1.2 | 1e-27 | 0.620 | 422 | 1319 | 2078 | 0.199 |
| Flipping_label | 1.8 | 1e-66 | 0.823 | 41 | 712 | 209 | 0.082 |
| Clean_label_narcissus | 1.5 | 1e-29 | 0.795 | 52 | 1933 | 198 | 0.047 |
| Ctrl | 1.8 | 1e-42 | 0.694 | 28 | 1876 | 222 | 0.026 |

Key insight: **AUC sets a hard ceiling that eps tuning cannot overcome.** Methods with AUC ≥ 0.90 (backdoor, blended, low_frequency, wanet, inputaware, badnets) get genuinely strong detection at their F1-optimal point. Below ~0.75 AUC (ctrl, backdoor_all2all, noisy_label), no eps choice rescues it — the fix there requires retraining the detective model, not further eps tuning.

## 7. Retraining weak-ASR poisoned models

Diagnosed why several ATAF methods trained weak backdoors: `ssba`'s trigger is sample-specific (no shared shortcut across poisoned images, so more poison rate barely helps), `low_frequency` and `blended` just needed more poison rate/epochs, `wanet` was still climbing at epoch 50.

Built two scripts to retrain and only keep genuinely-better checkpoints:
- **`retrain_weak_poisons.py`**: runs a list of `(method, epochs, poison_rate)` attempts, only overwrites `poisoned_models/<method>.pth` if the new attempt's ASR beats the best ASR seen so far (seeded from baseline + true CSV history).
- **`retrain_one_poison.py`**: same logic for a single hand-picked method/config, with its own separate results CSV to avoid colliding with a concurrently-running `retrain_weak_poisons.py`.

Results — clean accuracy / ASR before vs. after retraining:

| Method | Original ASR | Best retrained ASR | Config |
|---|---|---|---|
| low_frequency | 20.81% | **91.54%** | 50 epochs, 15% poison rate |
| blended | 73.96% | **92.79%*** | 50 epochs, 15% poison rate |
| wanet | 35.06% | **73.39%** | 50 epochs, 15% poison rate |
| noisy_all2all | 89.82% | 91.16% | 50 epochs, 10% poison rate |
| flipping_label | 85.08% | 85.38% | 50 epochs, 10% poison rate |
| ssba | 12.57% | 24.14% | 50 epochs, 20% poison rate (structurally hard to improve further — no shared trigger pattern) |

\* *blended's 92.79% result is logged in the CSV history but was never actually captured in a checkpoint file — see bugs below. The checkpoint currently on disk is the earlier 88.70% version.*

## 8. Critical bugs found in the retrain automation (and fixed)

- **Append-before-refresh self-comparison bug**: both retrain scripts wrote a completed attempt's row to the results CSV *before* re-reading the CSV to decide whether to save the checkpoint — so the freshly-written row was read back as "prior history" and compared against itself, making `asr > current_best` false even for genuine improvements. This silently prevented several good models from being saved (`ssba`'s first 24.28% attempt, `blended`'s two 92.79% attempts). Fixed by moving the refresh-and-compare step to *before* the CSV write.
- **CSV race condition**: switched both scripts from read-all/rewrite-all CSV updates to true single-row appends (`csv.DictWriter` in `'a'` mode), so two scripts (or two runs) can safely write to the same or cross-referenced results CSVs without one clobbering the other's rows.
- **Asymmetric cross-file awareness**: `retrain_weak_poisons.py` originally only read its own results CSV when computing the "best ASR to beat," unaware of results logged by `retrain_one_poison.py`'s separate file — fixed so both scripts check both CSVs (read-only for the other script's file) before deciding whether to overwrite a checkpoint.
- **Missing shebang**, a **missing comma** in an `ATTEMPTS` list (silent `TypeError` on run), and **`asset_eps_sweep.py`'s all-at-once CSV write** (no incremental save — a kill mid-run loses the entire sweep, unlike the other scripts) were also found; the last one was flagged but not yet fixed.
- **`blended` checkpoint currently missing from its expected location** — it was manually moved out of `poisoned_models/` at one point (to make room for an improved model that, due to the bug above, never actually got saved), and empirically verified via direct evaluation to be the older 88.70%-ASR model, not the 92.79% one.

## 9. Tooling built this session

- **`asset_threshold_sweep.py`** — resumable per-method training-threshold sweep (see §5).
- **`asset_eps_sweep.py`** — resumable-in-spirit (not yet incrementally-saving) per-method epsilon sweep over the full checkpoint set, discovering checkpoints dynamically by filename prefix (`asset_<model>_<method>_<threshold>.pth`).
- **`retrain_weak_poisons.py`** / **`retrain_one_poison.py`** — batch and single-shot retraining with safe checkpoint-overwrite logic (see §7–8).
- **`results_csv/asset_upstream_eval_summary.csv`** — hand-curated, append-only summary table of the final best threshold/epsilon/AUC/TP/FP/FN/F1 per method, the definitive upstream-evaluation reference going forward.

## 10. Miscellaneous

- Explained Linux process inspection (`ps`, `/proc/<pid>/cmdline`, job control `jobs`/`fg`/`bg`, `kill -CONT`) and how to make a Python process identifiable in `nvidia-smi`/`ps` via `setproctitle`.
- Clarified GPU/CPU resource contention from concurrently-running jobs (own training runs, other users' jobs on the shared server) as a factor in run timing and instability.
- Established that poison rate at *model-training* time is fully decoupled from poison rate at *ASSET-detection* time — training a strong backdoor at a high rate and then testing detection at a realistic low rate is valid and doesn't require the two to match.

## Pending / next steps

- **Downstream evaluation** (agreed direction, not yet started): train a *new* model on each method's dataset after removing the samples ASSET flagged as poison at its best threshold/eps operating point, then check whether ASR drops (backdoor removed) while clean accuracy is preserved. Agreed this should be a separate script (mirroring `retrain_weak_poisons.py`'s structure) rather than notebook cells, since it's batch automation across all 14 methods, not one-off interactive inspection — not yet started, pending a decision on how exactly the "cleaned dataset" should be constructed (drop all flagged TP+FP points at the chosen eps, retrain from scratch).
- `blended`'s checkpoint on disk still doesn't reflect its best logged result (92.79%) — needs one more retrain attempt that's allowed to actually beat the (now-cleared) CSV history to capture it for real.
- `asset_eps_sweep.py`'s lack of incremental saving is still an open gap (flagged, not fixed) — a kill mid-run currently loses the whole sweep.
