# XGBoost Anomaly Detection (Production Pipeline)

Two focused CLIs live in this folder:
- `xgboost/datasets.py` — dataset creation only.
- `xgboost/pipeline.py` — model training/tuning/prediction only.

Goal: extremely low false positives (precision ≈ 1, FPR ≈ 0) while keeping recall high (> 0.9). The pipeline weights benign higher and chooses a strict decision threshold on validation to cap FPR.

## Requirements
- Python 3.9+
- Install deps: `pip install -r xgboost/requirements.txt`
- Optional GPU: install a GPU-enabled `xgboost` (e.g., `conda install -c conda-forge xgboost`) and pass `--device gpu`. If `gpu_hist` is rejected, use CPU.

## Data Assumptions
- Each dataset is a directory that contains a single CSV, e.g. `benign_monday.csv/monday_benign.csv`, `malicious_dos_hulk.csv/dos_hulk.csv`.
- CSVs include a `label` column. Label mapping:
  - `Benign` or `Normal` (case-insensitive) → benign (0)
  - Anything else → attack (1)

## Preprocessing & Features
- Drops `label`, `flow_id`, `timestamp`, `src_ip`, `dst_ip` from features.
- Maps `protocol` strings (TCP/UDP/ICMP) to ints; unknown → -1.
- Casts remaining cols to numeric (`float32`), leaves NaNs for XGBoost to handle.
- Optional feature selection: `--feature-select topk --top-k N` uses XGBoost gain to keep top N features.

## Thresholding for Low FPR
- On validation, picks the largest threshold with FPR ≤ `--fpr-target` (default 0.0). This avoids flagging benign as attack.
- For labeled prediction sets, you can re-tune the threshold per-dataset with `--retune-threshold --fpr-target <small>` to push recall while keeping FPR tiny.

## One-Liners (Linux)

Dynamic paths: set your data root once, then pass it to the CLIs.

```
export DATA_ROOT=/path/to/your/network_datasets
```

- Create preset, leak-free train/test splits (Mon–Wed benign for train; Thu–Fri benign for eval; equal-share malicious in train; eval excludes any training `flow_id`):
```
python xgboost/datasets.py preset --data-root "$DATA_ROOT" --output-dir "$DATA_ROOT/presets" --train-ratio 0.6
```
- Create training mix (90:10 benign-majority) including a LOIT slice:
```
python xgboost/datasets.py merge --data-root "$DATA_ROOT" --input-dirs benign_monday.csv benign_tuesday.csv benign_wednesday.csv benign_thursday.csv malicious_dos_hulk.csv malicious_dos_golden_eye.csv malicious_dos_slowhttptest.csv malicious_dos_slowloris.csv malicious_portscan.csv malicious_web_brute_force.csv malicious_web_sql_injection.csv malicious_web_xss.csv malicious_botnet_ares.csv malicious_detected_outliers.csv malicious_ddos_loit.csv --output-csv "$DATA_ROOT/train_90_10_loit.csv" --target-majority benign --target-ratio 0.9 --max-rows-per-source 80000
```
- Create unseen DDoS LOIT 99:1 benign-majority:
```
python xgboost/datasets.py merge --data-root "$DATA_ROOT" --input-dirs benign_friday.csv malicious_ddos_loit.csv --output-csv "$DATA_ROOT/predict_ddos_99_1.csv" --target-majority benign --target-ratio 0.99 --max-rows-per-source 60000
```
- Create unseen Web 80:20:
```
python xgboost/datasets.py merge --data-root "$DATA_ROOT" --input-dirs benign_monday.csv benign_tuesday.csv malicious_web_sql_injection.csv malicious_web_brute_force.csv malicious_web_xss.csv --output-csv "$DATA_ROOT/predict_web_80_20.csv" --target-majority benign --target-ratio 0.8 --max-rows-per-source 60000

- Create multiple ratio datasets quickly (50:50, 60:40, 90:10, 99:1):
```
python xgboost/datasets.py merge --data-root "$DATA_ROOT" --input-dirs benign_monday.csv benign_tuesday.csv benign_wednesday.csv malicious_* --output-csv "$DATA_ROOT/mixed_50_50.csv" --target-majority benign --target-ratio 0.5
python xgboost/datasets.py merge --data-root "$DATA_ROOT" --input-dirs benign_monday.csv benign_tuesday.csv benign_wednesday.csv malicious_* --output-csv "$DATA_ROOT/mixed_60_40.csv" --target-majority benign --target-ratio 0.6
python xgboost/datasets.py merge --data-root "$DATA_ROOT" --input-dirs benign_monday.csv benign_tuesday.csv benign_wednesday.csv malicious_* --output-csv "$DATA_ROOT/mixed_90_10.csv" --target-majority benign --target-ratio 0.9
python xgboost/datasets.py merge --data-root "$DATA_ROOT" --input-dirs benign_friday.csv malicious_* --output-csv "$DATA_ROOT/mixed_99_1.csv" --target-majority benign --target-ratio 0.99
```
```
- Tune best model (maximize recall with FPR ≤ target, saves artifacts):
```
python xgboost/pipeline.py tune --csv "$DATA_ROOT/presets/train_60_40.csv" --device cpu --feature-select topk --top-k 100 --max-trials 16 --fpr-target 0.0 --model-dir "$DATA_ROOT/artifacts"
```
- Train with manual params (strict FPR):
```
python xgboost/pipeline.py train --csv "$DATA_ROOT/presets/train_60_40.csv" --device cpu --feature-select topk --top-k 100 --n-estimators 400 --learning-rate 0.1 --max-depth 6 --subsample 0.8 --colsample-bytree 0.6 --benign-weight 5.0 --attack-weight 1.0 --fpr-target 0.0 --model-dir "$DATA_ROOT/artifacts"
```
- Predict + evaluate (writes `<output>_metrics.json` when labels exist):
```
python xgboost/pipeline.py predict --model-dir "$DATA_ROOT/artifacts" --predict "$DATA_ROOT/presets/eval_mixed_60_40.csv" --output-csv "$DATA_ROOT/presets/pred_mixed_60_40.csv"
```
- Predict with per-dataset threshold retune (ultra-low FPR):
```
python xgboost/pipeline.py predict --model-dir "$DATA_ROOT/artifacts" --predict "$DATA_ROOT/presets/eval_mixed_90_9.csv" --output-csv "$DATA_ROOT/presets/pred_mixed_90_9_rt.csv" --retune-threshold --fpr-target 0.0001
```

## Typical End-to-End
1) Create leak-free splits: `python xgboost/datasets.py preset --output-dir xgboost/data/presets --train-ratio 0.6`
2) Tune model: `python xgboost/pipeline.py tune --train-csv xgboost/data/presets/train_60_40.csv --feature-select topk --top-k 100 --max-trials 16 --fpr-target 0.0 --device cpu --model-dir xgboost/artifacts`
3) Predict on mixed and family evals under `xgboost/data/presets` using the predict one-liners.

## Files & Artifacts
- `xgboost/datasets.py` — merge CLI + preset generator (leak-free).
- `xgboost/pipeline.py` — train/tune/predict CLI.
- `xgboost/artifacts/xgb_model.json` — saved model.
- `xgboost/artifacts/model_meta.json` — features, tuned threshold, validation metrics, params.
- `xgboost/artifacts/tuning_report.json` — trials + best config.
- Predictions: `<your_output>.csv` and `<your_output>_metrics.json`.

## Troubleshooting
- GPU not used: Your `xgboost` wheel likely lacks GPU. Install a GPU-enabled build or use `--device cpu`.
- Very large CSVs: add `--max-rows-per-source` to keep run-times manageable.
- Low recall for a family: include a small slice of that family in training, then retune or relax `--fpr-target` slightly (e.g., 0.0001) when predicting on that family.

## Version History (What We Built)
- v0.1: Base pipeline (merge/train/predict), benign weighting, strict thresholding.
- v0.2: Unlabeled predict support, metrics JSON, extreme ratios (e.g., 99:1), early-stopping fix.
- v0.3: `tune` command, feature selection (`topk`), device flag, per-dataset threshold retune.
- v0.4: Split CLIs; `datasets.py` dedicated to data, `pipeline.py` to model; added preset, leak-free train/test generator.
