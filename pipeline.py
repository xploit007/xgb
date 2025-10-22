import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except Exception as e:  # pragma: no cover
    XGBClassifier = None  # type: ignore


# ---------------------------- Logging Setup ---------------------------- #

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("xgb-pipeline")


# ---------------------------- Data Classes ---------------------------- #

@dataclass
class MergeConfig:
    input_dirs: List[str]
    output_csv: str
    target_majority: str = "benign"  # 'benign' or 'attack'
    target_ratio: float = 0.6  # majority class proportion, e.g. 0.5 or 0.6
    max_rows_per_source: Optional[int] = None  # per individual CSV
    random_state: int = 42


@dataclass
class TrainConfig:
    # Data
    train_csv: Optional[str] = None
    input_dirs: Optional[List[str]] = None
    target_majority: str = "benign"
    target_ratio: float = 0.6
    max_rows_per_source: Optional[int] = None
    random_state: int = 42
    test_size: float = 0.2

    # Model
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    n_jobs: int = -1  # use all cores

    # Class weights (emphasize benign to minimize false positives)
    benign_weight: float = 2.0
    attack_weight: float = 1.0

    # Threshold tuning
    fpr_target: float = 0.0  # try for 0 false positives on validation by default

    # Output
    model_dir: str = os.path.join("xgboost", "artifacts")
    # Device and feature selection
    device: str = "cpu"  # 'cpu' or 'gpu'
    feature_select: str = "none"  # 'none' or 'topk'
    top_k: int = 100


@dataclass
class TuneConfig:
    # Data
    train_csv: Optional[str] = None
    input_dirs: Optional[List[str]] = None
    target_majority: str = "benign"
    target_ratio: float = 0.6
    max_rows_per_source: Optional[int] = None
    random_state: int = 42
    test_size: float = 0.2

    # Search control
    max_trials: int = 12
    device: str = "cpu"
    feature_select: str = "none"
    top_k: int = 100

    # Training & eval
    model_dir: str = os.path.join("xgboost", "artifacts")
    fpr_target: float = 0.0


@dataclass
class ModelMeta:
    created_at: str
    feature_names: List[str]
    threshold: float
    params: Dict
    label_mapping: Dict[str, int]
    fpr_target: float
    metrics: Dict[str, float]


# ---------------------------- Utilities ---------------------------- #

BENIGN_VALUES = {"benign", "normal"}


def list_csv_files_under_dirs(dirs: List[str]) -> List[str]:
    csvs: List[str] = []
    for d in dirs:
        if not os.path.isdir(d):
            logger.warning("Input path is not a directory: %s", d)
            continue
        # Many datasets here are in directories named with .csv containing a single CSV file
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".csv"):
                    csvs.append(os.path.join(root, f))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under: {dirs}")
    return csvs


def read_dataset(csv_path: str, max_rows: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)
    df["__source"] = os.path.basename(csv_path)
    return df


def normalize_labels(df: pd.DataFrame, label_col: str = "label") -> pd.Series:
    if label_col not in df.columns:
        raise KeyError(f"Missing label column '{label_col}'")
    # binary: 0 -> benign, 1 -> attack
    labels = df[label_col].astype(str).str.strip().str.lower()
    y = (~labels.isin(BENIGN_VALUES)).astype(int)
    return y


def coerce_protocol(series: pd.Series) -> pd.Series:
    # Map common protocols to integers; unknowns -> -1
    mapping = {"tcp": 6, "udp": 17, "icmp": 1}
    # If already numeric, return as numeric
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.astype(str).str.lower().map(mapping).fillna(-1).astype(int)


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Drop non-feature columns and low-value identifiers
    drop_cols = [
        "label",
        "flow_id",
        "timestamp",
        "src_ip",
        "dst_ip",
    ]
    existing_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop, errors="ignore").copy()

    # Handle protocol
    if "protocol" in X.columns:
        X["protocol"] = coerce_protocol(X["protocol"])

    # Coerce any remaining non-numeric columns to numeric where possible
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Let XGBoost handle NaNs natively; ensure float32 for efficiency
    X = X.astype(np.float32)
    feature_names = list(X.columns)
    return X, feature_names


def preprocess_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    # Normalize y first
    y = normalize_labels(df)
    X, feature_names = _build_features(df)
    return X, y, feature_names


def preprocess_features_only(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Build features without requiring a label column (for inference on unlabeled CSVs)
    X, feature_names = _build_features(df)
    return X, feature_names


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def pick_threshold_min_fpr(y_true: np.ndarray, y_prob: np.ndarray, fpr_target: float) -> Tuple[float, Dict[str, float]]:
    from sklearn.metrics import roc_curve, precision_recall_fscore_support, confusion_matrix

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Pick the largest threshold with FPR <= target (bias towards predicting benign)
    mask = fpr <= fpr_target + 1e-12
    if np.any(mask):
        idx = np.where(mask)[0][-1]
    else:
        # If impossible, pick the threshold with minimal FPR
        idx = int(np.argmin(fpr))

    thr = thresholds[idx]
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    metrics = {
        "threshold": float(thr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tpr_recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "fpr": float(fp / max(1, (fp + tn))),
        "fnr": float(fn / max(1, (fn + tp))),
        "accuracy": float((tp + tn) / max(1, (tp + tn + fp + fn))),
    }
    return thr, metrics


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _device_params(device: str) -> Dict[str, str]:
    d = (device or "cpu").strip().lower()
    if d == "gpu":
        return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
    return {"tree_method": "hist"}


def _to_py(obj):
    try:
        import numpy as _np  # local alias
    except Exception:
        _np = None
    if _np is not None and isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _to_py(v) for v in obj ]
    return obj


def select_top_k_features(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    random_state: int = 42,
    device: str = "cpu",
) -> List[str]:
    if k is None or k <= 0 or k >= X.shape[1]:
        return list(X.columns)
    params = _device_params(device)
    rs = np.random.RandomState(random_state)
    seed = int(rs.randint(0, 2**31 - 1))
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric=["aucpr"],
        random_state=seed,
        verbosity=0,
        **params,
    )
    model.fit(X, y)
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    scores = {name: float(gain.get(name, 0.0)) for name in X.columns}
    ranked = sorted(scores.items(), key=lambda t: t[1], reverse=True)
    selected = [name for name, _ in ranked[:k]]
    return selected if selected else list(X.columns)


# ---------------------------- Merge Command ---------------------------- #

def merge_datasets(cfg: MergeConfig) -> str:
    np.random.seed(cfg.random_state)
    csvs = list_csv_files_under_dirs(cfg.input_dirs)
    logger.info("Found %d CSV files under %s", len(csvs), cfg.input_dirs)

    frames: List[pd.DataFrame] = []
    for csv in csvs:
        try:
            df = read_dataset(csv, cfg.max_rows_per_source, cfg.random_state)
            frames.append(df)
        except Exception as e:
            logger.warning("Skipping %s due to read error: %s", csv, e)
            continue

    if not frames:
        raise RuntimeError("No datasets could be read.")

    df_all = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Combined rows before sampling: %d", len(df_all))

    # Map labels to binary
    y = normalize_labels(df_all)
    df_all["__is_attack"] = y

    # Enforce target ratio between majority minority
    if cfg.target_majority not in {"benign", "attack"}:
        raise ValueError("target_majority must be 'benign' or 'attack'")

    majority_val = 0 if cfg.target_majority == "benign" else 1
    minority_val = 1 - majority_val

    maj_df = df_all[df_all["__is_attack"] == majority_val]
    min_df = df_all[df_all["__is_attack"] == minority_val]

    maj_ratio = cfg.target_ratio
    # Allow extreme ratios like 0.99 (benign-majority) or 0.01 (attack-majority)
    if not (0.01 <= maj_ratio <= 0.99):
        raise ValueError("target_ratio should be between 0.01 and 0.99")

    # Determine sample sizes to achieve maj:min = maj_ratio:(1-maj_ratio)
    total_desired = min(len(maj_df) + len(min_df), len(df_all))
    # Max we can get is bounded by available rows in each class
    # Let m = #maj, n = #min; pick k such that k*maj_ratio <= m and k*(1-maj_ratio) <= n
    # k <= m/maj_ratio and k <= n/(1-maj_ratio)
    k_bound_maj = len(maj_df) / max(1e-9, maj_ratio)
    k_bound_min = len(min_df) / max(1e-9, (1 - maj_ratio))
    k = int(min(k_bound_maj, k_bound_min))
    if k <= 0:
        raise RuntimeError("Insufficient data to achieve requested ratio with available rows")

    maj_target = int(round(k * maj_ratio))
    min_target = int(round(k * (1 - maj_ratio)))

    maj_sample = maj_df.sample(n=maj_target, random_state=cfg.random_state)
    min_sample = min_df.sample(n=min_target, random_state=cfg.random_state)

    merged = pd.concat([maj_sample, min_sample], ignore_index=True).sample(frac=1.0, random_state=cfg.random_state)

    # Drop helper columns before saving
    merged = merged.drop(columns=["__is_attack"], errors="ignore")

    ensure_dir(os.path.dirname(cfg.output_csv) or ".")
    merged.to_csv(cfg.output_csv, index=False)
    logger.info("Merged dataset saved: %s (rows=%d)", cfg.output_csv, len(merged))
    return cfg.output_csv


# ---------------------------- Train Command ---------------------------- #

def train_xgb(cfg: TrainConfig) -> Tuple[str, str]:
    if XGBClassifier is None:  # pragma: no cover
        raise RuntimeError("xgboost is not installed. Install with: pip install xgboost pandas scikit-learn numpy")

    # Prepare data source
    tmp_merged: Optional[str] = None
    if cfg.train_csv:
        merged_csv = cfg.train_csv
    elif cfg.input_dirs:
        tmp_merged = os.path.join("xgboost", "data", "merged_tmp.csv")
        merge_cfg = MergeConfig(
            input_dirs=cfg.input_dirs,
            output_csv=tmp_merged,
            target_majority=cfg.target_majority,
            target_ratio=cfg.target_ratio,
            max_rows_per_source=cfg.max_rows_per_source,
            random_state=cfg.random_state,
        )
        merged_csv = merge_datasets(merge_cfg)
    else:
        raise ValueError("Provide either train_csv or input_dirs")

    df = pd.read_csv(merged_csv)
    X, y, feature_names = preprocess_df(df)

    X_train, X_val, y_train, y_val = stratified_split(X, y, cfg.test_size, cfg.random_state)

    # Optional feature selection
    if cfg.feature_select.lower() == "topk":
        selected = select_top_k_features(X_train, y_train, cfg.top_k, cfg.random_state, cfg.device)
        X_train = X_train[selected]
        X_val = X_val[selected]
        feature_names = selected

    # Class weights: emphasize benign (0) to penalize false positives
    class_weight = {0: cfg.benign_weight, 1: cfg.attack_weight}
    sample_weight = y_train.map(class_weight).astype(float).values

    dev_params = _device_params(cfg.device)
    model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        n_jobs=cfg.n_jobs,
        objective="binary:logistic",
        eval_metric=["aucpr", "auc", "logloss"],
        random_state=cfg.random_state,
        verbosity=0,
        **dev_params,
    )

    logger.info("Training XGBoost with %d rows, %d features", X_train.shape[0], X_train.shape[1])
    # Use validation set for early stopping
    # Early stopping via callbacks (XGBoost 2.x); fallback gracefully if unavailable
    try:
        from xgboost.callback import EarlyStopping  # type: ignore
        callbacks = [EarlyStopping(rounds=50, save_best=True)]
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
            callbacks=callbacks,
        )
    except Exception:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    # Validate and tune threshold for target FPR
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_prob = model.predict_proba(X_val)[:, 1]
    thr, metrics = pick_threshold_min_fpr(y_val.values, y_prob, cfg.fpr_target)
    metrics["roc_auc"] = float(roc_auc_score(y_val.values, y_prob))
    metrics["pr_auc"] = float(average_precision_score(y_val.values, y_prob))
    logger.info(
        "Validation: thresh=%.6f | Acc=%.4f, Prec=%.4f, Recall=%.4f, FPR=%.6f, FNR=%.6f, ROC_AUC=%.4f, PR_AUC=%.4f",
        metrics["threshold"], metrics["accuracy"], metrics["precision"], metrics["tpr_recall"], metrics["fpr"], metrics["fnr"], metrics["roc_auc"], metrics["pr_auc"],
    )

    # Save artifacts
    ensure_dir(cfg.model_dir)
    model_path = os.path.join(cfg.model_dir, "xgb_model.json")
    meta_path = os.path.join(cfg.model_dir, "model_meta.json")

    model.save_model(model_path)
    meta = ModelMeta(
        created_at=datetime.now(timezone.utc).isoformat(),
        feature_names=feature_names,
        threshold=float(thr),
        params={
            "n_estimators": cfg.n_estimators,
            "learning_rate": cfg.learning_rate,
            "max_depth": cfg.max_depth,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_lambda": cfg.reg_lambda,
            "reg_alpha": cfg.reg_alpha,
            "device": cfg.device,
            "feature_select": cfg.feature_select,
            "top_k": cfg.top_k,
        },
        label_mapping={"benign": 0, "attack": 1},
        fpr_target=cfg.fpr_target,
        metrics=metrics,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    logger.info("Saved model to %s and metadata to %s", model_path, meta_path)

    # Clean up temporary merge
    if tmp_merged and os.path.exists(tmp_merged):
        try:
            os.remove(tmp_merged)
        except OSError:
            pass

    return model_path, meta_path


def tune_xgb(cfg: TuneConfig) -> Tuple[Dict, Dict]:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed. Install with: pip install xgboost pandas scikit-learn numpy")

    # Prepare data
    tmp_merged: Optional[str] = None
    if cfg.train_csv:
        merged_csv = cfg.train_csv
    elif cfg.input_dirs:
        tmp_merged = os.path.join("xgboost", "data", "merged_tmp_tune.csv")
        merge_cfg = MergeConfig(
            input_dirs=cfg.input_dirs,
            output_csv=tmp_merged,
            target_majority=cfg.target_majority,
            target_ratio=cfg.target_ratio,
            max_rows_per_source=cfg.max_rows_per_source,
            random_state=cfg.random_state,
        )
        merged_csv = merge_datasets(merge_cfg)
    else:
        raise ValueError("Provide either train_csv or input_dirs")

    df = pd.read_csv(merged_csv)
    X, y, feature_names = preprocess_df(df)
    X_train, X_val, y_train, y_val = stratified_split(X, y, cfg.test_size, cfg.random_state)

    # Optional feature selection once
    if cfg.feature_select.lower() == "topk":
        selected = select_top_k_features(X_train, y_train, cfg.top_k, cfg.random_state, cfg.device)
        X_train = X_train[selected]
        X_val = X_val[selected]
        feature_names = selected

    # Search space
    rs = np.random.RandomState(cfg.random_state)
    space = {
        "n_estimators": [300, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 6, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1.0, 2.0],
        "reg_alpha": [0.0, 0.3],
        "benign_weight": [3.0, 4.0, 5.0],
        "attack_weight": [1.0],
    }

    def sample_params() -> Dict:
        return {k: rs.choice(v) for k, v in space.items()}

    trials = []
    best: Optional[Dict] = None
    best_score = (-1.0, -1.0)  # (recall, pr_auc)
    dev_params = _device_params(cfg.device)

    for i in range(1, cfg.max_trials + 1):
        params = sample_params()
        class_weight = {0: float(params["benign_weight"]), 1: float(params["attack_weight"])}
        sample_weight = y_train.map(class_weight).astype(float).values
        model = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric=["aucpr", "auc", "logloss"],
            random_state=int(rs.randint(0, 2**31 - 1)),
            verbosity=0,
            **dev_params,
        )
        try:
            from xgboost.callback import EarlyStopping  # type: ignore
            callbacks = [EarlyStopping(rounds=50, save_best=True)]
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=callbacks,
            )
        except Exception:
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        from sklearn.metrics import roc_auc_score, average_precision_score
        y_prob = model.predict_proba(X_val)[:, 1]
        thr, metrics = pick_threshold_min_fpr(y_val.values, y_prob, cfg.fpr_target)
        metrics["roc_auc"] = float(roc_auc_score(y_val.values, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_val.values, y_prob))

        recall = float(metrics["tpr_recall"])
        pr_auc = float(metrics["pr_auc"])
        score = (recall, pr_auc)
        trial = {"params": params, "metrics": metrics, "threshold": float(thr)}
        trials.append(trial)
        logger.info(
            "Trial %d/%d | recall=%.4f, FPR=%.6f, prec=%.4f, PR_AUC=%.4f, ROC_AUC=%.4f, thr=%.5f | params=%s",
            i, cfg.max_trials, recall, metrics["fpr"], metrics["precision"], metrics["pr_auc"], metrics["roc_auc"], thr, params,
        )

        if score > best_score:
            best_score = score
            best = {
                "params": params,
                "metrics": metrics,
                "threshold": float(thr),
                "feature_names": feature_names,
            }

    # Save tuning report
    ensure_dir(cfg.model_dir)
    report_path = os.path.join(cfg.model_dir, "tuning_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_to_py({"best": best, "trials": trials}), f, indent=2)
    logger.info("Tuning report saved to %s", report_path)

    # Optionally save best model same as train
    if best is not None:
        # Train final model with best params on same split, save artifacts
        bp = best["params"]
        class_weight = {0: float(bp["benign_weight"]), 1: float(bp["attack_weight"])}
        sample_weight = y_train.map(class_weight).astype(float).values
        dev_params = _device_params(cfg.device)
        final = XGBClassifier(
            n_estimators=int(bp["n_estimators"]),
            learning_rate=float(bp["learning_rate"]),
            max_depth=int(bp["max_depth"]),
            subsample=float(bp["subsample"]),
            colsample_bytree=float(bp["colsample_bytree"]),
            reg_lambda=float(bp["reg_lambda"]),
            reg_alpha=float(bp["reg_alpha"]),
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric=["aucpr", "auc", "logloss"],
            random_state=cfg.random_state,
            verbosity=0,
            **dev_params,
        )
        try:
            from xgboost.callback import EarlyStopping  # type: ignore
            callbacks = [EarlyStopping(rounds=50, save_best=True)]
            final.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=callbacks,
            )
        except Exception:
            final.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        model_path = os.path.join(cfg.model_dir, "xgb_model.json")
        meta_path = os.path.join(cfg.model_dir, "model_meta.json")
        final.save_model(model_path)
        bp_py = _to_py(bp)
        meta = ModelMeta(
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_names=feature_names,
            threshold=float(best["threshold"]),
            params={**bp_py, "device": cfg.device, "feature_select": cfg.feature_select, "top_k": cfg.top_k},
            label_mapping={"benign": 0, "attack": 1},
            fpr_target=cfg.fpr_target,
            metrics=best["metrics"],
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)
        logger.info("Saved tuned model to %s and metadata to %s", model_path, meta_path)

    if tmp_merged and os.path.exists(tmp_merged):
        try:
            os.remove(tmp_merged)
        except OSError:
            pass

    return best or {}, {"report_path": report_path}


# ---------------------------- Predict Command ---------------------------- #

def load_model_and_meta(model_dir: str):
    from xgboost import XGBClassifier

    model_path = os.path.join(model_dir, "xgb_model.json")
    meta_path = os.path.join(model_dir, "model_meta.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Artifacts not found in {model_dir}")
    model = XGBClassifier()
    model.load_model(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def predict_on_csv(model_dir: str, input_csv: str, output_csv: str, fpr_target: Optional[float] = None, retune_threshold: bool = False):
    model, meta = load_model_and_meta(model_dir)
    df = pd.read_csv(input_csv)
    # Build features without label requirement
    X, feature_names = preprocess_features_only(df)

    # Align features to training set
    train_feats = meta["feature_names"]
    missing = [c for c in train_feats if c not in X.columns]
    if missing:
        for m in missing:
            X[m] = np.nan
    extra = [c for c in X.columns if c not in train_feats]
    if extra:
        X = X[train_feats]

    y_prob = model.predict_proba(X)[:, 1]
    thr = float(meta["threshold"])
    # Optionally retune threshold on this dataset if labels exist
    if retune_threshold and "label" in df.columns:
        try:
            y_true = normalize_labels(df).values
            target = float(meta.get("fpr_target", 0.0)) if fpr_target is None else float(fpr_target)
            thr_rt, _ = pick_threshold_min_fpr(y_true, y_prob, target)
            thr = float(thr_rt)
            logger.info("Retuned threshold on input to %.6f for FPR target %.6f", thr, target)
        except Exception as e:
            logger.warning("Could not retune threshold: %s", e)
    y_pred = (y_prob >= thr).astype(int)

    out = df.copy()
    out["attack_probability"] = y_prob
    out["prediction"] = np.where(y_pred == 1, "attack", "benign")
    out.to_csv(output_csv, index=False)
    logger.info("Predictions written to %s", output_csv)

    # If labels are present, compute and log evaluation metrics
    if "label" in df.columns:
        try:
            from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score, average_precision_score
            y_true = normalize_labels(df).values
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            fpr = float(fp / max(1, (fp + tn)))
            fnr = float(fn / max(1, (fn + tp)))
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
            metrics = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
                "fnr": float(fnr),
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "threshold": float(meta.get("threshold", 0.5)),
            }
            logger.info(
                "Evaluate: Acc=%.4f, Prec=%.4f, Recall=%.4f, F1=%.4f, FPR=%.6f, FNR=%.6f, ROC_AUC=%.4f, PR_AUC=%.4f | TP=%d, TN=%d, FP=%d, FN=%d",
                acc, precision, recall, f1, fpr, fnr, roc_auc, pr_auc, tp, tn, fp, fn,
            )
            # Save metrics JSON next to predictions
            try:
                metrics_path = os.path.splitext(output_csv)[0] + "_metrics.json"
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                logger.info("Metrics written to %s", metrics_path)
            except Exception as e:
                logger.warning("Could not write metrics JSON: %s", e)
        except Exception as e:
            logger.warning("Could not compute evaluation metrics: %s", e)


# ---------------------------- CLI ---------------------------- #

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Production-grade XGBoost pipeline for anomaly detection (benign vs attack)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Dataset creation lives in xgboost/datasets.py; this CLI focuses on train/tune/predict

    tp = sub.add_parser("train", help="Train XGBoost model; can merge on the fly")
    tp.add_argument("--train-csv", help="Pre-merged CSV to train on")
    tp.add_argument("--input-dirs", nargs="+", help="Directories to scan and merge for training")
    tp.add_argument("--target-majority", choices=["benign", "attack"], default="benign")
    tp.add_argument("--target-ratio", type=float, default=0.6)
    tp.add_argument("--max-rows-per-source", type=int, default=None)
    tp.add_argument("--random-state", type=int, default=42)
    tp.add_argument("--test-size", type=float, default=0.2)
    tp.add_argument("--n-estimators", type=int, default=500)
    tp.add_argument("--learning-rate", type=float, default=0.05)
    tp.add_argument("--max-depth", type=int, default=6)
    tp.add_argument("--subsample", type=float, default=0.8)
    tp.add_argument("--colsample-bytree", type=float, default=0.8)
    tp.add_argument("--reg-lambda", type=float, default=1.0)
    tp.add_argument("--reg-alpha", type=float, default=0.0)
    tp.add_argument("--n-jobs", type=int, default=-1)
    tp.add_argument("--benign-weight", type=float, default=2.0)
    tp.add_argument("--attack-weight", type=float, default=1.0)
    tp.add_argument("--fpr-target", type=float, default=0.0, help="Desired max false positive rate on validation")
    tp.add_argument("--model-dir", default=os.path.join("xgboost", "artifacts"))
    tp.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    tp.add_argument("--feature-select", choices=["none", "topk"], default="none")
    tp.add_argument("--top-k", type=int, default=100)

    tunep = sub.add_parser("tune", help="Random-search hyperparameters to maximize recall at FPR<=target")
    tunep.add_argument("--train-csv", help="Pre-merged CSV to tune on")
    tunep.add_argument("--input-dirs", nargs="+", help="Directories to scan and merge for tuning")
    tunep.add_argument("--target-majority", choices=["benign", "attack"], default="benign")
    tunep.add_argument("--target-ratio", type=float, default=0.6)
    tunep.add_argument("--max-rows-per-source", type=int, default=None)
    tunep.add_argument("--random-state", type=int, default=42)
    tunep.add_argument("--test-size", type=float, default=0.2)
    tunep.add_argument("--max-trials", type=int, default=12)
    tunep.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    tunep.add_argument("--feature-select", choices=["none", "topk"], default="none")
    tunep.add_argument("--top-k", type=int, default=100)
    tunep.add_argument("--model-dir", default=os.path.join("xgboost", "artifacts"))
    tunep.add_argument("--fpr-target", type=float, default=0.0)

    pp = sub.add_parser("predict", help="Run inference on a CSV with trained artifacts")
    pp.add_argument("--model-dir", default=os.path.join("xgboost", "artifacts"))
    pp.add_argument("--input-csv", required=True)
    pp.add_argument("--output-csv", required=True)
    pp.add_argument("--fpr-target", type=float, default=None, help="Optional FPR target to retune threshold on input (requires labels)")
    pp.add_argument("--retune-threshold", action="store_true", help="If set and labels exist, retune threshold on this input for given FPR target")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.cmd == "train":
            cfg = TrainConfig(
                train_csv=args.train_csv,
                input_dirs=args.input_dirs,
                target_majority=args.target_majority,
                target_ratio=args.target_ratio,
                max_rows_per_source=args.max_rows_per_source,
                random_state=args.random_state,
                test_size=args.test_size,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                reg_lambda=args.reg_lambda,
                reg_alpha=args.reg_alpha,
                n_jobs=args.n_jobs,
                benign_weight=args.benign_weight,
                attack_weight=args.attack_weight,
                fpr_target=args.fpr_target,
                model_dir=args.model_dir,
                device=args.device,
                feature_select=args.feature_select,
                top_k=args.top_k,
            )
            train_xgb(cfg)
            return 0

        if args.cmd == "tune":
            cfg = TuneConfig(
                train_csv=args.train_csv,
                input_dirs=args.input_dirs,
                target_majority=args.target_majority,
                target_ratio=args.target_ratio,
                max_rows_per_source=args.max_rows_per_source,
                random_state=args.random_state,
                test_size=args.test_size,
                max_trials=args.max_trials,
                device=args.device,
                feature_select=args.feature_select,
                top_k=args.top_k,
                model_dir=args.model_dir,
                fpr_target=args.fpr_target,
            )
            tune_xgb(cfg)
            return 0

        if args.cmd == "predict":
            predict_on_csv(args.model_dir, args.input_csv, args.output_csv, args.fpr_target, args.retune_threshold)
            return 0

        logger.error("Unknown command: %s", args.cmd)
        return 2
    except Exception as e:
        logger.exception("Error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
