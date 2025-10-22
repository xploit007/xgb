import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("datasets-cli")


@dataclass
class MergeConfig:
    input_dirs: List[str]
    output_csv: str
    target_majority: str = "benign"  # 'benign' or 'attack'
    target_ratio: float = 0.6  # majority class proportion
    max_rows_per_source: Optional[int] = None
    random_state: int = 42


@dataclass
class PresetConfig:
    train_benign_dirs: List[str]
    eval_benign_dirs: List[str]
    malicious_dirs: Optional[List[str]]  # if None, autodetect
    output_dir: str = os.path.join("xgboost", "data", "splits")
    train_ratio: float = 0.6  # benign-majority
    mixed_eval_ratios: Optional[List[float]] = None  # benign-majority ratios
    max_rows_per_source: Optional[int] = None
    random_state: int = 42
    family_eval: bool = True


BENIGN_VALUES = {"benign", "normal"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_csv_files_under_dirs(dirs: List[str]) -> List[str]:
    csvs: List[str] = []
    for d in dirs:
        if not os.path.isdir(d):
            logger.warning("Input path is not a directory: %s", d)
            continue
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
    labels = df[label_col].astype(str).str.strip().str.lower()
    y = (~labels.isin(BENIGN_VALUES)).astype(int)
    return y


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

    y = normalize_labels(df_all)
    df_all["__is_attack"] = y

    if cfg.target_majority not in {"benign", "attack"}:
        raise ValueError("target_majority must be 'benign' or 'attack'")

    majority_val = 0 if cfg.target_majority == "benign" else 1
    minority_val = 1 - majority_val

    maj_df = df_all[df_all["__is_attack"] == majority_val]
    min_df = df_all[df_all["__is_attack"] == minority_val]

    maj_ratio = cfg.target_ratio
    if not (0.01 <= maj_ratio <= 0.99):
        raise ValueError("target_ratio should be between 0.01 and 0.99")

    # Choose k so that k*maj_ratio <= |maj| and k*(1-maj_ratio) <= |min|
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

    merged = merged.drop(columns=["__is_attack"], errors="ignore")
    ensure_dir(os.path.dirname(cfg.output_csv) or ".")
    merged.to_csv(cfg.output_csv, index=False)
    logger.info("Merged dataset saved: %s (rows=%d)", cfg.output_csv, len(merged))
    return cfg.output_csv


def _autodetect_malicious_dirs() -> List[str]:
    dirs = []
    for name in os.listdir('.'):
        if os.path.isdir(name) and name.lower().startswith('malicious_') and name.lower().endswith('.csv'):
            dirs.append(name)
    return sorted(dirs)


def _read_dirs_concat(dirs: List[str], max_rows: Optional[int], seed: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for d in dirs:
        csvs = list_csv_files_under_dirs([d])
        for csv in csvs:
            frames.append(read_dataset(csv, max_rows, seed))
    if not frames:
        raise RuntimeError(f"No CSV data found under: {dirs}")
    return pd.concat(frames, ignore_index=True, sort=False)


def _sample_equal_share(dfs: List[pd.DataFrame], total_target: int, seed: int) -> List[pd.DataFrame]:
    if total_target <= 0:
        return [df.iloc[0:0] for df in dfs]
    counts = [len(df) for df in dfs]
    n = len(dfs)
    base = total_target // max(1, n)
    remain = total_target - base * n
    selected_sizes = [min(base, c) for c in counts]
    i = 0
    while remain > 0 and any(selected_sizes[j] < counts[j] for j in range(n)):
        if selected_sizes[i % n] < counts[i % n]:
            selected_sizes[i % n] += 1
            remain -= 1
        i += 1
    rs = np.random.RandomState(seed)
    out = []
    for df, k in zip(dfs, selected_sizes):
        if k <= 0:
            out.append(df.iloc[0:0])
        else:
            out.append(df.sample(n=k, random_state=int(rs.randint(0, 2**31 - 1))))
    return out


def _make_eval_mix(benign_df: pd.DataFrame, mal_df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    if not (0.01 <= ratio <= 0.99):
        raise ValueError("ratio must be between 0.01 and 0.99")
    b_avail, m_avail = len(benign_df), len(mal_df)
    k_b = b_avail / max(1e-9, ratio)
    k_m = m_avail / max(1e-9, 1 - ratio)
    k = int(min(k_b, k_m))
    if k <= 0:
        raise RuntimeError("Insufficient data to compose requested ratio")
    b_target = int(round(k * ratio))
    m_target = int(round(k * (1 - ratio)))
    rs = np.random.RandomState(seed)
    b = benign_df.sample(n=b_target, random_state=int(rs.randint(0, 2**31 - 1)))
    m = mal_df.sample(n=m_target, random_state=int(rs.randint(0, 2**31 - 1)))
    return pd.concat([b, m], ignore_index=True).sample(frac=1.0, random_state=seed)


def create_presets(cfg: PresetConfig) -> Dict[str, str]:
    malicious_dirs = cfg.malicious_dirs or _autodetect_malicious_dirs()
    if not malicious_dirs:
        raise RuntimeError("No malicious directories provided or autodetected")
    ensure_dir(cfg.output_dir)

    # Read benign
    train_benign = _read_dirs_concat(cfg.train_benign_dirs, cfg.max_rows_per_source, cfg.random_state)
    eval_benign = _read_dirs_concat(cfg.eval_benign_dirs, cfg.max_rows_per_source, cfg.random_state)

    # Read malicious families
    mal_family_frames: List[Tuple[str, pd.DataFrame]] = []
    for d in malicious_dirs:
        try:
            df = _read_dirs_concat([d], cfg.max_rows_per_source, cfg.random_state)
            mal_family_frames.append((d, df))
        except Exception as e:
            logger.warning("Skipping %s: %s", d, e)
    if not mal_family_frames:
        raise RuntimeError("No malicious data loaded")

    # Compute training counts
    ratio = cfg.train_ratio
    if not (0.01 <= ratio <= 0.99):
        raise ValueError("train_ratio must be between 0.01 and 0.99")
    b_avail = len(train_benign)
    m_avail = sum(len(df) for _, df in mal_family_frames)
    k_b = b_avail / max(1e-9, ratio)
    k_m = m_avail / max(1e-9, 1 - ratio)
    k = int(min(k_b, k_m))
    if k <= 0:
        raise RuntimeError("Insufficient data to create training set")
    b_target = int(round(k * ratio))
    m_target = int(round(k * (1 - ratio)))

    # Equal-share malicious selection
    mal_selected = _sample_equal_share([df for _, df in mal_family_frames], m_target, cfg.random_state)
    mal_train = pd.concat(mal_selected, ignore_index=True) if mal_selected else pd.DataFrame()
    b_train = train_benign.sample(n=b_target, random_state=cfg.random_state)
    train_df = pd.concat([b_train, mal_train], ignore_index=True).sample(frac=1.0, random_state=cfg.random_state)

    # Save training set and keys
    train_out = os.path.join(cfg.output_dir, f"train_{int(ratio*100)}_{int((1-ratio)*100)}.csv")
    train_df.to_csv(train_out, index=False)
    if 'flow_id' not in train_df.columns:
        raise KeyError("Expected 'flow_id' column to create leakage-free eval")
    train_keys = train_df[['flow_id']].astype(str).drop_duplicates()
    keys_out = os.path.join(cfg.output_dir, "train_keys.csv")
    train_keys.to_csv(keys_out, index=False)
    logger.info("Wrote training set: %s (rows=%d) and keys: %s", train_out, len(train_df), keys_out)

    # Build remaining malicious (exclude training keys)
    all_mal = pd.concat([df for _, df in mal_family_frames], ignore_index=True)
    all_mal['flow_id'] = all_mal['flow_id'].astype(str)
    train_keys['flow_id'] = train_keys['flow_id'].astype(str)
    mal_remaining = all_mal.merge(train_keys, on='flow_id', how='left', indicator=True)
    mal_remaining = mal_remaining[mal_remaining['_merge'] == 'left_only'].drop(columns=['_merge'])

    outputs: Dict[str, str] = {"train": train_out, "train_keys": keys_out}
    # Mixed eval ratios
    ratios = cfg.mixed_eval_ratios or [0.6, 0.9, 0.99]
    for r in ratios:
        try:
            ev = _make_eval_mix(eval_benign, mal_remaining, r, cfg.random_state)
            path = os.path.join(cfg.output_dir, f"eval_mixed_{int(r*100)}_{int((1-r)*100)}.csv")
            ev.to_csv(path, index=False)
            outputs[f"eval_mixed_{int(r*100)}_{int((1-r)*100)}"] = path
            logger.info("Wrote eval mix (%.2f): %s (rows=%d)", r, path, len(ev))
        except Exception as e:
            logger.warning("Could not build eval mix %.2f: %s", r, e)

    # Per-family eval (60:40 benign-majority)
    if cfg.family_eval:
        r = 0.6
        for d, df in mal_family_frames:
            df = df.copy()
            df['flow_id'] = df['flow_id'].astype(str)
            rem = df.merge(train_keys, on='flow_id', how='left', indicator=True)
            rem = rem[rem['_merge'] == 'left_only'].drop(columns=['_merge'])
            if len(rem) == 0:
                continue
            try:
                ev = _make_eval_mix(eval_benign, rem, r, cfg.random_state)
                fam = d.replace('malicious_', '').replace('.csv', '')
                path = os.path.join(cfg.output_dir, f"eval_{fam}_{int(r*100)}_{int((1-r)*100)}.csv")
                ev.to_csv(path, index=False)
                outputs[f"eval_{fam}_{int(r*100)}_{int((1-r)*100)}"] = path
                logger.info("Wrote eval family %s: %s (rows=%d)", fam, path, len(ev))
            except Exception as e:
                logger.warning("Could not build family eval for %s: %s", d, e)

    return outputs


def parse_args():
    p = argparse.ArgumentParser(description="Datasets CLI: merge CSVs and generate preset train/eval splits")
    sub = p.add_subparsers(dest="cmd")

    mp = sub.add_parser("merge", help="Merge CSV folders into a single dataset with class ratio")
    mp.add_argument("--input-dirs", nargs="+", default=["."], help="Directories to scan for CSV files")
    mp.add_argument("--output-csv", required=True, help="Path to write merged CSV")
    mp.add_argument("--target-majority", choices=["benign", "attack"], default="benign")
    mp.add_argument("--target-ratio", type=float, default=0.6, help="Majority class proportion (0.01-0.99)")
    mp.add_argument("--max-rows-per-source", type=int, default=None, help="Optional cap per input CSV for sampling")
    mp.add_argument("--random-state", type=int, default=42)

    pp = sub.add_parser("preset", help="Create preset train/eval splits (Mon-Wed benign train; Thu-Fri benign eval; equal-share malicious in train; leakage-free eval)")
    pp.add_argument("--train-benign-dirs", nargs="+", default=["benign_monday.csv", "benign_tuesday.csv", "benign_wednesday.csv"], help="Dirs for training benign")
    pp.add_argument("--eval-benign-dirs", nargs="+", default=["benign_thursday.csv", "benign_friday.csv"], help="Dirs for evaluation benign")
    pp.add_argument("--malicious-dirs", nargs="+", help="Dirs for malicious families; autodetect if omitted")
    pp.add_argument("--output-dir", default=os.path.join("xgboost", "data", "splits"))
    pp.add_argument("--train-ratio", type=float, default=0.6, help="Benign-majority proportion for training (0.01-0.99)")
    pp.add_argument("--mixed-eval-ratios", nargs="+", type=float, default=[0.6, 0.9, 0.99], help="Benign-majority proportions for mixed eval sets")
    pp.add_argument("--max-rows-per-source", type=int, default=None)
    pp.add_argument("--random-state", type=int, default=42)
    pp.add_argument("--no-family-eval", action="store_true", help="Disable per-family evaluation set generation")

    return p.parse_args()


def main() -> int:
    try:
        args = parse_args()
        if getattr(args, 'cmd', None) == 'preset':
            cfg = PresetConfig(
                train_benign_dirs=args.train_benign_dirs,
                eval_benign_dirs=args.eval_benign_dirs,
                malicious_dirs=args.malicious_dirs,
                output_dir=args.output_dir,
                train_ratio=args.train_ratio,
                mixed_eval_ratios=args.mixed_eval_ratios,
                max_rows_per_source=args.max_rows_per_source,
                random_state=args.random_state,
                family_eval=(not args.no_family_eval),
            )
            outputs = create_presets(cfg)
            for k, v in outputs.items():
                logger.info("%s -> %s", k, v)
            return 0

        # default to merge
        cfg = MergeConfig(
            input_dirs=getattr(args, 'input_dirs', ["."]),
            output_csv=getattr(args, 'output_csv'),
            target_majority=getattr(args, 'target_majority', 'benign'),
            target_ratio=getattr(args, 'target_ratio', 0.6),
            max_rows_per_source=getattr(args, 'max_rows_per_source', None),
            random_state=getattr(args, 'random_state', 42),
        )
        merge_datasets(cfg)
        return 0
    except Exception as e:
        logger.exception("Error: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
