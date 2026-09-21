#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse, os, time, torch, subprocess, inspect, threading, re, json, shutil, glob
from pathlib import Path, PurePosixPath, PureWindowsPath
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
import logging

import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

logging.basicConfig(level=logging.INFO)


# Model loading
def _prepare_load_kwargs(name_or_path: str):
    is_local = os.path.isdir(name_or_path)
    if is_local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        load_kwargs = dict(trust_remote_code=True, local_files_only=True)
    else:
        load_kwargs = dict(trust_remote_code=True)
    return name_or_path, load_kwargs


def _assert_hf_checkpoint_if_local(path: str):
    if not os.path.isdir(path):
        return
    cfg = os.path.join(path, "config.json")
    if not os.path.exists(cfg):
        raise RuntimeError(
            f"[FATAL] --model_name 指向本地目录，但该目录缺少 config.json：\n"
            f"  {path}\n\n"
            f"你可能传了源码目录而不是 Hugging Face checkpoint。\n"
            f"解决：传真正 checkpoint 目录（含 config.json + 权重）。\n"
        )


# GPU monitoring
def monitor_gpu(interval=60):
    def loop():
        while True:
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                )
                if r.stdout.strip():
                    used, total = map(int, r.stdout.strip().split(","))
                    logging.info(f"[GPU] memory used: {used:4d} MiB / {total:4d} MiB")
            except Exception:
                pass
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()


# Data preparation
def clean_dna_seq(s: str) -> str:
    s = str(s).upper().replace("U", "T")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^ACGTN]", "N", s)
    return s


def to_kmer_spaced(seq: str, k: int) -> str:
    if k <= 0:
        return seq
    if len(seq) < k:
        return seq
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


def _detect_seq_col(columns):
    for k in ("sequence", "seq", "dna_sequence", "dna_seq", "text"):
        if k in columns:
            return k
    return list(columns)[0]


def _ensure_dir(p: str):
    if p and len(p) > 0:
        os.makedirs(p, exist_ok=True)


def _coerce_and_filter_df(df: pd.DataFrame, seq_col: str, label_col: str) -> pd.DataFrame:
    df = df.copy()
    df[seq_col] = df[seq_col].astype(str)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[label_col])
    df = df[df[seq_col].str.len() > 0]
    after = len(df)
    print(f"[Filter] kept {after}/{before} (dropped {before-after})")
    return df


def _seq_leakage_check(train_df, val_df, seq_col: str):
    st = set(train_df[seq_col].tolist())
    sv = set(val_df[seq_col].tolist())
    inter = st & sv
    return len(inter), (next(iter(inter)) if len(inter) else None)


def _label_conflict_report(df: pd.DataFrame, seq_col: str, label_col: str, out_json_path: str):
    g = df.groupby(seq_col)[label_col]
    counts = g.size().astype(int)
    stds = g.std(ddof=0)
    means = g.mean()

    n_total = int(len(df))
    n_unique = int(counts.shape[0])
    n_dup_seq = int((counts > 1).sum())
    dup_rows = int(df.duplicated(subset=[seq_col]).sum())

    conflict_mask = (counts > 1) & (stds.fillna(0.0) > 0)
    n_conflict_seq = int(conflict_mask.sum())

    top = (
        pd.DataFrame({
            "count": counts,
            "mean": means,
            "std": stds.fillna(0.0),
        })
        .sort_values(["std", "count"], ascending=[False, False])
        .head(20)
        .reset_index()
        .rename(columns={seq_col: "sequence"})
    )

    report = {
        "n_total_rows": n_total,
        "n_unique_sequences": n_unique,
        "n_duplicate_sequences": n_dup_seq,
        "n_duplicate_rows": dup_rows,
        "n_conflict_sequences_std_gt_0": n_conflict_seq,
        "top_conflicts": top.to_dict(orient="records"),
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def prepare_split_from_one_csv(
    data_csv: str,
    label_col: str,
    split_dir: str,
    val_ratio: float,
    split_seed: int,
    dedup_policy: str = "mean",
    leakage_strict: bool = False,
):
    _ensure_dir(split_dir)

    df = pd.read_csv(data_csv)
    seq_col = _detect_seq_col(df.columns)
    df = _coerce_and_filter_df(df, seq_col=seq_col, label_col=label_col)
    df[seq_col] = df[seq_col].map(clean_dna_seq)

    conflict_json = os.path.join(split_dir, "label_conflict_report.json")
    conflict_report = _label_conflict_report(df, seq_col, label_col, conflict_json)

    if dedup_policy in ("mean", "median"):
        agg_fn = "mean" if dedup_policy == "mean" else "median"
        df = df.groupby(seq_col, as_index=False)[label_col].agg(agg_fn)
    elif dedup_policy == "keep_first":
        df = df.drop_duplicates(subset=[seq_col], keep="first").reset_index(drop=True)
    elif dedup_policy == "drop_conflict":
        g = df.groupby(seq_col)[label_col]
        stds = g.std(ddof=0).fillna(0.0)
        counts = g.size()
        ok = (counts == 1) | (stds == 0.0)
        keep_seqs = set(ok[ok].index.tolist())
        df = df[df[seq_col].isin(keep_seqs)].copy()
        df = df.groupby(seq_col, as_index=False)[label_col].mean()
    else:
        raise ValueError(f"Unknown dedup_policy: {dedup_policy}")

    seed = None if (split_seed is None or int(split_seed) < 0) else int(split_seed)
    rng = np.random.default_rng(seed)

    y = df[label_col].to_numpy(dtype=float)
    n = len(df)
    n_val = int(round(n * float(val_ratio)))

    n_bins = 10
    n_bins = max(2, min(n_bins, int(np.sqrt(n))))
    ranks = pd.Series(y).rank(method="average").to_numpy()
    bins = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    bins = np.asarray(bins, dtype=int)

    val_mask = np.zeros(n, dtype=bool)
    for b in np.unique(bins):
        idx_b = np.where(bins == b)[0]
        rng.shuffle(idx_b)
        take = int(round(len(idx_b) * float(val_ratio)))
        if take > 0:
            val_mask[idx_b[:take]] = True

    cur = int(val_mask.sum())
    if cur < n_val:
        rest = np.where(~val_mask)[0]
        rng.shuffle(rest)
        val_mask[rest[:(n_val - cur)]] = True
    elif cur > n_val:
        chosen = np.where(val_mask)[0]
        rng.shuffle(chosen)
        val_mask[chosen[:(cur - n_val)]] = False

    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]

    tr_df = df.iloc[train_idx].reset_index(drop=True)
    va_df = df.iloc[val_idx].reset_index(drop=True)

    y_tr = tr_df[label_col].to_numpy(dtype=float)
    y_va = va_df[label_col].to_numpy(dtype=float)

    tr_max = float(np.max(y_tr)) if len(y_tr) else float("-inf")
    va_max = float(np.max(y_va)) if len(y_va) else float("-inf")
    eps = 1e-9

    if va_max + eps < tr_max and len(tr_df) > 0 and len(va_df) > 0:
        i_tr = int(np.argmax(y_tr))
        row_tr = tr_df.iloc[[i_tr]].copy()

        med_va = float(np.median(y_va))
        j_va = int(np.argmin(np.abs(y_va - med_va)))
        row_va = va_df.iloc[[j_va]].copy()

        tr_df = pd.concat([tr_df.drop(index=i_tr), row_va], ignore_index=True)
        va_df = pd.concat([va_df.drop(index=j_va), row_tr], ignore_index=True)

        y_tr2 = tr_df[label_col].to_numpy(dtype=float)
        y_va2 = va_df[label_col].to_numpy(dtype=float)
        print(f"[TailCover] swapped 1 sample to ensure val tail. "
              f"train_max: {tr_max:.6g}->{float(np.max(y_tr2)):.6g}  "
              f"val_max: {va_max:.6g}->{float(np.max(y_va2)):.6g}")
    else:
        print(f"[TailCover] no swap needed. train_max={tr_max:.6g} val_max={va_max:.6g}")

    n_inter, example = _seq_leakage_check(tr_df, va_df, seq_col)
    print(f"[LeakageCheck] shared_sequences={n_inter}  "
          f"train_unique={tr_df[seq_col].nunique()}  val_unique={va_df[seq_col].nunique()}")
    if n_inter > 0:
        print("[LeakageCheck] example:", str(example)[:120])
        if leakage_strict:
            raise RuntimeError(f"[LEAKAGE] shared_sequences={n_inter}, example={str(example)[:80]}")

    def _summ(arr):
        arr = np.asarray(arr, dtype=float)
        return {
            "n": int(arr.size),
            "min": float(np.min(arr)),
            "med": float(np.median(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    print("[LabelDist] train:", _summ(tr_df[label_col].to_numpy()))
    print("[LabelDist] val  :", _summ(va_df[label_col].to_numpy()))

    train_out = os.path.join(split_dir, "train.csv")
    val_out = os.path.join(split_dir, "validation.csv")
    tr_df.to_csv(train_out, index=False)
    va_df.to_csv(val_out, index=False)

    meta = {
        "source_csv": data_csv,
        "seq_col": seq_col,
        "label_col": label_col,
        "val_ratio": float(val_ratio),
        "split_seed": seed,
        "dedup_policy": dedup_policy,
        "leakage_strict": bool(leakage_strict),
        "n_total_after_filter": int(conflict_report["n_total_rows"]),
        "n_unique_sequences_before_dedup": int(conflict_report["n_unique_sequences"]),
        "n_duplicate_sequences": int(conflict_report["n_duplicate_sequences"]),
        "n_duplicate_rows": int(conflict_report["n_duplicate_rows"]),
        "n_conflict_sequences_std_gt_0": int(conflict_report["n_conflict_sequences_std_gt_0"]),
        "n_after_dedup": int(len(df)),
        "n_train": int(len(tr_df)),
        "n_val": int(len(va_df)),
        "post_split_shared_sequences": int(n_inter),
        "reports": {
            "label_conflict_report": conflict_json,
        }
    }
    with open(os.path.join(split_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"🪄 Split done (grouped by sequence): train={len(tr_df)} val={len(va_df)}")
    return train_out, val_out


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("predictor_train_config.json")


def _is_absolute_path(value):
    return (
        Path(value).is_absolute()
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    )


def _load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    for key in (
        "model_name",
        "train_csv",
        "validation_csv",
        "data_csv",
        "split_dir",
        "output_dir",
    ):
        value = config.get(key)
        if value and not _is_absolute_path(value):
            config[key] = str(PROJECT_ROOT / value)

    if config.pop("wandb_disabled", False):
        os.environ["WANDB_DISABLED"] = "true"

    print(f"[CONFIG] Loaded: {config_path}")
    return config


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    config_args, remaining_args = config_parser.parse_known_args()
    config = _load_config(config_args.config)

    p = argparse.ArgumentParser("Route-A (NO PRIORS): Frozen encoder + CNN regression")

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--validation_csv", type=str, default=None)

    p.add_argument("--data_csv", type=str, default=None)
    p.add_argument("--split_dir", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)

    p.add_argument("--dedup_policy", type=str, default="mean",
                   choices=["mean", "median", "keep_first", "drop_conflict"])
    p.add_argument("--leakage_strict", action="store_true")

    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_length", type=int, default=200)
    p.add_argument("--pad_to_multiple_of", type=int, default=8)

    p.add_argument("--monitor_interval", type=int, default=60)
    p.add_argument("--label_col", type=str, default="activity")

    p.add_argument("--label_transform", type=str, default="zscore",
                   choices=["none", "zscore"])

    p.add_argument("--tokenizer_sanity_n", type=int, default=3)
    p.add_argument("--kmer", type=int, default=0)

    p.add_argument("--loss_type", type=str, default="huber", choices=["mse", "huber"])
    p.add_argument("--huber_delta", type=float, default=1.0)

    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--pooling", type=str, default="mean_last4",
                   choices=["mean", "cls", "mean_last4"])

    p.add_argument("--head_type", type=str, default="cnn", choices=["cnn"])
    p.add_argument("--head_hidden", type=int, default=512)
    p.add_argument("--head_dropout", type=float, default=0.05)

    p.add_argument("--cnn_kernels", type=str, default="3,5,7,11")
    p.add_argument("--cnn_channels", type=int, default=128)
    p.add_argument("--cnn_pool", type=str, default="maxmean",
                   choices=["max", "maxmean", "lse"])
    p.add_argument("--lse_tau", type=float, default=1.0)

    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--early_stopping_patience", type=int, default=8)
    p.add_argument("--early_stopping_threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    boolean_flags = {"freeze_encoder", "leakage_strict"}
    config_cli_args = []
    for key, value in config.items():
        option = f"--{key}"
        if key in boolean_flags:
            if value:
                config_cli_args.append(option)
        elif value is not None:
            config_cli_args.extend([option, str(value)])

    args = p.parse_args(config_cli_args + remaining_args)

    if args.data_csv is None and args.train_csv is None:
        raise RuntimeError("You must provide either --data_csv OR --train_csv/--validation_csv.")
    return args


# Tokenizer
def setup_tokenizer(name_or_path: str):
    _assert_hf_checkpoint_if_local(name_or_path)
    src, kw = _prepare_load_kwargs(name_or_path)
    tok = AutoTokenizer.from_pretrained(src, **kw)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    return tok


def sanity_check_tokenizer(tok, seqs, max_len=200, n=3):
    if n <= 0:
        return
    print("\n===== Tokenizer sanity check =====")
    unk_id = tok.unk_token_id
    for s in seqs[:n]:
        s = str(s)
        enc = tok(s, truncation=True, max_length=max_len, padding=False)
        ids = enc["input_ids"]
        unk_ratio = 0.0 if (unk_id is None or len(ids) == 0) else float((np.array(ids) == unk_id).mean())
        print("SEQ[:80]   =", s[:80])
        print("len_ids    =", len(ids))
        print("unk_ratio  =", unk_ratio)
        print("-" * 60)
    print("=================================\n")


# Dataset and label normalization
def setup_dataset(args, tokenizer):
    raw_train = pd.read_csv(args.train_csv)
    seq_col = _detect_seq_col(raw_train.columns)
    os.makedirs(args.output_dir, exist_ok=True)

    def clean_csv(in_csv: str, out_csv: str, seq_col: str, label_col: str):
        df = pd.read_csv(in_csv)
        df[seq_col] = df[seq_col].astype(str)
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        before = len(df)
        df = df.dropna(subset=[label_col])
        df = df[df[seq_col].str.len() > 0]
        after = len(df)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[Clean] {in_csv} -> {out_csv} | kept {after}/{before}")
        return out_csv

    train_csv_clean = clean_csv(args.train_csv, os.path.join(args.output_dir, "clean_train.csv"), seq_col, args.label_col)

    val_csv_clean = None
    if args.validation_csv:
        raw_val = pd.read_csv(args.validation_csv)
        seq_col_v = _detect_seq_col(raw_val.columns)
        val_csv_clean = clean_csv(args.validation_csv, os.path.join(args.output_dir, "clean_val.csv"), seq_col_v, args.label_col)

    train_df = pd.read_csv(train_csv_clean)
    y = pd.to_numeric(train_df[args.label_col], errors="coerce").dropna()

    mu, sd = 0.0, 1.0
    if args.label_transform == "zscore":
        mu = float(y.mean())
        sd = float(y.std(ddof=0) + 1e-8)
        print(f"[ZScore] mean={mu:.6g} std={sd:.6g}")

    def transform_y(v):
        v = float(v)
        if args.label_transform == "zscore":
            return (v - mu) / sd
        return v

    files = {"train": train_csv_clean}
    if val_csv_clean:
        files["validation"] = val_csv_clean

    ds = load_dataset("csv", data_files=files)
    seq_col2 = _detect_seq_col(ds["train"].column_names)

    def tok_one(seqs):
        seqs_clean = [clean_dna_seq(s) for s in seqs]
        seqs_in = seqs_clean if args.kmer <= 0 else [to_kmer_spaced(s, args.kmer) for s in seqs_clean]
        out = tokenizer(
            seqs_in,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_attention_mask=True,
        )
        return out

    def tok_fn(batch):
        raw_seqs = batch[seq_col2]
        out = tok_one(raw_seqs)
        out["labels"] = [transform_y(v) for v in batch[args.label_col]]
        return out

    remove_cols = ds["train"].column_names
    ds = {k: v.map(tok_fn, batched=True, remove_columns=remove_cols) for k, v in ds.items()}

    train_ds = ds["train"]
    eval_ds = ds.get("validation")
    if eval_ds is None:
        raise RuntimeError("No validation set found.")

    pad_mul = args.pad_to_multiple_of if args.pad_to_multiple_of and args.pad_to_multiple_of > 0 else None
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_mul)

    stats = {
        "label_transform": args.label_transform,
        "mu": mu,
        "sd": sd,
        "label_col": args.label_col
    }
    with open(os.path.join(args.output_dir, "label_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return train_ds, eval_ds, collator, stats


# Model
class FrozenEncoderCNNRegressor(nn.Module):
    def __init__(
        self,
        encoder,
        hidden_size: int,
        pooling: str = "mean_last4",
        head_hidden: int = 512,
        head_dropout: float = 0.05,
        loss_type: str = "huber",
        huber_delta: float = 1.0,
        cnn_kernels=(3, 5, 7, 11),
        cnn_channels=128,
        cnn_pool: str = "maxmean",
        lse_tau: float = 1.0,
        label_transform: str = "none",
        label_mu: float = 0.0,
        label_sd: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.loss_type = loss_type
        self.huber_delta = float(huber_delta)

        self.label_transform = label_transform
        self.label_mu = float(label_mu)
        self.label_sd = float(label_sd)

        self.cnn_kernels = [int(k) for k in cnn_kernels]
        self.cnn_channels = int(cnn_channels)
        self.cnn_pool = str(cnn_pool)
        self.lse_tau = float(lse_tau)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, self.cnn_channels, int(k), padding=int(k)//2),
                nn.GELU(),
                nn.GroupNorm(num_groups=8, num_channels=self.cnn_channels),
            )
            for k in self.cnn_kernels
        ])

        base_dim = self.cnn_channels * len(self.cnn_kernels)
        feat_dim = base_dim * 2 if self.cnn_pool == "maxmean" else base_dim
        self.feat_dim = int(feat_dim)

        self.reg_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def _encode_hidden(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=(self.pooling == "mean_last4"),
            return_dict=True,
        )
        if self.pooling == "mean_last4":
            hs = out.hidden_states
            H = torch.stack(hs[-4:], dim=0).mean(dim=0)
        else:
            H = out.last_hidden_state
        return H

    def _cnn_feats(self, H, attention_mask):
        x = H.transpose(1, 2)
        m = attention_mask.unsqueeze(1).to(dtype=x.dtype)
        feats = []

        for conv in self.convs:
            y = conv(x)
            y_masked = y.masked_fill(m == 0, -1e9)
            y_max = torch.amax(y_masked, dim=-1)

            if self.cnn_pool == "max":
                feats.append(y_max)
            elif self.cnn_pool == "maxmean":
                denom = m.sum(dim=-1).clamp(min=1.0)
                y_mean = (y * m).sum(dim=-1) / denom
                feats.append(torch.cat([y_max, y_mean], dim=-1))
            else:
                tau = max(self.lse_tau, 1e-6)
                y_lse = torch.logsumexp(y_masked / tau, dim=-1) * tau
                feats.append(y_lse)

        return torch.cat(feats, dim=-1)

    def _reg_loss(self, preds, labels):
        labels = labels.view(-1).to(preds.dtype)
        if self.loss_type == "huber":
            return F.huber_loss(preds, labels, delta=self.huber_delta)
        return F.mse_loss(preds, labels)

    def denorm_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        if self.label_transform == "zscore":
            return preds * self.label_sd + self.label_mu
        return preds

    @torch.no_grad()
    def predict_activity(self, input_ids=None, attention_mask=None):
        H = self._encode_hidden(input_ids, attention_mask)
        f = self._cnn_feats(H, attention_mask)
        logits = self.reg_head(f).view(-1)
        return self.denorm_predictions(logits)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        H = self._encode_hidden(input_ids, attention_mask)
        f = self._cnn_feats(H, attention_mask)
        logits = self.reg_head(f).view(-1)
        loss = None if labels is None else self._reg_loss(logits, labels)
        return {"loss": loss, "logits": logits}

    def save_model(self, output_dir, model_meta: dict = None):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        meta = model_meta or {}
        with open(os.path.join(output_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Model saved: {output_dir}/pytorch_model.bin")


def setup_model(args, label_stats):
    _assert_hf_checkpoint_if_local(args.model_name)
    src, kw = _prepare_load_kwargs(args.model_name)
    encoder = AutoModel.from_pretrained(src, **kw)
    hidden = int(getattr(encoder.config, "hidden_size", 768))

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        print("🧊 Encoder frozen (Route-A). Only training CNN head.")
    else:
        print("🔥 Encoder trainable.")

    kernels = [int(x) for x in args.cnn_kernels.split(",") if x.strip()]
    print(f"[ModelInit] hidden={hidden} kernels={tuple(kernels)} channels={args.cnn_channels} pool={args.cnn_pool}")

    return FrozenEncoderCNNRegressor(
        encoder=encoder,
        hidden_size=hidden,
        pooling=args.pooling,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        cnn_kernels=kernels,
        cnn_channels=args.cnn_channels,
        cnn_pool=args.cnn_pool,
        lse_tau=args.lse_tau,
        label_transform=label_stats["label_transform"],
        label_mu=label_stats["mu"],
        label_sd=label_stats["sd"],
    )


# Metrics
def build_compute_metrics(label_stats):
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        preds = np.asarray(preds).reshape(-1)
        labels = np.asarray(labels).reshape(-1)

        m = ~(np.isnan(preds) | np.isnan(labels))
        preds = preds[m]
        labels = labels[m]

        if preds.size == 0:
            return {"mse": float("nan"), "pearson": float("nan"), "spearman": float("nan")}

        mse = float(np.mean((preds - labels) ** 2))
        pstd = float(np.std(preds))
        lstd = float(np.std(labels))
        print(f"[EvalDebug-z] preds_std={pstd:.6g} labels_std={lstd:.6g} "
              f"preds_minmax=({preds.min():.6g},{preds.max():.6g}) "
              f"labels_minmax=({labels.min():.6g},{labels.max():.6g})")

        if label_stats.get("label_transform") == "zscore":
            mu = float(label_stats["mu"])
            sd = float(label_stats["sd"])
            preds_raw = preds * sd + mu
            labels_raw = labels * sd + mu
            print(f"[EvalDebug-raw] preds_minmax=({preds_raw.min():.6g},{preds_raw.max():.6g}) "
                  f"labels_minmax=({labels_raw.min():.6g},{labels_raw.max():.6g})")

        if pstd == 0.0 or lstd == 0.0:
            return {"mse": mse, "pearson": float("nan"), "spearman": float("nan")}

        p = float(pearsonr(preds, labels)[0])
        s = float(spearmanr(preds, labels)[0])
        return {"mse": mse, "pearson": p, "spearman": s}
    return compute_metrics


# Checkpoint cleanup
def cleanup_checkpoints(output_dir: str):
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    for p in ckpts:
        if os.path.isdir(p):
            print(f"[Cleanup] removing {p}")
            shutil.rmtree(p, ignore_errors=True)


# Training
def train(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    monitor_gpu(args.monitor_interval)

    if args.data_csv is not None:
        split_dir = args.split_dir if args.split_dir is not None else os.path.join(args.output_dir, "split")
        _ensure_dir(split_dir)
        tr_path, va_path = prepare_split_from_one_csv(
            data_csv=args.data_csv,
            label_col=args.label_col,
            split_dir=split_dir,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
            dedup_policy=args.dedup_policy,
            leakage_strict=args.leakage_strict,
        )
        args.train_csv = tr_path
        args.validation_csv = va_path

    tok = setup_tokenizer(args.model_name)

    raw_train = pd.read_csv(args.train_csv)
    seq_col = _detect_seq_col(raw_train.columns)
    preview = [to_kmer_spaced(clean_dna_seq(s), args.kmer) for s in raw_train[seq_col].astype(str).tolist()]
    sanity_check_tokenizer(tok, preview, max_len=args.max_length, n=args.tokenizer_sanity_n)

    train_ds, val_ds, collator, label_stats = setup_dataset(args, tok)
    model = setup_model(args, label_stats)

    sig = inspect.signature(TrainingArguments.__init__)
    kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        report_to="none",
    )

    if "warmup_ratio" in sig.parameters:
        kwargs["warmup_ratio"] = args.warmup_ratio
    if "weight_decay" in sig.parameters:
        kwargs["weight_decay"] = args.weight_decay
    if "max_grad_norm" in sig.parameters:
        kwargs["max_grad_norm"] = args.max_grad_norm

    if "bf16" in sig.parameters:
        kwargs["bf16"] = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if "fp16" in sig.parameters and not kwargs.get("bf16", False):
        kwargs["fp16"] = torch.cuda.is_available()

    if "load_best_model_at_end" in sig.parameters:
        kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in sig.parameters:
        kwargs["metric_for_best_model"] = "eval_pearson"
    if "greater_is_better" in sig.parameters:
        kwargs["greater_is_better"] = True
    if "save_total_limit" in sig.parameters:
        kwargs["save_total_limit"] = 1

    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch"
        if "save_strategy" in sig.parameters:
            kwargs["save_strategy"] = "epoch"
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
        if "save_strategy" in sig.parameters:
            kwargs["save_strategy"] = "epoch"

    t_args = TrainingArguments(**kwargs)

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )
    ]

    trainer = Trainer(
        model=model,
        args=t_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=build_compute_metrics(label_stats),
        callbacks=callbacks,
    )

    print("🏋️ Start training ...")
    t0 = time.time()
    trainer.train()

    best_metric = trainer.state.best_metric
    best_ckpt = trainer.state.best_model_checkpoint
    if best_metric is not None:
        print(f"🏆 Best eval_pearson = {best_metric:.6f}")
        if best_ckpt is not None:
            print(f"🏆 Best checkpoint = {best_ckpt}")

    print(f"✅ Training done in {(time.time() - t0) / 60:.2f} min")

    meta = {
        "model_name": args.model_name,
        "freeze_encoder": bool(args.freeze_encoder),
        "pooling": args.pooling,
        "cnn_kernels": args.cnn_kernels,
        "cnn_channels": int(args.cnn_channels),
        "cnn_pool": args.cnn_pool,
        "lse_tau": float(args.lse_tau),
        "head_hidden": int(args.head_hidden),
        "head_dropout": float(args.head_dropout),
        "loss_type": args.loss_type,
        "huber_delta": float(args.huber_delta),
        "label_transform": label_stats["label_transform"],
        "label_mu": float(label_stats["mu"]),
        "label_sd": float(label_stats["sd"]),
        "data_csv": args.data_csv,
        "split_dir": args.split_dir if args.split_dir else os.path.join(args.output_dir, "split"),
        "val_ratio": float(args.val_ratio),
        "split_seed": int(args.split_seed),
        "dedup_policy": args.dedup_policy,
        "leakage_strict": bool(args.leakage_strict),
    }

    trainer.model.save_model(args.output_dir, model_meta=meta)
    tok.save_pretrained(args.output_dir)

    cleanup_checkpoints(args.output_dir)

    print("✨ Model + tokenizer saved successfully.")
    print("✨ This exported model supports raw-activity prediction via model.predict_activity(...)")


if __name__ == "__main__":
    train(parse_args())
