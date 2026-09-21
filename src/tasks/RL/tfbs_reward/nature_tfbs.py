#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# 扫描序列并聚合全局 TFBS 基准向量（1×TF）


import argparse, json, os, re, sys, math
from pathlib import Path, PurePosixPath, PureWindowsPath
import numpy as np
import pandas as pd
from typing import List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
from motifs_fimo import read_meme, scan

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("nature_tfbs_config.json")


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

    for key in ("seq_csv", "meme", "out", "global_out_csv", "global_out_npy"):
        value = config.get(key)
        if value and not _is_absolute_path(value):
            config[key] = str(PROJECT_ROOT / value)

    print(f"[CONFIG] Loaded: {config_path}")
    return config

ACGT_RE = re.compile(r'[^ACGT]')

def normalize_seq(s: str) -> str:
    return ACGT_RE.sub('', (s or '').upper())

_COMP = str.maketrans("ACGT", "TGCA")
def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]

def norm_motif_keys(m) -> set:
    """
    生成若干匹配键，尽力兼容 JASPAR 命名：
    例：MA0139.1_CTCF -> {'MA0139.1_CTCF','MA0139.1','CTCF','MA0139'}
    """
    name = m.name.decode() if isinstance(m.name, bytes) else str(m.name)
    name = (name or "").strip()
    keys = {name}
    if '_' in name:
        acc, gene = name.split('_', 1)
        keys.add(acc.strip()); keys.add(gene.strip())
    if '.' in name:
        keys.add(name.split('.', 1)[0].strip())
    for attr in ("accession", "acc", "id"):
        if hasattr(m, attr):
            val = getattr(m, attr)
            try:
                val = val.decode() if isinstance(val, bytes) else str(val)
            except Exception:
                pass
            if val:
                keys.add(val.strip())
                if '.' in val:
                    keys.add(val.split('.', 1)[0].strip())
    return {k for k in keys if k}

def pick_motifs(motifs, selected_ids: List[str]):
    sel = {str(x).strip() for x in selected_ids}
    buckets = [(m, norm_motif_keys(m)) for m in motifs]
    picked = [m for (m, keys) in buckets if (sel & keys)]
    return picked

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def coerce_col(df: pd.DataFrame, prefer: str, candidates: List[str]) -> Optional[str]:
    """在 df 里找一个合适的列名映射到 prefer。"""
    if prefer in df.columns:
        return prefer
    for c in candidates:
        if c in df.columns:
            return c
    return None

def debug_sites_preview(sites: pd.DataFrame, k: int = 5):
    if sites is None or sites.empty:
        print("[DBG] sites empty")
        return
    cols = list(sites.columns)
    print(f"[DBG] sites columns: {cols[:10]}{'...' if len(cols)>10 else ''}")
    print(sites.head(k))

def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, remaining_args = config_parser.parse_known_args()

    config_path = config_args.config
    if config_path is None and not remaining_args:
        config_path = str(DEFAULT_CONFIG_PATH)
    config = _load_config(config_path) if config_path else {}

    ap = argparse.ArgumentParser("Scan natural sequences TFBS → global TF vector (1×TF) — memory-lite")
    ap.add_argument("--seq-csv", required=True)
    ap.add_argument("--seq-col", default=None,
                    help="有表头时可指定列名；不指定时默认使用第一列")
    ap.add_argument("--no-header", action="store_true",
                    help="输入 CSV 没有表头；此时默认第一列为 sequence")
    ap.add_argument("--meme", required=True)
    ap.add_argument("--out", default=None, help="方案A已禁用逐序列矩阵输出，此参数忽略，仅提示")
    ap.add_argument("--fimo-thr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--scan-rc", action="store_true")
    ap.add_argument("--debug-k", type=int, default=0,
                    help=">0 时只用前K条序列，且暂时把阈值放宽到 max(fimo_thr, 1e-1) 做连通性检查")
    ap.add_argument("--global-out-csv", default=None)
    ap.add_argument("--global-out-npy", default=None)
    ap.add_argument("--reduce", default="sum", choices=["sum","mean","presence-mean"])
    ap.add_argument("--l1-normalize", action="store_true")
    ap.add_argument("--zscore", action="store_true")
    boolean_flags = {"no_header", "scan_rc", "l1_normalize", "zscore"}
    config_cli_args = []
    for key, value in config.items():
        option = f"--{key.replace('_', '-')}"
        if key in boolean_flags:
            if value:
                config_cli_args.append(option)
        elif value is not None:
            config_cli_args.extend([option, str(value)])

    args = ap.parse_args(config_cli_args + remaining_args)

    if args.out:
        print("ℹ️ [方案A] 忽略 --out：本版本不生成 (Seq × TF) 矩阵，仅输出全局 1×TF 向量。")

    read_kwargs = {}
    if args.no_header:
        read_kwargs["header"] = None
    df = pd.read_csv(args.seq_csv, **read_kwargs)

    if args.no_header:
        seq_series = df.iloc[:, 0]
        print(f"[SEQ] no-header 模式，使用第 1 列作为 sequence，n={len(seq_series)}")
    else:
        col = args.seq_col or df.columns[0]    # 有表头时，默认第一列；也可以用 --seq-col 指定
        seq_series = df[col]
        print(f"[SEQ] 使用列 '{col}' 作为 sequence，n={len(seq_series)}")

    seqs = [normalize_seq(x) for x in seq_series.astype(str).tolist()]
    if args.debug_k and args.debug_k < len(seqs):
        seqs = seqs[:args.debug_k]
        print(f"[DBG] debug_k={args.debug_k}, use first {len(seqs)} sequences")
    N = len(seqs)
    print(f"[SEQ] n={N}")

    motifs, bg = read_meme(args.meme)
    print(f"[MEME] motifs_in={len(motifs)}  bg={bg}")

    picked = motifs
    print(f"[FILTER] 使用 MEME 文件中的全部 TFBS motifs，picked_motifs={len(picked)}")

    B = args.batch
    n_batches = math.ceil(N / B)
    thr = max(args.fimo_thr, 1e-1) if args.debug_k else args.fimo_thr
    print(f"[SCAN] threshold={thr} (debug_k={args.debug_k})  motifs_used={len(picked)}")

    global_counts = {}    # {Matrix_id: 总命中次数}
    global_presence = {}  # {Matrix_id: 至少一次命中的序列条数}

    for bi, start in enumerate(range(0, N, B), 1):
        end = min(start + B, N)
        sub = seqs[start:end]
        ext = sub + [revcomp(s) for s in sub] if args.scan_rc else sub

        sites = scan(ext, picked, bg, threshold=thr, both_strands=(not args.scan_rc))

        if sites is not None and not sites.empty:
            seq_col = coerce_col(sites, "SeqID", ["SeqID", "seq_id", "sequence_id", "seq", "id"])
            if seq_col is None:
                raise ValueError("scan 返回里找不到 SeqID 列")
            sites[seq_col] = sites[seq_col].astype(int)

            # 折返 RC 扩展
            if args.scan_rc and len(ext) == 2 * len(sub):
                sites.loc[sites[seq_col] >= len(sub), seq_col] -= len(sub)
            sites[seq_col] += start

            motif_col = coerce_col(sites, "Matrix_id",
                                   ["Matrix_id", "motif_id", "Motif_ID", "matrix_id", "name", "id", "acc", "accession"])
            if motif_col is None:
                raise ValueError("scan 返回里找不到 motif 列（Matrix_id/motif_id/...）")
            if motif_col != "Matrix_id":
                sites = sites.rename(columns={motif_col: "Matrix_id"})

            if bi == 1:
                print(f"[DBG] first-batch sites rows={len(sites)}")
                debug_sites_preview(sites, k=5)

            # 1) 总命中次数（reduce=sum / mean）
            cnt = sites["Matrix_id"].value_counts()
            for k, v in cnt.items():
                global_counts[k] = global_counts.get(k, 0) + int(v)

            # 2) 出现过的序列条数（reduce=presence-mean）
            pres = sites.drop_duplicates([seq_col, "Matrix_id"])["Matrix_id"].value_counts()
            for k, v in pres.items():
                global_presence[k] = global_presence.get(k, 0) + int(v)

            sites_rows = int(len(sites))
            del cnt, pres, sites
        else:
            sites_rows = 0

        import gc; gc.collect()
        tf_cols_now = len(set(global_counts.keys()) | set(global_presence.keys()))
        print(f"[BATCH] {bi}/{n_batches}  sites_rows={sites_rows}  uniq_seqs={end - start}  TF_seen={tf_cols_now}")

    if not args.global_out_csv and not args.global_out_npy:
        print("⚠️ 未指定 --global-out-csv / --global-out-npy，已完成扫描但不会写出全局向量。")
        return

    all_tfs = sorted(set(global_counts) | set(global_presence))

    if len(all_tfs) == 0:
        raise ValueError("❌ 所有批次均无 motif 命中，无法生成全局向量。请放宽阈值或检查 MEME 文件。")

    if args.reduce in ("sum", "mean"):
        vec = np.array([global_counts.get(tf, 0) for tf in all_tfs], dtype=np.float64)
        if args.reduce == "mean":
            vec = vec / float(N)  # 每序列平均次数
    else:  # presence-mean
        vec = np.array([global_presence.get(tf, 0) for tf in all_tfs], dtype=np.float64) / float(N)

    if args.zscore:
        mu, sd = vec.mean(), vec.std()
        if sd > 0:
            vec = (vec - mu) / sd
    if args.l1_normalize:
        s = np.abs(vec).sum()
        if s > 0:
            vec = vec / s

    if args.global_out_csv:
        global_row = pd.DataFrame([vec], columns=all_tfs)
        ensure_dir(args.global_out_csv)
        global_row.to_csv(args.global_out_csv, index=False)
        nz = (vec != 0)
        if nz.sum() > 0:
            top_idx = np.argsort(-vec)[:5]
            tops = [(all_tfs[i], float(vec[i])) for i in top_idx]
            print("[INFO] global top5 TF:", tops)

        print(f"[OK] saved global vector -> {args.global_out_csv}  shape={(1, len(all_tfs))}")

    if args.global_out_npy:
        ensure_dir(args.global_out_npy)
        np.save(args.global_out_npy, vec)
        print(f"[OK] saved global vector npy -> {args.global_out_npy}")

if __name__ == "__main__":
    main()
