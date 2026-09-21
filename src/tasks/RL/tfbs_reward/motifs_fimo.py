#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
如直接使用：

python motifs_fimo.py \
  --seq-csv /path/to/your.csv \
  --seq-col sequence \
  --id-col SeqID \
  --has-header \
  --meme /path/to/motifs.meme \
  --fimo-thr 1e-2 \
  --both-strands \
  --out-sites /path/to/sites_long.csv \
  --out-counts /path/to/tfbs_counts.csv

'''

import argparse
import hashlib
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import anndata  # 仅用于把矩阵装进 AnnData；没装也能跑其它功能
    _HAS_ANNDATA = True
except Exception:
    _HAS_ANNDATA = False

from pymemesuite.common import MotifFile, Sequence, Alphabet, Background
from pymemesuite.fimo import FIMO


# I/O

def _decode(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode(errors="ignore")
    return str(x)

def read_meme(meme_path):

    motiffile = MotifFile(meme_path)
    motifs = list(motiffile)

    try:
        bg = motiffile.background
    except Exception as e:
        print(f"[MEME][WARN] background not found or invalid: {e} -> use uniform 0.25")

        try:
            alphabet = getattr(motiffile, "alphabet", None)
            if alphabet is None:
                alphabet = Alphabet.dna()
        except Exception:
            alphabet = Alphabet.dna()

        # Build a uniform DNA background with the pymemesuite API.
        bg = Background.from_uniform(alphabet)

    return motifs, bg


def _ensure_seqid_column(df: pd.DataFrame, id_col: str, seq_col: str) -> pd.DataFrame:
    """确保存在 id_col；若无则用 行号+序列MD5 生成稳定唯一的 SeqID。"""
    df = df.copy()
    if id_col not in df.columns:
        def make_id(seq, i):
            h = hashlib.md5(str(seq).encode()).hexdigest()[:10]
            return f"seq_{i:06d}_{h}"
        df.insert(0, id_col, [make_id(s, i) for i, s in enumerate(df[seq_col])])
    return df


def _clean_sequence_col(df: pd.DataFrame, seq_col: str) -> pd.DataFrame:
    """大写化、去空白、把非ACGT字符置为N。"""
    df = df.copy()
    df[seq_col] = (
        df[seq_col]
        .astype(str)
        .str.upper()
        .str.replace(r"[^ACGT]", "N", regex=True)
        .str.strip()
    )
    return df


# FIMO scanning

def scan(
    seq_df,
    motifs,
    bg,
    threshold: float = 1e-3,
    id_col: str = "SeqID",
    seq_col: str = "sequence",
    both_strands: bool = True,
) -> pd.DataFrame:
    """
    FIMO 扫描主函数（兼容旧版 pymemesuite：Sequence 不带 accession）。
    返回长表：SeqID / Matrix_id / start / end / strand / score / pval / qval
    """
    if isinstance(seq_df, pd.DataFrame):
        df = seq_df.copy()
    else:
        df = pd.DataFrame({seq_col: list(seq_df)})

    if seq_col not in df.columns:
        raise ValueError(f"输入缺少序列列 `{seq_col}`")

    df[seq_col] = (
        df[seq_col].astype(str).str.upper().str.replace(r"[^ACGT]", "N", regex=True).str.strip()
    )
    if id_col not in df.columns:
        df.insert(0, id_col, [f"{i}" for i in range(len(df))])

    # Older pymemesuite versions do not accept an accession argument.
    sequences = [
        Sequence(row[seq_col], name=str(row[id_col]).encode())
        for _, row in df.iterrows()
    ]

    fimo = FIMO(both_strands=both_strands, threshold=threshold)

    d = defaultdict(list)
    for motif in tqdm(motifs):
        if "_printed_motif_debug" not in scan.__dict__:
            scan._printed_motif_debug = True
            print("[DBG-MOTIF] motif.name=", repr(getattr(motif, "name", None)),
                  " motif.accession=", repr(getattr(motif, "accession", None)))
        # Prefer the motif name, then accession, then a generated identifier.
        motif_name = _decode(getattr(motif, "name", None)).strip()
        if motif_name == "":
            motif_name = _decode(getattr(motif, "accession", None)).strip()
        if motif_name == "":
            motif_name = "UNKNOWN_MOTIF"

        res = fimo.score_motif(motif, sequences, bg)

        for m in res.matched_elements:
            sid = getattr(m.source, "name", None)
            if isinstance(sid, bytes):
                sid = sid.decode()
            if not sid:
                sid = getattr(m.source, "accession", None)
                if isinstance(sid, bytes):
                    sid = sid.decode()
            if not sid:
                sid = "0"

            d["Matrix_id"].append(motif_name)
            d["SeqID"].append(str(sid))
            d["strand"].append(m.strand)
            d["score"].append(m.score)
            d["pval"].append(m.pvalue)
            d["qval"].append(m.qvalue)
            if m.strand == "-":
                d["start"].append(int(m.stop))
                d["end"].append(int(m.start))
            else:
                d["start"].append(int(m.start))
                d["end"].append(int(m.stop))

    if len(d) == 0:
        sites_df = pd.DataFrame(
            columns=["SeqID", "Matrix_id", "start", "end", "strand", "score", "pval", "qval"]
        )
        return sites_df

    sites_df = pd.DataFrame(d)

    need = {"SeqID", "Matrix_id", "start", "end"}
    missing = need - set(sites_df.columns)
    if missing:
        # Preserve the RL run when optional result columns are absent.
        print(f"[FIMO][WARN] scan 结果缺少列：{missing}，返回空表")
        return pd.DataFrame(
            columns=["SeqID", "Matrix_id", "start", "end", "strand", "score", "pval", "qval"]
        )

    return sites_df



def calculate_motif_counts(sites_df: pd.DataFrame):
    """
    由长表生成计数矩阵（Seq × Motif）。
    返回：
      - 若安装了 anndata：AnnData 对象（.X 是矩阵）
      - 否则：直接返回 pandas.DataFrame
    """
    mat = (
        pd.pivot_table(
            sites_df,
            values="start",
            index="SeqID",
            columns="Matrix_id",
            aggfunc="count",
        )
        .fillna(0)
        .astype(int)
    )
    if _HAS_ANNDATA:
        return anndata.AnnData(mat)
    return mat


# Analysis helpers

def accuracy_by_motif(seqs: pd.DataFrame, sites: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    给定:
      seqs: 包含列 ['Sequence','acc']（acc 是逐位准确率向量，list/np.array）
      sites: scan() 返回的长表
    输出：每条序列的 “In Motif / Out of Motif” 平均准确率
    """
    sites = sites[sites["SeqID"].isin(seqs.index)]
    sites = sites.copy()
    sites["positions"] = sites.apply(lambda r: set(range(r.start, r.end)), axis=1)

    positions = pd.DataFrame(
        sites.groupby("SeqID").positions.apply(list).apply(lambda xs: set().union(*xs))
    )
    seqs = seqs.merge(positions, left_index=True, right_on="SeqID", how="left")
    seqs["positions"] = seqs["positions"].fillna("").apply(set)
    seqs["all_positions"] = [set(range(len(seq))) for seq in seqs.Sequence]
    seqs["negative_positions"] = seqs.apply(
        lambda row: row["all_positions"].difference(row["positions"]), axis=1
    )
    seqs["In Motif"] = seqs.apply(lambda row: np.mean([row.acc[i] for i in row.positions]) if row.positions else np.nan, axis=1)
    seqs["Out of Motif"] = seqs.apply(
        lambda row: np.mean([row.acc[i] for i in row.negative_positions]) if row.negative_positions else np.nan, axis=1
    )
    return seqs


def generate_tfbs_structure(tfbs_sites: pd.DataFrame):
    """
    将长表转为 {SeqID: [{Matrix_id,start,end}, ...]} 的结构列表，按 SeqID 排序。
    """
    result = []
    for seq_id, group in tfbs_sites.groupby("SeqID"):
        group_sorted = group.sort_values(["start", "end"])
        tfbs_list = [
            {"Matrix_id": r["Matrix_id"], "start": int(r["start"]), "end": int(r["end"])}
            for _, r in group_sorted.iterrows()
        ]
        result.append({str(seq_id): tfbs_list})
    result = sorted(result, key=lambda x: list(x.keys())[0])
    return result


# CLI

def _load_sequences_csv(path: str, seq_col: str, id_col: str, has_header: bool = True) -> pd.DataFrame:

    if has_header:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None, names=[seq_col, "expression"])

    if seq_col not in df.columns:
        raise ValueError(f"CSV 中不存在列 `{seq_col}`")

    df = _clean_sequence_col(df, seq_col)
    df = _ensure_seqid_column(df, id_col, seq_col)
    return df


def main():
    ap = argparse.ArgumentParser(description="TFBS 扫描（MEME+FIMO）端到端脚本")
    ap.add_argument("--seq-csv", required=True, help="输入 CSV 路径（至少包含一列序列）")
    ap.add_argument("--seq-col", default="sequence", help="序列列名（默认 sequence）")
    ap.add_argument("--id-col", default="SeqID", help="ID 列名（默认 SeqID）")
    ap.add_argument("--has-header", action="store_true", help="CSV 是否有表头（默认无则用两列 sequence,expression）")
    ap.add_argument("--meme", required=True, help="MEME/JASPAR motif 文件路径")
    ap.add_argument("--fimo-thr", type=float, default=1e-2, help="FIMO 阈值（默认 1e-2）")
    ap.add_argument("--both-strands", action="store_true", help="同时扫描两条链（默认 False）")
    ap.add_argument("--out-sites", default=None, help="输出长表 CSV（默认：seq-csv 同目录 sites_long.csv）")
    ap.add_argument("--out-counts", default=None, help="输出计数矩阵 CSV（默认：seq-csv 同目录 tfbs_counts.csv）")

    args = ap.parse_args()

    df = _load_sequences_csv(args.seq_csv, args.seq_col, args.id_col, has_header=args.has_header)

    motifs, bg = read_meme(args.meme)
    print(f"[MEME] motifs_in={len(motifs)}  bg={bg}")

    sites = scan(
        seq_df=df,
        motifs=motifs,
        bg=bg,
        threshold=args.fimo_thr,
        id_col=args.id_col,
        seq_col=args.seq_col,
        both_strands=args.both_strands,
    )

    base_dir = os.path.dirname(os.path.abspath(args.seq_csv))
    out_sites = args.out_sites or os.path.join(base_dir, "sites_long.csv")
    out_counts = args.out_counts or os.path.join(base_dir, "tfbs_counts.csv")

    sites.to_csv(out_sites, index=False)
    print(f"[OK] 写出扫描结果：{out_sites}  （{len(sites)} rows）")

    counts_obj = calculate_motif_counts(sites)
    if _HAS_ANNDATA:
        counts_df = pd.DataFrame(
            counts_obj.X, index=counts_obj.obs_names, columns=counts_obj.var_names
        ).astype(int)
    else:
        counts_df = counts_obj
    counts_df.index.name = args.id_col
    counts_df.to_csv(out_counts)
    print(f"[OK] 写出计数矩阵：{out_counts}  （{counts_df.shape[0]}×{counts_df.shape[1]}）")


if __name__ == "__main__":
    main()
