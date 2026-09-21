#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd

# Delay matplotlib import to avoid optional CXXABI failures.
def _lazy_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# Sequence cleaning
def _clean(s: str) -> str:
    return re.sub(r"[^ACGTacgt]", "", (s or "")).upper()


# Text decoding
def read_text_auto(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()

    # Try common encodings in order; latin-1 is the final fallback.
    encodings = ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e

    raise last_err


# K-mer frequencies
def kmer_freq(seq: str, k: int) -> dict:
    seq = seq.upper().strip()
    n = len(seq)
    if n < k:
        return {}
    cnt = {}
    total = 0
    for i in range(n - k + 1):
        kmer = seq[i:i + k]
        cnt[kmer] = cnt.get(kmer, 0) + 1
        total += 1
    return {kmer: c / total for kmer, c in cnt.items()}


# K-mer correlation
def kmer_pcc(seq1: str, seq2: str, k: int) -> float:
    f1 = kmer_freq(seq1, k)
    f2 = kmer_freq(seq2, k)
    all_keys = set(f1) | set(f2)
    if not all_keys:
        return np.nan
    x = np.array([f1.get(t, 0.0) for t in all_keys], dtype=float)
    y = np.array([f2.get(t, 0.0) for t in all_keys], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# Scatter plot
def plot_scatter(seq_g, seq_n, k, outdir):
    plt = _lazy_import_matplotlib()
    fg = kmer_freq(seq_g, k)
    fn = kmer_freq(seq_n, k)
    keys = set(fg) | set(fn)
    x = [fg.get(t, 0.0) for t in keys]
    y = [fn.get(t, 0.0) for t in keys]
    if not x:
        return
    m = max(max(x), max(y))
    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(x, y, s=12, alpha=0.6)
    plt.plot([0, m], [0, m], "r--", lw=1)
    plt.xlim(0, m)
    plt.ylim(0, m)
    plt.xlabel("Generated promoter k-mer frequency")
    plt.ylabel("Natural promoter k-mer frequency")
    plt.title(f"{k}-mer frequency comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{k}-mer_scatter.png"), dpi=300)
    plt.close()


# Summary plot
def plot_pcc_curve(ks, pccs, out_png, title="PCC vs k-mer"):
    plt = _lazy_import_matplotlib()
    plt.figure(figsize=(8, 4.6))
    plt.plot(ks, pccs, marker="o", linewidth=2)
    for k, r in zip(ks, pccs):
        s = "nan" if np.isnan(r) else f"{r:.3f}"
        if not np.isnan(r):
            plt.text(k, r, s, ha="center", va="bottom", fontsize=10)
    plt.xlabel(r"$k$-mer")
    plt.ylabel("PCC")
    plt.title(title)
    plt.grid(alpha=0.25, linestyle="--")
    plt.ylim(0.0, 1.02)
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_png.replace(".png", ".svg"))
    plt.close()



if __name__ == "__main__":
    nat_path = "datasets/Scoli/dataset.txt"
    gen_path = "outputs/GENERator/noft/2090/TXT/generated_20260429-162839.txt"
    outdir   = "outputs/Scoli/result"

    os.makedirs(outdir, exist_ok=True)

    gen_seq = _clean(read_text_auto(gen_path))
    nat_seq = _clean(read_text_auto(nat_path))

    print(f"[INFO] gen_len={len(gen_seq)} nat_len={len(nat_seq)}")

    ks = list(range(1, 8))
    pccs = []
    for k in ks:
        r = kmer_pcc(gen_seq, nat_seq, k)
        pccs.append(r)
        print(f"k={k}: PCC={r:.4f}" if not np.isnan(r) else f"k={k}: PCC=nan")

    df = pd.DataFrame({"k": ks, "PCC": pccs})
    csv_path = os.path.join(outdir, "kmer_1to8_pcc_summary.csv")
    df.to_csv(csv_path, index=False)

    png_path = os.path.join(outdir, "kmer_1to8_pcc_curve.png")
    try:
        plot_pcc_curve(ks, pccs, png_path, title="k-mer PCC (Generated vs Natural)")
        print(f"✅ PCC curve: {png_path}")
    except Exception as e:
        print(f"[WARN] plot skipped due to matplotlib error: {e}")

    print(f"\n✅ CSV: {csv_path}")
