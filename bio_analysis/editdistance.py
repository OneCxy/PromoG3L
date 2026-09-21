#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Configuration
# 输入文件：支持 txt / csv

NATURAL_FILE_PATH = "datasets/Scoli/Scoli_165.txt"

GENERATED_FILE_PATH = "outputs/Scoli/result/RL_base/TXT/generated.txt"

RANDOM_FILE_PATH = "datasets/randoms/scoli/scoli.txt"


OUT_DIR = "outputs/Scoli/result/RL_base/editdistance"

OUT_PREFIX = "Sco"

PLOT_TITLE = "Sco"

SAMPLE_SIZE = 100

RANDOM_SEED = 2025

KDE_POINTS = 400

HIST_BIN_WIDTH = 2

SAVE_DPI = 600

NAT_LABEL = "Nat"
GEN_LABEL = "Gen"
RAND_LABEL = "Rand"

COLOR_NAT = "#4C78A8"      # 蓝
COLOR_RAND = "#E0C36E"     # 黄
COLOR_GEN = "#9A89C9"      # 紫

COLOR_NAT_GEN = "#C6D180"   # 浅绿
COLOR_NAT_RAND = "#E3716E"  # 红橙
COLOR_GEN_RAND = "#4C78A8"  # 蓝


def detect_seq_col(df):
    candidates = [
        "sequence", "seq", "dna_sequence", "dna_seq",
        "Sequence", "SEQ", "promoter", "Promoter"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"找不到序列列。当前 CSV 列名为: {list(df.columns)}")


def clean_seq(s):
    s = str(s).upper().strip()
    s = "".join([x for x in s if x in "ACGTN"])
    return s


def read_sequences(file_path, sample_size=None, seed=2025):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
        seq_col = detect_seq_col(df)
        seqs = [clean_seq(x) for x in df[seq_col].tolist()]
    else:
        seqs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(">"):
                    continue
                first_col = line.split()[0]
                seqs.append(clean_seq(first_col))

    seqs = [s for s in seqs if len(s) > 0]

    if len(seqs) == 0:
        raise ValueError(f"没有读取到有效序列: {file_path}")

    if sample_size is not None and len(seqs) > sample_size:
        rng = random.Random(seed)
        seqs = rng.sample(seqs, sample_size)

    lengths = [len(s) for s in seqs]
    print(f"Read {len(seqs)} sequences from {file_path}")
    print(f"Length range: min={min(lengths)}, max={max(lengths)}")

    return seqs


def edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)

    if m == 0:
        return n
    if n == 0:
        return m

    prev = np.arange(n + 1, dtype=np.int32)
    curr = np.zeros(n + 1, dtype=np.int32)

    for i in range(1, m + 1):
        curr[0] = i
        c1 = seq1[i - 1]

        for j in range(1, n + 1):
            c2 = seq2[j - 1]
            cost = 0 if c1 == c2 else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost  # substitution
            )

        prev, curr = curr, prev

    return int(prev[n])


def calculate_intragroup_distances(sequences, group_name):
    """
    组内编辑距离：
    只计算 i < j，避免自己和自己比较，也避免重复计算。
    """
    distances = []
    n = len(sequences)

    for i in range(n):
        for j in range(i + 1, n):
            d = edit_distance(sequences[i], sequences[j])
            distances.append(d)

    distances = np.array(distances, dtype=float)
    print(f"{group_name} intragroup distances: n_pairs={len(distances)}")
    return distances


def calculate_intergroup_distances(sequences_a, sequences_b, group_name):
    """
    组间编辑距离：
    计算 A 组每条序列与 B 组每条序列之间的编辑距离。
    """
    distances = []

    for seq_a in sequences_a:
        for seq_b in sequences_b:
            d = edit_distance(seq_a, seq_b)
            distances.append(d)

    distances = np.array(distances, dtype=float)
    print(f"{group_name} intergroup distances: n_pairs={len(distances)}")
    return distances


def print_stats(name, values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        print(f"{name}: empty")
        return

    print(
        f"{name}: "
        f"mean={values.mean():.3f}, "
        f"median={np.median(values):.3f}, "
        f"std={values.std(ddof=1):.3f}, "
        f"min={values.min():.3f}, "
        f"max={values.max():.3f}"
    )


def kde_curve(values, x_grid=None, bw=None):
    """
    简单 Gaussian KDE，不依赖 seaborn/scipy。
    用于画组间的平滑曲线图。
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return np.array([]), np.array([])

    if x_grid is None:
        xmin = np.floor(values.min() - 5)
        xmax = np.ceil(values.max() + 5)
        x_grid = np.linspace(xmin, xmax, KDE_POINTS)

    if bw is None:
        std = values.std(ddof=1)
        n = len(values)
        if std <= 0:
            bw = 1.0
        else:
            bw = 1.06 * std * (n ** (-1 / 5))
            bw = max(bw, 1.0)

    diff = (x_grid[:, None] - values[None, :]) / bw
    y = np.exp(-0.5 * diff ** 2).sum(axis=1)
    y = y / (len(values) * bw * np.sqrt(2 * np.pi))

    return x_grid, y


def save_both_formats(fig, out_base_path, dpi=600):

    png_path = out_base_path + ".png"
    tif_path = out_base_path + ".tif"

    fig.savefig(png_path, dpi=dpi, format="png", bbox_inches="tight")
    fig.savefig(
        tif_path,
        dpi=dpi,
        format="tiff",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {tif_path}")


def plot_intragroup_hist(series, title, out_base_path):
    """
    组内图：
    Nat / Rand / Gen 各自内部编辑距离
    画成重叠直方图
    """
    fig = plt.figure(figsize=(5.2, 4.2))
    ax = plt.gca()

    all_values = []
    for values, _, _ in series:
        values = np.asarray(values, dtype=float)
        if values.size > 0:
            all_values.append(values)

    all_values = np.concatenate(all_values)

    x_min = max(0, np.floor(all_values.min() - 5))
    x_max = np.ceil(all_values.max() + 5)
    bins = np.arange(x_min, x_max + HIST_BIN_WIDTH, HIST_BIN_WIDTH)

    for values, label, color in series:
        values = np.asarray(values, dtype=float)
        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.45,
            label=label,
            color=color,
            edgecolor="none"
        )

    ax.set_title(title, fontsize=14, fontstyle="italic")
    ax.set_xlabel("Edit Distance(intragroup)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)

    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_both_formats(fig, out_base_path, dpi=SAVE_DPI)
    plt.close(fig)


def plot_intergroup_curve(series, title, out_base_path):
    """
    组间图：
    Nat-Gen / Nat-Rand / Gen-Rand
    画成平滑曲线图
    """
    fig = plt.figure(figsize=(5.2, 4.2))
    ax = plt.gca()

    all_values = []
    for values, _, _ in series:
        values = np.asarray(values, dtype=float)
        if values.size > 0:
            all_values.append(values)

    all_values = np.concatenate(all_values)

    x_min = max(0, np.floor(all_values.min() - 5))
    x_max = np.ceil(all_values.max() + 5)
    x_grid = np.linspace(x_min, x_max, KDE_POINTS)

    for values, label, color in series:
        values = np.asarray(values, dtype=float)
        x, y = kde_curve(values, x_grid=x_grid, bw=None)
        ax.plot(
            x, y,
            label=label,
            color=color,
            linewidth=1.8
        )

    ax.set_title(title, fontsize=14, fontstyle="italic")
    ax.set_xlabel("Edit Distance(groups)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_both_formats(fig, out_base_path, dpi=SAVE_DPI)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    natural_sequences = read_sequences(
        NATURAL_FILE_PATH,
        sample_size=SAMPLE_SIZE,
        seed=RANDOM_SEED
    )

    generated_sequences = read_sequences(
        GENERATED_FILE_PATH,
        sample_size=SAMPLE_SIZE,
        seed=RANDOM_SEED + 1
    )

    random_sequences = read_sequences(
        RANDOM_FILE_PATH,
        sample_size=SAMPLE_SIZE,
        seed=RANDOM_SEED + 2
    )

    # Within-group distances
    natural_intra = calculate_intragroup_distances(
        natural_sequences,
        NAT_LABEL
    )

    generated_intra = calculate_intragroup_distances(
        generated_sequences,
        GEN_LABEL
    )

    random_intra = calculate_intragroup_distances(
        random_sequences,
        RAND_LABEL
    )

    # Between-group distances
    natural_generated_inter = calculate_intergroup_distances(
        natural_sequences,
        generated_sequences,
        f"{NAT_LABEL}-{GEN_LABEL}"
    )

    natural_random_inter = calculate_intergroup_distances(
        natural_sequences,
        random_sequences,
        f"{NAT_LABEL}-{RAND_LABEL}"
    )

    generated_random_inter = calculate_intergroup_distances(
        generated_sequences,
        random_sequences,
        f"{GEN_LABEL}-{RAND_LABEL}"
    )

    print("\n========== Intragroup Edit Distance ==========")
    print_stats(f"{NAT_LABEL} intragroup", natural_intra)
    print_stats(f"{GEN_LABEL} intragroup", generated_intra)
    print_stats(f"{RAND_LABEL} intragroup", random_intra)

    print("\n========== Intergroup Edit Distance ==========")
    print_stats(f"{NAT_LABEL}-{GEN_LABEL}", natural_generated_inter)
    print_stats(f"{NAT_LABEL}-{RAND_LABEL}", natural_random_inter)
    print_stats(f"{GEN_LABEL}-{RAND_LABEL}", generated_random_inter)

    intra_base = os.path.join(OUT_DIR, f"{OUT_PREFIX}_EditDistance_Intragroup")

    plot_intragroup_hist(
        series=[
            (natural_intra, NAT_LABEL, COLOR_NAT),
            (random_intra, RAND_LABEL, COLOR_RAND),
            (generated_intra, GEN_LABEL, COLOR_GEN),
        ],
        title=PLOT_TITLE,
        out_base_path=intra_base
    )

    inter_base = os.path.join(OUT_DIR, f"{OUT_PREFIX}_EditDistance_groups")

    plot_intergroup_curve(
        series=[
            (natural_generated_inter, f"{NAT_LABEL}-{GEN_LABEL}", COLOR_NAT_GEN),
            (natural_random_inter, f"{NAT_LABEL}-{RAND_LABEL}", COLOR_NAT_RAND),
            (generated_random_inter, f"{GEN_LABEL}-{RAND_LABEL}", COLOR_GEN_RAND),
        ],
        title=PLOT_TITLE,
        out_base_path=inter_base
    )


if __name__ == "__main__":
    main()
