
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker


# Paths

NAT_CSV = "results/motif/motif_pssm_scan_10_35_nat.csv"
GEN_CSV = "results/motif/motif_pssm_scan_10_35_gen.csv"
OUT_DIR = "results/motif/seqlogo"

os.makedirs(OUT_DIR, exist_ok=True)


# Parameters
WINDOW_LEN = 50
BASES = ["A", "C", "G", "T"]

# 固定位置
ANCHOR_35_START = -36
ANCHOR_10_START = -12

# CSV match coordinates are mirrored relative to the logo coordinates.
REVERSE_MATCH = True

Y_MAX = 1.0

COLOR_SCHEME = {
    "A": "#2ca02c",
    "C": "#1f77b4",
    "G": "#ff7f0e",
    "T": "#d62728",
}


# Coordinate conversion
def coord_to_index(coord):
    return coord + WINDOW_LEN


def clean_match(x):
    if pd.isna(x):
        return ""

    x = str(x).strip().upper()

    if x in ["", "NAN", "NONE"]:
        return ""

    x = "".join([b for b in x if b in BASES])

    if REVERSE_MATCH:
        x = x[::-1]

    return x


# Input filtering
def load_valid_rows(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = ["pos_10", "pos_35", "match_10", "match_35", "spacer_len"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    pos10 = pd.to_numeric(df["pos_10"], errors="coerce")
    pos35 = pd.to_numeric(df["pos_35"], errors="coerce")
    spacer = pd.to_numeric(df["spacer_len"], errors="coerce")

    match10 = df["match_10"].apply(clean_match)
    match35 = df["match_35"].apply(clean_match)

    mask = (
        pos10.notna()
        & pos35.notna()
        & (pos10 >= 0)
        & (pos35 >= 0)
        & spacer.notna()
        & (spacer >= 15)
        & (spacer <= 19)
        & (match10.str.len() > 0)
        & (match35.str.len() > 0)
    )

    df2 = df.loc[mask].copy()
    df2["match_10_clean"] = match10.loc[mask].values
    df2["match_35_clean"] = match35.loc[mask].values

    print(f"{os.path.basename(csv_path)}: {len(df2)} / {len(df)} used for seqlogo")

    return df2


# Reconstruct 50 bp motif-only sequences from match_35 and match_10.
def rebuild_seq_from_csv_row(row):
    out = ["N"] * WINDOW_LEN

    match35 = row["match_35_clean"]
    match10 = row["match_10_clean"]

    def place(match, start_coord):
        start_idx = coord_to_index(start_coord)

        for k, b in enumerate(match):
            idx = start_idx + k
            if 0 <= idx < WINDOW_LEN and b in BASES:
                out[idx] = b

    place(match35, ANCHOR_35_START)
    place(match10, ANCHOR_10_START)

    return "".join(out)


# Information matrix
def build_info_matrix_from_csv(csv_path, prefix):
    df = load_valid_rows(csv_path)

    seqs = [rebuild_seq_from_csv_row(row) for _, row in df.iterrows()]

    fixed_fa = os.path.join(OUT_DIR, f"{prefix}_fixed_motif_only.fa")
    with open(fixed_fa, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{prefix}_{i}\n{s}\n")

    counts = pd.DataFrame(0.0, index=range(WINDOW_LEN), columns=BASES)
    valid_n = pd.Series(0.0, index=range(WINDOW_LEN))

    for seq in seqs:
        for i, b in enumerate(seq):
            if b in BASES:
                counts.iloc[i, counts.columns.get_loc(b)] += 1
                valid_n.iloc[i] += 1

    freq = counts.copy()

    for i in range(WINDOW_LEN):
        if valid_n.iloc[i] > 0:
            freq.iloc[i, :] = counts.iloc[i, :] / valid_n.iloc[i]
        else:
            freq.iloc[i, :] = 0.0

    info = pd.DataFrame(0.0, index=range(WINDOW_LEN), columns=BASES)

    for i in range(WINDOW_LEN):
        p = freq.iloc[i].values.astype(float)

        if p.sum() <= 0:
            continue

        entropy = -np.sum(p * np.log2(p + 1e-12))
        R = 2.0 - entropy

        # 视觉压缩到 1 bits
        R = min(R, Y_MAX)

        info.iloc[i, :] = p * R

    info.index = range(-50, 0)
    freq.index = range(-50, 0)
    counts.index = range(-50, 0)

    counts.to_csv(os.path.join(OUT_DIR, f"{prefix}_counts.csv"), index_label="position")
    freq.to_csv(os.path.join(OUT_DIR, f"{prefix}_frequency.csv"), index_label="position")
    info.to_csv(os.path.join(OUT_DIR, f"{prefix}_information.csv"), index_label="position")

    return info, seqs


# Plotting
def draw_logo(ax, info_df, label=None):
    logomaker.Logo(
        info_df,
        ax=ax,
        color_scheme=COLOR_SCHEME,
        stack_order="big_on_top",
        fade_below=0,
        shade_below=0,
    )

    ax.set_xlim(-50, 0)
    ax.set_ylim(0, Y_MAX)

    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks([-50, -40, -30, -20, -10, 0])
    ax.set_yticks([0, 1])
    ax.tick_params(axis="both", length=3, width=1, labelsize=10)

    ax.set_ylabel("bits", fontsize=10)

    if label is not None:
        ax.text(-48.5, 0.78, label, fontsize=13)


def save_single(info_df, out_base_path):
    fig, ax = plt.subplots(figsize=(10, 2.0))
    draw_logo(ax, info_df, label=None)
    plt.tight_layout()

    plt.savefig(out_base_path + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_base_path + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(out_base_path + ".tif", dpi=600, format="tiff", bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )

    plt.close()


def save_combined(nat_info, gen_info, out_base_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.2), sharex=False)

    draw_logo(axes[0], nat_info, label="Nat")
    draw_logo(axes[1], gen_info, label="Gen")

    plt.tight_layout(h_pad=0.45)

    plt.savefig(out_base_path + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_base_path + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(out_base_path + ".tif", dpi=600, format="tiff", bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )

    plt.close()


# Main
def main():
    nat_info, nat_seqs = build_info_matrix_from_csv(NAT_CSV, "nat")
    gen_info, gen_seqs = build_info_matrix_from_csv(GEN_CSV, "gen")

    print(f"Nat seqlogo sequences: {len(nat_seqs)}")
    print(f"Gen seqlogo sequences: {len(gen_seqs)}")

    save_single(nat_info, os.path.join(OUT_DIR, "seqlogo_nat"))
    save_single(gen_info, os.path.join(OUT_DIR, "seqlogo_gen"))
    save_combined(nat_info, gen_info, os.path.join(OUT_DIR, "seqlogo_combined"))

    print("Done.")
    print("Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
