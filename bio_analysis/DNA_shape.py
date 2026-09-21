#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Input files
shape_configs = {
    "ROLL": {
        "Rand": "outputs/DNAshape/Ecoli/random519/random.Roll",
        "Nat": "outputs/DNAshape/Ecoli/nature/low/low.Roll",
        "Gen": "outputs/DNAshape/Ecoli/generate_RL/low/phpXE0iwF.Roll",
    },
    "MGW": {
        "Rand": "outputs/DNAshape/Ecoli/random519/random.MGW",
        "Nat": "outputs/DNAshape/Ecoli/nature/low/low.MGW",
        "Gen": "outputs/DNAshape/Ecoli/generate_RL/low/phpXE0iwF.MGW",
    },
    "HELT": {
        "Rand": "outputs/DNAshape/Ecoli/random519/random.HelT",
        "Nat": "outputs/DNAshape/Ecoli/nature/low/low.HelT",
        "Gen": "outputs/DNAshape/Ecoli/generate_RL/low/phpXE0iwF.HelT",
    },
    "PROT": {
        "Rand": "outputs/DNAshape/Ecoli/random519/random.ProT",
        "Nat": "outputs/DNAshape/Ecoli/nature/low/low.ProT",
        "Gen": "outputs/DNAshape/Ecoli/generate_RL/low/phpXE0iwF.ProT",
    },
}

# Output
save_dir = "outputs/DNAshape/Ecoli/result/generate_RL/low"
os.makedirs(save_dir, exist_ok=True)

stats_dir = os.path.join(save_dir, "curve_stats")
os.makedirs(stats_dir, exist_ok=True)

save_prefix = "Dnashape"

# KDE grid
grid_x = np.linspace(0, 1, 400)


def parse_single_group_numeric_file(path):

    data = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue

            parts = re.split(r"[,\t ]+", line)

            start = 0
            try:
                float(parts[0])
            except ValueError:
                start = 1

            for p in parts[start:]:
                if (not p) or (p.upper() == "NA"):
                    continue
                try:
                    data.append(float(p))
                except ValueError:
                    pass

    return data


def normalize_data(data):
    if len(data) == 0:
        return np.array([])

    min_val = min(data)
    max_val = max(data)

    if max_val == min_val:
        return np.zeros(len(data))

    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return np.array(normalized_data, dtype=float)


def safe_kde_on_grid(data_norm, grid):
    
    if data_norm is None or len(data_norm) <= 1:
        return np.full_like(grid, np.nan, dtype=float)

    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        sns.kdeplot(data_norm, ax=ax, linewidth=1)
        lines = ax.get_lines()
        if len(lines) == 0:
            plt.close(fig)
            return np.full_like(grid, np.nan, dtype=float)

        x = lines[0].get_xdata()
        y = lines[0].get_ydata()

        if len(x) < 2:
            plt.close(fig)
            return np.full_like(grid, np.nan, dtype=float)

        # 限制在 [0, 1]
        mask = (x >= 0) & (x <= 1)
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            plt.close(fig)
            return np.full_like(grid, np.nan, dtype=float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        y_interp = np.interp(grid, x, y, left=np.nan, right=np.nan)
        plt.close(fig)
        return y_interp

    except Exception:
        plt.close(fig)
        return np.full_like(grid, np.nan, dtype=float)


for shape_name, cfg in shape_configs.items():
    print("\n" + "=" * 100)
    print(f"[INFO] Processing shape: {shape_name}")
    print("=" * 100)

    rand_file = cfg["Rand"]
    nat_file = cfg["Nat"]
    gen_file = cfg["Gen"]

    random_data = parse_single_group_numeric_file(rand_file)
    natural_data = parse_single_group_numeric_file(nat_file)
    generated_data = parse_single_group_numeric_file(gen_file)

    print(f"[INFO] {shape_name} Random values: {len(random_data)}")
    print(f"[INFO] {shape_name} Natural values: {len(natural_data)}")
    print(f"[INFO] {shape_name} Generated values: {len(generated_data)}")

    random_norm = normalize_data(random_data)
    natural_norm = normalize_data(natural_data)
    generated_norm = normalize_data(generated_data)

    rnd_kde = safe_kde_on_grid(random_norm, grid_x)
    nat_kde = safe_kde_on_grid(natural_norm, grid_x)
    gen_kde = safe_kde_on_grid(generated_norm, grid_x)

    plt.figure(figsize=(8, 6))

    if np.isfinite(gen_kde).any():
        plt.plot(grid_x, gen_kde, color="blue", label="Generated", linewidth=6, linestyle="-")
    if np.isfinite(nat_kde).any():
        plt.plot(grid_x, nat_kde, color="orange", label="Natural", linewidth=6, linestyle="-")
    if np.isfinite(rnd_kde).any():
        plt.plot(grid_x, rnd_kde, color="green", label="Random", linewidth=6, linestyle="-")

    plt.ylabel("Frequency", fontsize=24)
    plt.title(f"{shape_name}", fontsize=18)

    plt.xlim(0, 1)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], fontsize=20)
    plt.tick_params(axis="both", labelsize=20)

    ax = plt.gca()
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=8)
    plt.subplots_adjust(left=0.16, bottom=0.15)

    final_png = os.path.join(save_dir, f"{save_prefix}_{shape_name}.png")
    final_svg = os.path.join(save_dir, f"{save_prefix}_{shape_name}.svg")

    plt.savefig(final_png, dpi=300, bbox_inches="tight")
    plt.savefig(final_svg, bbox_inches="tight")

    print("Saved:", final_png)
    plt.show()
    plt.close()

    curve_csv = os.path.join(stats_dir, f"{save_prefix}_{shape_name}_curves.csv")
    df_curve = pd.DataFrame({
        "x": grid_x,
        "Generated_density": gen_kde,
        "Natural_density": nat_kde,
        "Random_density": rnd_kde,
    })
    df_curve.to_csv(curve_csv, index=False)
    print("Saved:", curve_csv)

print("\n✅ All done (4 DNA shape plots: Generated vs Natural vs Random).")
