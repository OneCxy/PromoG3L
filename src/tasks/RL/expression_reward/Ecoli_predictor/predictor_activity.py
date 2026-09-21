#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import re
import os
import argparse
import numpy as np
import torch
from torch.nn.functional import pad
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

import matplotlib.pyplot as plt
try:
    from scipy.stats import gaussian_kde
    _SCIPY_OK = True
except Exception:
    gaussian_kde = None
    _SCIPY_OK = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
from predictor_models import LSTMModel  


class PREDICT:
    def __init__(self, model_path, batch_size: int = 512, confirm_channels: bool = False):
        self.input_size = 4
        self.hidden_size = 256
        self.output_size = 1
        self.lambda_l2 = 0.001
        self.dropout_rate = 0.2
        self.model_path = model_path
        self.use_gpu = torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = self.load_model()
        self.batch_size = batch_size
        self.confirm_channels = confirm_channels

        print("[INFO] device:", self.device)
        print("[CHK] input_size passed to LSTMModel =", self.input_size)
        try:
            bn = self.model.cnn1[0]  # cnn1 = Sequential(BN, Conv, ReLU, Pool)
            print("[CHK] cnn1 BatchNorm1d num_features =", getattr(bn, "num_features", "NA"))
        except Exception as e:
            print("[WARN] cnn1 结构未按预期存在（这不影响推理）：", repr(e))

    def load_model(self):
        model = LSTMModel(self.input_size, self.hidden_size, self.output_size,
                          self.dropout_rate, self.lambda_l2)
        state = torch.load(self.model_path, map_location=(self.device))
        model.load_state_dict(state)
        if self.use_gpu:
            model.cuda()
        model.eval()
        return model

    @staticmethod
    def clean_seq(s: str) -> str:
        s = s.strip()
        s = s.lower()
        s = re.sub('[^acgtACGT]', 'z', s)
        return s

    @staticmethod
    def string_to_array(my_string: str):
        my_string = my_string.lower()
        my_string = re.sub('[^acgtACGT]', 'z', my_string)
        return np.array(list(my_string))

    @staticmethod
    def one_hot_encode(my_array: np.ndarray):
        mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        idx = np.fromiter((mapping.get(ch, 4) for ch in my_array), dtype=np.int64, count=len(my_array))
        oh5 = np.eye(5, dtype=np.float32)[idx]
        return oh5[:, :4]  # 4 通道

    def seq_onehot(self, seq_list):
        mats = [torch.tensor(self.one_hot_encode(self.string_to_array(s)),
                             dtype=torch.float32) for s in seq_list]
        if self.confirm_channels and len(mats) > 0:
            print("[CHK] example one-hot shape (L, C) =", mats[0].shape)
        max_len = max(m.shape[0] for m in mats) if mats else 0
        padded = [pad(m, (0, 0, 0, max_len - m.shape[0]), value=0) for m in mats]
        return torch.stack(padded, dim=0)  # (B, Lmax, 4)

    @torch.no_grad()
    def predict_list(self, seqs):
        preds = []
        for i in range(0, len(seqs), self.batch_size):
            batch = seqs[i:i+self.batch_size]
            x = self.seq_onehot(batch).to(self.device)  # (B, L, 4)
            y = self.model(x).detach().cpu().numpy().flatten().tolist()
            preds.extend([float(v) for v in y])
        if len(preds) < len(seqs):
            preds.extend([0.0] * (len(seqs) - len(preds)))
        return preds


# I/O
def read_csv_sequences(path: str, seq_col: str):
    df = pd.read_csv(path)
    if seq_col not in df.columns:
        first_col = df.columns[0]
        print(f"[WARN] 未找到列 '{seq_col}'，自动使用第一列 '{first_col}' 作为序列列")
        seq_col = first_col
    seqs = df[seq_col].astype(str).fillna("").tolist()
    return df, seqs

def write_csv_with_preds(df: pd.DataFrame, preds, out_path: str, pred_col: str = "pred_activity"):
    df_out = df.copy()
    df_out[pred_col] = preds
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    df_out.to_csv(out_path, index=False)
    print(f"[DONE] 写出 CSV：{out_path}  (共 {len(df_out)} 行, 新增列: {pred_col})")

def read_txt_sequences(path: str):
    with open(path, "r") as f:
        seqs = [line.strip() for line in f if line.strip()]
    return seqs

def write_txt_with_preds(seqs, preds, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, "w") as f:
        for s, p in zip(seqs, preds):
            f.write(f"{s}\t{p:.6f}\n")
    print(f"[DONE] 写出 TXT：{out_path}  (共 {len(seqs)} 行，格式：sequence<tab>pred)")


# Metrics
def _gc_fraction(seq: str) -> float:
    seq = seq.replace("z", "")  
    if not seq:
        return float("nan")
    seq = seq.lower()
    gc = seq.count("g") + seq.count("c")
    return gc / max(1, len(seq))

def summarize_metrics(preds, seqs=None, topk=50, thr=None):
    """
    preds: List[float]
    seqs:  List[str] (clean后的也行)
    topk:  计算 topk mean
    thr:   list of thresholds, 打印超过阈值的比例
    """
    a = np.asarray(preds, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print("[METRIC] no finite predictions.")
        return

    n = int(a.size)
    mean = float(a.mean())
    std = float(a.std())
    mn = float(a.min())
    mx = float(a.max())
    p50 = float(np.percentile(a, 50))
    p90 = float(np.percentile(a, 90))
    p95 = float(np.percentile(a, 95))
    p99 = float(np.percentile(a, 99))

    k = int(min(max(1, topk), n))
    topk_mean = float(np.mean(np.sort(a)[-k:]))

    print("\n========== [METRICS] predictor ==========")
    print(f"[METRIC] n               = {n}")
    print(f"[METRIC] mean(activity)  = {mean:.6f}")   
    print(f"[METRIC] std            = {std:.6f}")
    print(f"[METRIC] min / max      = {mn:.6f} / {mx:.6f}")
    print(f"[METRIC] p50/p90/p95/p99= {p50:.6f} / {p90:.6f} / {p95:.6f} / {p99:.6f}")
    print(f"[METRIC] top{k}_mean     = {topk_mean:.6f}")

    if thr:
        for t in thr:
            frac = float(np.mean(a >= float(t)))
            print(f"[METRIC] frac>= {float(t):.6g}   = {frac:.6f}")

    if seqs is not None and len(seqs) > 0:
        seqs2 = [str(s) for s in seqs]
        uniq = len(set(seqs2))
        uniq_rate = uniq / max(1, len(seqs2))
        gcs = np.asarray([_gc_fraction(s) for s in seqs2], dtype=np.float32)
        gcs = gcs[np.isfinite(gcs)]
        if gcs.size > 0:
            print(f"[METRIC] GC_mean / GC_std = {float(gcs.mean()):.6f} / {float(gcs.std()):.6f}")
        print(f"[METRIC] unique_rate       = {uniq_rate:.6f}  (unique={uniq})")

    print("========================================\n")


# Plotting
def plot_activity_distribution_fixed_bins(
    acts,
    bin_start=1.0,
    bin_end=18.0,
    out_path=None,
    title="Promoter activity distribution",
):
    acts = np.asarray(acts, dtype=np.float32)
    acts = acts[np.isfinite(acts)]
    if acts.size == 0:
        print("[WARN] 无有效活性值，跳过绘图。")
        return

    if bin_end <= bin_start:
        bin_end = bin_start + 1.0
        print(f"[WARN] bin_end <= bin_start，自动调整为 [{bin_start}, {bin_end})")

    bin_edges = np.arange(bin_start, bin_end + 1, 1, dtype=float)
    counts, _ = np.histogram(acts, bins=bin_edges)
    total = int(counts.sum())

    kde_y = None
    x_eval = np.linspace(bin_start, bin_end, 600)
    if _SCIPY_OK:
        acts_for_kde = acts[(acts >= bin_start) & (acts <= bin_end)]
        if acts_for_kde.size > 1 and acts_for_kde.std() > 0:
            kde = gaussian_kde(acts_for_kde)
            kde_y = kde(x_eval) * total
    else:
        print("[INFO] 未检测到 scipy，KDE 将被跳过（仅绘制直方图）")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.hist(
        acts, bins=bin_edges, density=False, alpha=0.55,
        color="#6fa8dc", edgecolor="#4a86c5", linewidth=0.8,
        label="Histogram (count)"
    )
    if kde_y is not None:
        ax.plot(x_eval, kde_y, linewidth=2.0, color="#ff7f0e", label="KDE (smoothed)")

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("Predicted activity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlim(bin_start, bin_end)
    ax.set_xticks(np.arange(bin_start, bin_end + 1, 1))
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)

    mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    max_c = max(int(counts.max()), 1)
    y_off = max(1, int(0.02 * max_c))
    for c, x in zip(counts, mids):
        if c > 0:
            ax.text(x, c + y_off, str(int(c)), ha="center", va="bottom", fontsize=9)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        fontsize=10,
    )
    legend.get_frame().set_edgecolor("#888888")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        print(f"[OK] 图像已保存: {out_path}")
    else:
        plt.show()
    plt.close(fig)


# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="LSTMModel 的权重路径（.pth）")

    parser.add_argument("--in_csv", type=str, default=None, help="输入 CSV 路径")
    parser.add_argument("--seq_col", type=str, default="sequence", help="CSV 中序列列名")
    parser.add_argument("--out_csv", type=str, default=None, help="输出 CSV 路径")

    parser.add_argument("--in_txt", type=str, default=None, help="输入 TXT 路径（一行一条序列）")
    parser.add_argument("--out_txt", type=str, default=None, help="输出 TXT 路径")

    parser.add_argument("--batch_size", type=int, default=512, help="推理 batch size")
    parser.add_argument("--confirm_channels", action="store_true", help="打印 one-hot 形状和通道数")
    parser.add_argument("--pred_col", type=str, default="pred_activity", help="输出列名（CSV）")

    parser.add_argument("--plot_out", type=str, default=None, help="活性分布图输出路径（提供则绘图）")
    parser.add_argument("--bin_start", type=float, default=-10.0, help="固定分箱起点")
    parser.add_argument("--bin_end", type=float, default=10.0, help="固定分箱终点")
    parser.add_argument("--plot_title", type=str, default="Promoter activity distribution", help="图标题")

    parser.add_argument("--topk", type=int, default=50, help="打印 topK mean 的 K（默认50）")
    parser.add_argument(
        "--thr", type=float, nargs="*", default=[],
        help="阈值列表：打印 frac(pred>=thr)。例如 --thr 0 2 5"
    )

    args = parser.parse_args()

    if not args.in_csv and not args.in_txt:
        raise SystemExit("必须提供 --in_csv 或 --in_txt 之一。")

    predictor = PREDICT(args.ckpt, batch_size=args.batch_size, confirm_channels=args.confirm_channels)

    preds_for_plot = None

    if args.in_csv:
        df, seqs = read_csv_sequences(args.in_csv, args.seq_col)
        print(f"[INFO] 读取 CSV: {args.in_csv}  共 {len(seqs)} 条")
        seqs = [predictor.clean_seq(s) for s in seqs]
        preds = predictor.predict_list(seqs)

        preds_for_plot = preds

        out_csv = args.out_csv or (
            args.in_csv[:-4] + "_pred.csv" if args.in_csv.lower().endswith(".csv") else args.in_csv + "_pred.csv"
        )
        write_csv_with_preds(df, preds, out_csv, pred_col=args.pred_col)

        summarize_metrics(preds, seqs=seqs, topk=args.topk, thr=args.thr)

    if args.in_txt:
        seqs = read_txt_sequences(args.in_txt)
        print(f"[INFO] 读取 TXT: {args.in_txt}  共 {len(seqs)} 条")
        seqs = [predictor.clean_seq(s) for s in seqs]
        preds = predictor.predict_list(seqs)

        preds_for_plot = preds

        out_txt = args.out_txt or (args.in_txt + "_pred.txt")
        write_txt_with_preds(seqs, preds, out_txt)

        summarize_metrics(preds, seqs=seqs, topk=args.topk, thr=args.thr)

    if args.plot_out and preds_for_plot is not None:
        finite_preds = np.asarray(preds_for_plot, dtype=np.float32)
        finite_preds = finite_preds[np.isfinite(finite_preds)]
        if finite_preds.size == 0:
            print("[WARN] 预测列表为空或无有效值，跳过绘图。")
        else:
            print(f"[INFO] 预测活性范围: {finite_preds.min():.3f} ~ {finite_preds.max():.3f}")
            plot_activity_distribution_fixed_bins(
                finite_preds,
                bin_start=args.bin_start,
                bin_end=args.bin_end,
                out_path=args.plot_out,
                title=args.plot_title,
            )

if __name__ == "__main__":
    main()
