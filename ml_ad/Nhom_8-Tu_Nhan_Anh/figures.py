# -*- coding: utf-8 -*-
"""Sinh toan bo hinh cho bao cao JTE tu outputs/ (chay sau train.py).

H1 workflow.png      : so do khoi phuong phap FRF-MLP
H2 label_dist.png    : phan bo nhan theo split
H3 memberships.png   : ham thanh vien LOW/MED/HIGH
H4 training.png      : (a) duong hoc dev-F1, (b) quet lambda
H5 comparison.png    : so sanh accuracy / macro-F1 cac mo hinh
H6 confusion.png     : ma tran nham lan (a) MLP thuan, (b) FRF-MLP
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
OUT = HERE / "outputs"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 200,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
LABELS = ["CLEAN", "OFFENSIVE", "HATE"]

metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
curves = json.loads((OUT / "curves.json").read_text(encoding="utf-8"))
confs = np.load(OUT / "confusions.npz")


# --- H1: workflow ----------------------------------------------------------
def h1():
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec="#2b5aa0", fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.0, color="#333"))

    # cot 1: dau vao + tien xu ly
    box(0.01, 0.42, 0.15, 0.16, "Bình luận MXH\n(ViHSD)", fc="#fdf3e3", ec="#b07d2b")
    box(0.22, 0.42, 0.15, 0.16, "Tiền xử lý\n(chuẩn hóa, tách token)")
    # nhanh tren: TF-IDF -> MLP
    box(0.44, 0.70, 0.16, 0.16, "TF-IDF\ntừ (1–2) + ký tự (2–4)")
    box(0.66, 0.70, 0.13, 0.16, "MLP\n256–128–3")
    # nhanh duoi: fuzzy
    box(0.44, 0.14, 0.16, 0.16, "Lexicon log-odds\n+ biến (S, D, T)")
    box(0.66, 0.14, 0.13, 0.16, "Hệ luật mờ\nMamdani (7 luật)")
    # fusion
    box(0.84, 0.42, 0.15, 0.16, "Kết hợp quyết định\n(1−λ)·p_MLP + λ·p_mờ",
        fc="#e8f6ec", ec="#2b7a3f")
    arrow(0.16, 0.50, 0.22, 0.50)
    arrow(0.37, 0.53, 0.44, 0.76)
    arrow(0.37, 0.47, 0.44, 0.24)
    arrow(0.60, 0.78, 0.66, 0.78)
    arrow(0.60, 0.22, 0.66, 0.22)
    arrow(0.79, 0.75, 0.875, 0.58)
    arrow(0.79, 0.25, 0.875, 0.42)
    # fuzzy features -> MLP (fusion dac trung)
    ax.add_patch(FancyArrowPatch((0.705, 0.30), (0.705, 0.70), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.0, color="#2b7a3f",
                                 linestyle="--"))
    ax.text(0.695, 0.50, "22 đặc trưng mờ\n(fusion đặc trưng)", fontsize=7,
            color="#2b7a3f", ha="right", va="center")
    ax.text(0.915, 0.30, "→ CLEAN / OFFENSIVE\n     / HATE", fontsize=7.5,
            ha="center", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIG / "workflow.png")
    plt.close(fig)


# --- H2: label distribution -------------------------------------------------
def h2():
    counts = {"train": [19885, 1605, 2556], "dev": [2190, 212, 270],
              "test": [5548, 444, 688]}
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    x = np.arange(3); w = 0.26
    colors = ["#4878a8", "#e0913d", "#c34a4a"]
    for i, lb in enumerate(LABELS):
        vals = [counts[s][i] for s in ["train", "dev", "test"]]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lb, color=colors[i])
        ax.bar_label(bars, fontsize=6.5, padding=1)
    ax.set_xticks(x, ["train (24 046)", "dev (2 672)", "test (6 680)"])
    ax.set_ylabel("Số bình luận")
    ax.set_ylim(0, 22500)
    ax.legend(fontsize=7.5)
    fig.savefig(FIG / "label_dist.png")
    plt.close(fig)


# --- H3: membership functions -----------------------------------------------
def h3():
    from fuzzy import _trap
    v = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    ax.plot(v, _trap(v, -1, 0, 0.15, 0.40), label="LOW", color="#4878a8")
    ax.plot(v, _trap(v, 0.20, 0.45, 0.55, 0.80), label="MED", color="#e0913d")
    ax.plot(v, _trap(v, 0.60, 0.85, 1.0, 2.0), label="HIGH", color="#c34a4a")
    ax.set_xlabel("Giá trị biến ngôn ngữ (chuẩn hóa [0, 1])")
    ax.set_ylabel(r"Độ thuộc $\mu$")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "memberships.png")
    plt.close(fig)


# --- H4: training curves + lambda sweep -------------------------------------
def h4():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    for k, lb, c in [("mlp", "MLP thuần", "#4878a8"),
                     ("mlp_feat", "MLP + đặc trưng mờ", "#2b7a3f")]:
        f1 = curves[k]["dev_f1"]
        ax.plot(range(1, len(f1) + 1), f1, marker="o", ms=3, label=lb, color=c)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro-F1 (dev)")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
    ax.set_title("(a) Đường học trên tập dev", fontsize=8.5)

    ax = axes[1]
    for k, lb, c in [("lambda_sweep_mlp", "MLP thuần + p_mờ", "#4878a8"),
                     ("lambda_sweep_frf", "FRF-MLP (đề xuất)", "#2b7a3f")]:
        d = curves[k]
        ax.plot(d["lambdas"], d["dev_f1"], marker="o", ms=3, label=lb, color=c)
    ax.set_xlabel(r"Trọng số kết hợp $\lambda$"); ax.set_ylabel("Macro-F1 (dev)")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
    ax.set_title(r"(b) Quét $\lambda$ trên tập dev", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "training.png")
    plt.close(fig)


# --- H5: comparison bar ------------------------------------------------------
def h5():
    order = ["fuzzy", "softmax", "mlp", "mlp_dec", "mlp_feat", "frf_mlp"]
    names = ["Hệ mờ\nthuần", "Hồi quy\nsoftmax", "MLP\nthuần",
             "MLP+mờ\nquyết định", "MLP+mờ\nđặc trưng", "FRF-MLP\n(đề xuất)"]
    acc = [metrics[k]["accuracy"] * 100 for k in order]
    mf1 = [metrics[k]["macro_f1"] * 100 for k in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    b1 = ax.bar(x - w / 2, acc, w, label="Accuracy (%)", color="#4878a8")
    b2 = ax.bar(x + w / 2, mf1, w, label="Macro-F1 (%)", color="#e0913d")
    ax.bar_label(b1, fmt="%.1f", fontsize=6.5)
    ax.bar_label(b2, fmt="%.1f", fontsize=6.5)
    ax.set_xticks(x, names, fontsize=7.5)
    ax.set_ylim(0, 100); ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG / "comparison.png")
    plt.close(fig)


# --- H6: confusion matrices --------------------------------------------------
def h6():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for ax, key, ttl in [(axes[0], "mlp", "(a) MLP thuần"),
                         (axes[1], "frf_mlp", "(b) FRF-MLP (đề xuất)")]:
        cm = confs[key]
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]*100:.1f}%)",
                        ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks(range(3), LABELS, fontsize=7)
        ax.set_yticks(range(3), LABELS, fontsize=7)
        ax.set_xlabel("Dự đoán", fontsize=8); ax.set_ylabel("Thực tế", fontsize=8)
        ax.set_title(ttl, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "confusion.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in (h1, h2, h3, h4, h5, h6):
        f()
        print("ok", f.__name__)
    print("figures ->", FIG)
