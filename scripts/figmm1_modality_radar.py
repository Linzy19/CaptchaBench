"""
figmm_fig7_radar.py  (renamed from figmm1_modality_radar.py)
=============================================================
ACM MM 2026 — CaptchaShield Fig.7

Three-axis radar chart: CR(Q1)↑ / (1-TVR)(Q2)↑ / ASR(Q3)↑
Each axis normalized to [0,1] (higher = better protection on all axes).

色系：与 Fig3/Fig5/Fig6 完全一致的模态逻辑色
  Image-only: AMP #0D47A1 / ASPL #1976D2 / XTransfer #42A5F5 / Glaze #90CAF9
  Text-only:  Nightshade #E64A19
  Image+Text: MMCoA #2E7D32

2026-03-28 更新：
   - 改名为 fig7，输出文件为 fig7_radar.pdf
   - 所有线条统一（实线），Nightshade 不再特殊处理
   - 图例放在图内右上角（bbox外侧），整洁简明
   - 4VLM均值数据（排除Gemini）
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Output dir ──────────────────────────────────────────────────────────────
HERE    = Path(__file__).parent
OUT_DIR = HERE.parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":  "serif",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# ── 颜色（与 Fig3/Fig5/Fig6 完全一致的模态逻辑色）───────────────────────────
COLORS = {
    "AMP":        "#0D47A1",   # deep blue   (Image-only)
    "ASPL":       "#1976D2",   # mid blue    (Image-only)
    "XTransfer":  "#42A5F5",   # light blue  (Image-only)
    "Glaze":      "#90CAF9",   # pale blue   (Image-only)
    "Nightshade": "#E64A19",   # deep orange (Text-only)
    "MMCoA":      "#2E7D32",   # dark green  (Image+Text)
}

MODALITY_GROUP = {
    "AMP":        "Image-only",
    "ASPL":       "Image-only",
    "XTransfer":  "Image-only",
    "Glaze":      "Image-only",
    "Nightshade": "Text-only",
    "MMCoA":      "Image+Text",
}

LINESTYLES = {
    "AMP":        "-",
    "ASPL":       "-",
    "XTransfer":  "-",
    "Glaze":      "-",
    "Nightshade": "-",   # 统一实线
    "MMCoA":      "-",
}

LINEWIDTHS = {
    "AMP":        1.8,
    "ASPL":       1.8,
    "XTransfer":  1.8,
    "Glaze":      1.8,
    "Nightshade": 1.8,
    "MMCoA":      2.2,   # Image+Text 稍粗
}

# ── Data（4VLM 均值，排除 Gemini）───────────────────────────────────────────
METHODS = ["AMP", "ASPL", "XTransfer", "Glaze", "Nightshade", "MMCoA"]

# [CR(%), 1-TVR(%), ASR(%)]  — 4VLM avg (Qwen+Kimi+GPT-5.2+GLM-4V)
DATA = {
    "AMP":        [99.5,  96.3,  99.7],
    "ASPL":       [99.1,  94.0,  99.4],
    "XTransfer":  [97.8,  88.4,  98.1],
    "Glaze":      [97.5,  87.6,  97.7],
    "Nightshade": [99.3,  93.8,  99.6],
    "MMCoA":      [98.7,  93.3,  98.9],
}

AXES  = ["CR\n(Q1)↑", "1−TVR\n(Q2)↑", "ASR\n(Q3)↑"]
N     = len(AXES)

FLOOR = [96.0, 85.0, 96.0]
CEIL  = [100.0, 98.0, 100.0]

def normalise(val, floor, ceil):
    return max(0.0, min(1.0, (val - floor) / (ceil - floor)))

# ── Angles ───────────────────────────────────────────────────────────────────
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.8, 5.8),
                        subplot_kw=dict(projection="polar"))

# 背景：淡灰填充区域（最外圈）
bg_vals = [1.0] * (N + 1)
ax.fill(angles, bg_vals, color="#F5F5F5", alpha=0.60, zorder=0)

# 网格圆圈
for level, alpha in [(0.25, 0.35), (0.50, 0.40), (0.75, 0.40), (1.00, 0.55)]:
    ax.plot(angles, [level] * (N + 1),
            color="#BDBDBD", linewidth=0.7, linestyle="--", zorder=1, alpha=alpha)

# 方法多边形
for m in METHODS:
    vals   = DATA[m]
    vals_n = [normalise(v, FLOOR[i], CEIL[i]) for i, v in enumerate(vals)]
    vals_n += vals_n[:1]
    ax.plot(angles, vals_n,
            color=COLORS[m],
            linewidth=LINEWIDTHS[m],
            linestyle=LINESTYLES[m],
            zorder=4)
    ax.fill(angles, vals_n, color=COLORS[m], alpha=0.08, zorder=3)

# ── 轴标签 ────────────────────────────────────────────────────────────────────
ax.set_xticks(angles[:-1])
ax.set_xticklabels(AXES, fontsize=11, fontfamily="serif", fontweight="bold")
ax.tick_params(pad=20)
ax.set_ylim(0, 1.0)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(["", "", "", ""], fontsize=0)
ax.set_rlabel_position(30)

# 关闭极坐标默认网格（用自定义圆圈替代）
ax.grid(False)
ax.spines["polar"].set_visible(False)

# ── 图例：方法颜色线段，放右上角外侧 ─────────────────────────────────────────
method_lines = [
    plt.Line2D([0], [0],
               color=COLORS[m],
               linewidth=2.0,
               linestyle=LINESTYLES[m],
               label=m)
    for m in METHODS
]
ax.legend(handles=method_lines,
          fontsize=8.5,
          prop={"family": "serif"},
          loc="upper right",
          bbox_to_anchor=(1.38, 1.18),
          framealpha=0.92,
          edgecolor="#CCCCCC",
          ncol=1,
          handlelength=1.8)

ax.set_title("CR / (1−TVR) / ASR by Input Modality\n(4-VLM avg, ID-based targets)",
             fontsize=10.5, fontfamily="serif", fontweight="bold", pad=22)

# ── Save — 输出文件名改为 fig7_radar ─────────────────────────────────────────
for d, name in [(OUT_DIR, "figmm_fig7_radar"), (FIG_DIR, "fig7_radar")]:
    fig.savefig(d / f"{name}.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved → {d / name}.pdf")

plt.close(fig)
