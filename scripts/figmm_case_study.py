"""
figmm_case_study.py
───────────────────
Generate two case-study figures for the adversarial CAPTCHA paper.

Figure G1  (fig_case_study_1.pdf) — 2×6 grid: originals vs. adversarial
Figure G2  (fig_case_study_2.pdf) — 4 side-by-side [orig | adv] pairs
"""

from pathlib import Path
import matplotlib
import matplotlib.lines
matplotlib.rcParams.update({
    "font.family":        "serif",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
import matplotlib.font_manager as fm
import numpy as np

# ── CJK font ──────────────────────────────────────────────────────────────────
_CJK_CANDIDATES = ["PingFang HK", "Heiti TC", "Arial Unicode MS",
                    "Hiragino Sans GB", "STHeiti"]
_avail = {f.name for f in fm.fontManager.ttflist}
_cjk_font = next((c for c in _CJK_CANDIDATES if c in _avail), None)
print(f"[font] CJK font: {_cjk_font!r}")

def cjk_fp(size=8, weight="normal"):
    if _cjk_font:
        return fm.FontProperties(family=_cjk_font, size=size, weight=weight)
    return fm.FontProperties(size=size, weight=weight)

# ── paths ─────────────────────────────────────────────────────────────────────
ORIG_DIR = Path("/Users/lzy/Code_lzy/AttackVLM/split")
ADV_DIR  = Path("/Users/lzy/Code_lzy/AttackVLM/imgae_test/result"
                "/source_target_pic_eps32_steps1200")
OUT_DIR  = Path("/Users/lzy/Code_lzy/Posion_paper/ACM_MM26/main/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── scan images ───────────────────────────────────────────────────────────────
orig_files = sorted(ORIG_DIR.glob("*.png"))
adv_files  = sorted(ADV_DIR.glob("*_adv.png"))
print(f"[scan] {len(orig_files)} originals,  {len(adv_files)} adversarials")

orig_map = {}
for p in orig_files:
    parts = p.stem.split("_", 1)
    if len(parts) == 2:
        orig_map[(parts[0], parts[1])] = p

adv_map = {}
for p in adv_files:
    stem = p.stem                          # e.g. "贞_stunning_danxia_adv"
    if stem.endswith("_adv"):
        parts = stem[:-4].split("_", 1)
        if len(parts) == 2:
            adv_map[(parts[0], parts[1])] = p

# ── matched pairs ─────────────────────────────────────────────────────────────
matched = []
for key in sorted(adv_map):
    if key in orig_map:
        matched.append((key[0], key[1], orig_map[key], adv_map[key]))

print(f"[match] {len(matched)} pairs:")
for c, b, *_ in matched:
    print(f"        char={c}  bg={b}")

# ── helpers ───────────────────────────────────────────────────────────────────
def load_img(path):
    img = imread(str(path))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img

def hide_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def pretty_bg(bg):
    return bg.replace("_", " ").title()

# ── pair selection ────────────────────────────────────────────────────────────
PREFERRED_BG = [
    "stunning_danxia", "aerial_top",        "autumn_forest",
    "rugged_canyon",   "tropical_highland",  "colorful_autumn",
    "rocky_mountain",  "crumbling_ancient",  "dramatic_rocky",
    "terraced_hillside","high_altitude",     "aerial_view",
]
by_bg = {item[1]: item for item in matched}

def pick_pairs(n):
    chosen, seen = [], set()
    for bg in PREFERRED_BG:
        if bg in by_bg and bg not in seen and len(chosen) < n:
            chosen.append(by_bg[bg]); seen.add(bg)
    for item in matched:
        if len(chosen) >= n: break
        if item[1] not in seen:
            chosen.append(item); seen.add(item[1])
    return chosen

g1_pairs = pick_pairs(6)
g2_pairs = pick_pairs(4)
print(f"\n[G1] {len(g1_pairs)} pairs: {[b for _,b,*_ in g1_pairs]}")
print(f"[G2] {len(g2_pairs)} pairs: {[b for _,b,*_ in g2_pairs]}")

# ═══════════════════════════════════════════════════════════════════════════════
#  Figure G1 — 2 × 6 grid
# ═══════════════════════════════════════════════════════════════════════════════
N = len(g1_pairs)
fig1, axes1 = plt.subplots(2, N, figsize=(14, 5), facecolor="white")
fig1.patch.set_facecolor("white")

for col, (char, bg, orig_p, adv_p) in enumerate(g1_pairs):
    ax_o = axes1[0, col]
    ax_a = axes1[1, col]

    # row 0: original
    ax_o.imshow(load_img(orig_p))
    hide_axes(ax_o)
    # Chinese char label (large, via CJK font)
    ax_o.text(0.5, -0.03, char,
              transform=ax_o.transAxes,
              ha="center", va="top", clip_on=False,
              fontproperties=cjk_fp(size=12, weight="bold"),
              color="#111111")
    # background label (small, serif)
    ax_o.text(0.5, -0.17, pretty_bg(bg),
              transform=ax_o.transAxes,
              ha="center", va="top", clip_on=False,
              fontsize=6.5, color="#555555", fontfamily="serif")

    # row 1: adversarial
    ax_a.imshow(load_img(adv_p))
    hide_axes(ax_a)
    ax_a.text(0.5, -0.07, "Perturbed  (AMP-style)",
              transform=ax_a.transAxes,
              ha="center", va="top", clip_on=False,
              fontsize=7, color="#CC2222", fontfamily="serif")

# row labels
axes1[0, 0].set_ylabel("Original",    fontsize=9, fontweight="bold",
                        color="#333333", labelpad=6)
axes1[1, 0].set_ylabel("Adversarial", fontsize=9, fontweight="bold",
                        color="#CC2222", labelpad=6)

# dashed row separator
fig1.add_artist(matplotlib.lines.Line2D(
    [0.04, 0.97], [0.50, 0.50],
    transform=fig1.transFigure,
    color="#BBBBBB", linewidth=0.8, linestyle="--"
))

fig1.suptitle(
    "Case Study: Original vs. Adversarially Perturbed CAPTCHA Images",
    fontsize=12, fontweight="bold", y=1.02, color="#111111"
)
fig1.text(0.98, 0.99, "ε = 32,  steps = 1200",
          fontsize=7, color="#999999",
          ha="right", va="top", style="italic",
          transform=fig1.transFigure)

plt.tight_layout(rect=[0, 0, 1, 1])
out1 = OUT_DIR / "fig_case_study_1.pdf"
fig1.savefig(str(out1), dpi=200, bbox_inches="tight",
             facecolor="white", edgecolor="none")
plt.close(fig1)
print(f"\n[G1] Saved → {out1}")

# ═══════════════════════════════════════════════════════════════════════════════
#  Figure G2 — 4 pair columns: [orig | adv] × 4
# ═══════════════════════════════════════════════════════════════════════════════
NP   = len(g2_pairs)   # 4
SEPW = 0.06
IMGW = 1.0

col_widths = []
for i in range(NP):
    col_widths += [IMGW, IMGW]
    if i < NP - 1:
        col_widths.append(SEPW)

fig2 = plt.figure(figsize=(14, 4), facecolor="white")
fig2.patch.set_facecolor("white")
gs = gridspec.GridSpec(
    1, len(col_widths),
    width_ratios=col_widths,
    figure=fig2,
    wspace=0.03, hspace=0.0,
    left=0.02, right=0.98,
    top=0.76, bottom=0.20,
)

for pi, (char, bg, orig_p, adv_p) in enumerate(g2_pairs):
    gc_o = pi * 3
    gc_a = pi * 3 + 1

    ax_o = fig2.add_subplot(gs[0, gc_o])
    ax_a = fig2.add_subplot(gs[0, gc_a])

    ax_o.imshow(load_img(orig_p))
    ax_a.imshow(load_img(adv_p))

    for ax in (ax_o, ax_a):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # thin vertical separator between orig and adv
    ax_o.spines["right"].set_visible(True)
    ax_o.spines["right"].set_color("#CCCCCC")
    ax_o.spines["right"].set_linewidth(0.9)

    ax_o.set_title("Orig", fontsize=7, pad=2.5, color="#444444")
    ax_a.set_title("Adv",  fontsize=7, pad=2.5, color="#BB1111",
                   fontweight="bold")

    # pair label — centred at x=1.0 (midpoint between the two axes)
    label = f"char: {char},  bg: {pretty_bg(bg)}"
    ax_o.annotate(
        label,
        xy=(1.0, -0.07),
        xycoords="axes fraction",
        fontsize=7.5, ha="center", va="top",
        color="#333333",
        clip_on=False,
        fontproperties=cjk_fp(size=7.5),
    )

fig2.suptitle(
    "Adversarial Perturbation Preserves Visual Quality",
    fontsize=12, fontweight="bold", y=0.98, color="#111111"
)
fig2.text(0.98, 0.93, "ε = 32 / 1200 steps  (AMP-style)",
          fontsize=7, color="#999999", ha="right", va="top", style="italic")

out2 = OUT_DIR / "fig_case_study_2.pdf"
fig2.savefig(str(out2), dpi=200, bbox_inches="tight",
             facecolor="white", edgecolor="none")
plt.close(fig2)
print(f"[G2] Saved → {out2}")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────")
print(f"  Originals scanned   : {len(orig_files)}")
print(f"  Adversarials scanned: {len(adv_files)}")
print(f"  Matched pairs found : {len(matched)}")
print(f"  G1 pairs used       : {len(g1_pairs)}")
print(f"  G2 pairs used       : {len(g2_pairs)}")
print(f"  Output G1 : {out1}  exists={out1.exists()}")
print(f"  Output G2 : {out2}  exists={out2.exists()}")
