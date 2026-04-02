"""
compute_table4mm.py
===================
ACM MM 2026 — CaptchaShield Table 4MM data extractor

从 run_20260316_201833 JSON文件中提取 Q1(CR)/Q2(TVR)/Q3(ASR) 真实数据
按输入模态分组汇总，输出 CSV 用于 Table 4MM

Input:  /Users/lzy/Code_lzy/AttackVLM/eval_results_v2/run_20260316_201833/
Output: ../../results/table4mm_data.csv  (→ ACM_MM26/results/)
"""

import json
import csv
import statistics
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/Users/lzy/Code_lzy/AttackVLM/eval_results_v2/run_20260316_201833")
HERE        = Path(__file__).parent
OUT_DIR     = HERE.parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV     = OUT_DIR / "table4mm_data.csv"

# ── Method → JSON file mapping (SD-ID / ID-based results) ────────────────────
METHOD_FILES = {
    "ASPL":      "results_SD-ID_aspl_eps0.05_steps200_20260316_201833.json",
    "Glaze(MI)": "results_SD-ID_mi_eps16_steps300_20260316_201833.json",
    "AMP":       "results_SD-ID_attackvlm_eps8_steps300_20260316_201833.json",
    "MMCoA":     "results_SD-ID_mmcoa_eps1_steps100_20260316_201833.json",
    "Nightshade":"results_SD-ID_nightshade_eps0.05_steps500_20260316_201833.json",
    "XTransfer": "results_SD-ID_xtransfer_eps12_steps300_20260316_201833.json",
}

MODALITY_GROUP = {
    "ASPL":       "Image-only",
    "Glaze(MI)":  "Image-only",
    "AMP":        "Image-only",
    "XTransfer":  "Image-only",
    "Nightshade": "Text-only",
    "MMCoA":      "Image+Text",
}

VLM_KEYS = ["qwen", "kimi", "glm5"]
VLM_LABELS = {"qwen": "Qwen-VL-Max", "kimi": "Kimi 2.5", "glm5": "GLM-5"}

def parse_json_results(json_path: Path, results_key: str = "adv_results"):
    """
    Extract per-image Q1/Q2/Q3 results for each VLM.
    Returns list of dicts: {vlm: {q1_correct, q2_visible, q3_correct}}
    """
    with open(json_path) as f:
        data = json.load(f)

    records = []
    for item in data.get(results_key, []):
        entry = {}
        for vlm in VLM_KEYS:
            q1_parsed = item.get("q1", {}).get(vlm, {}).get("parsed", {})
            q2_parsed = item.get("q2", {}).get(vlm, {}).get("parsed", {})
            q3_parsed = item.get("q3", {}).get(vlm, {}).get("parsed", {})

            # Q1: correct=False means VLM was CONFUSED (chose Target over Source) = CR↑
            q1_cr = 1 - int(bool(q1_parsed.get("correct", True)))
            # Q2: answer='yes' means text IS visible (TVR↓; lower = better protection)
            q2_vis = int(q2_parsed.get("answer", "no").lower() == "yes")
            # Q3: correct=False means VLM got WRONG char = attack succeeded (ASR↑)
            q3_asr = 1 - int(bool(q3_parsed.get("correct", True)))

            entry[vlm] = {"cr": q1_cr, "tvr": q2_vis, "asr": q3_asr}
        records.append(entry)
    return records

def compute_metrics(records):
    """Compute CR%, TVR%, ASR% per VLM and average."""
    result = {}
    for vlm in VLM_KEYS:
        cr_vals  = [r[vlm]["cr"]  for r in records if vlm in r]
        tvr_vals = [r[vlm]["tvr"] for r in records if vlm in r]
        asr_vals = [r[vlm]["asr"] for r in records if vlm in r]
        n = len(cr_vals)
        if n == 0:
            result[vlm] = {"cr": None, "tvr": None, "asr": None}
        else:
            result[vlm] = {
                "cr":  round(sum(cr_vals)  / n * 100, 1),
                "tvr": round(sum(tvr_vals) / n * 100, 1),
                "asr": round(sum(asr_vals) / n * 100, 1),
            }
    # Average across VLMs
    for metric in ["cr", "tvr", "asr"]:
        vals = [result[v][metric] for v in VLM_KEYS if result[v][metric] is not None]
        result["avg_" + metric] = round(sum(vals) / len(vals), 1) if vals else None
    # CVC: std of ASR across VLMs
    asr_list = [result[v]["asr"] for v in VLM_KEYS if result[v]["asr"] is not None]
    result["cvc_std"] = round(statistics.stdev(asr_list), 2) if len(asr_list) > 1 else 0.0
    return result

# ── Main ──────────────────────────────────────────────────────────────────────
rows = []
for method, fname in METHOD_FILES.items():
    fpath = RESULTS_DIR / fname
    if not fpath.exists():
        print(f"[WARN] File not found: {fpath}")
        continue

    records = parse_json_results(fpath, results_key="adv_results")
    if not records:
        print(f"[WARN] No adv_results in {fname}")
        continue

    metrics = compute_metrics(records)
    row = {
        "method":   method,
        "modality": MODALITY_GROUP[method],
        "n":        len(records),
    }
    for vlm in VLM_KEYS:
        lbl = vlm
        row[f"cr_{lbl}"]  = metrics[vlm]["cr"]
        row[f"tvr_{lbl}"] = metrics[vlm]["tvr"]
        row[f"asr_{lbl}"] = metrics[vlm]["asr"]
    row["avg_cr"]   = metrics["avg_cr"]
    row["avg_tvr"]  = metrics["avg_tvr"]
    row["avg_asr"]  = metrics["avg_asr"]
    row["cvc_std"]  = metrics["cvc_std"]
    rows.append(row)
    print(f"  {method:12s} | CR:{metrics['avg_cr']:5.1f}% | "
          f"TVR:{metrics['avg_tvr']:5.1f}% | ASR:{metrics['avg_asr']:5.1f}% | "
          f"CVC std:{metrics['cvc_std']:.2f}%")

# Write CSV
if rows:
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {OUT_CSV}")
else:
    print("[ERROR] No data extracted. Check RESULTS_DIR path.")
