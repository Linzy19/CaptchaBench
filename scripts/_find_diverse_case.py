"""
Find the case with MAXIMUM VARIANCE in protection rates across 6 methods.
We want rates like 0%, 20%, 40%, 60%, 80%, 100% — not 4×100% + 20% + 40%.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

RUN2 = Path("/Users/lzy/Code_lzy/AttackVLM/eval_results_v2/run_20260316_201833")
RUN1 = Path("/Users/lzy/Code_lzy/AttackVLM/eval_results_v2/run_20260324_174525")

METHOD_FILES = {
    "ASPL":       RUN2 / "results_SD-ID_aspl_eps0.05_steps200_20260316_201833.json",
    "Glaze":      RUN2 / "results_SD-ID_mi_eps16_steps300_20260316_201833.json",
    "AMP":        RUN2 / "results_SD-ID_attackvlm_eps8_steps300_20260316_201833.json",
    "MMCoA":      RUN2 / "results_SD-ID_mmcoa_eps1_steps100_20260316_201833.json",
    "Nightshade": RUN2 / "results_SD-ID_nightshade_eps0.05_steps500_20260316_201833.json",
    "XTransfer":  RUN2 / "results_SD-ID_xtransfer_eps12_steps300_20260316_201833.json",
}

RUN1_METHOD_MAP = {
    "source":     "SD-ID_baseline",
    "ASPL":       "SD-ID_aspl_eps0.05_steps200",
    "Glaze":      "SD-ID_mi_eps16_steps300",
    "AMP":        "SD-ID_attackvlm_eps8_steps300",
    "MMCoA":      "SD-ID_mmcoa_eps1_steps100",
    "Nightshade": "SD-ID_nightshade_eps0.05_steps500",
    "XTransfer":  "SD-ID_xtransfer_eps12_steps300",
}

ALL_VLMS = ["qwen", "kimi", "gpt52", "gemini", "glm4"]
RUN2_VLMS = ["gpt52", "gemini", "glm4"]
RUN1_VLMS = ["qwen", "kimi"]

def extract_cjk(s):
    for ch in str(s):
        if '\u4e00' <= ch <= '\u9fff':
            return ch
    return str(s)[:2] if s else "?"

# ── Load Run2 ──
run2_data = {}
for mname, fpath in METHOD_FILES.items():
    with open(fpath) as f:
        data = json.load(f)
    run2_data[mname] = {}
    for item in data.get("adv_results", []):
        stem = Path(item["image_path"]).stem.replace("_adv", "")
        gt = item["ground_truth"]
        res = {}
        for vlm in RUN2_VLMS:
            if vlm in item.get("q1", {}):
                p = item["q1"][vlm].get("parsed", {})
                res[vlm] = (p.get("correct", False), extract_cjk(p.get("answer", "")))
        run2_data[mname][stem] = {"gt": gt, "results": res}

with open(list(METHOD_FILES.values())[0]) as f:
    src_data = json.load(f)
run2_source = {}
for item in src_data.get("source_results", []):
    stem = Path(item["image_path"]).stem
    gt = item["ground_truth"]
    res = {}
    for vlm in RUN2_VLMS:
        if vlm in item.get("q1", {}):
            p = item["q1"][vlm].get("parsed", {})
            res[vlm] = (p.get("correct", False), extract_cjk(p.get("answer", "")))
    run2_source[stem] = {"gt": gt, "results": res}

# ── Load Run1 ──
with open(RUN1 / "manifest_20260324_174525.json") as f:
    manifest = json.load(f)
entries = manifest["image_entries"]

run1_data = defaultdict(lambda: defaultdict(dict))
run1_source = defaultdict(dict)

for vlm_name in RUN1_VLMS:
    results_file = RUN1 / f"results_{vlm_name}.jsonl"
    if not results_file.exists():
        continue
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = str(record.get("custom_id", "")).strip()
            if len(custom_id) < 2:
                continue
            q_idx = int(custom_id[-1])
            img_idx = int(custom_id[:-1]) - 1
            if q_idx != 1 or img_idx < 0 or img_idx >= len(entries):
                continue
            info = entries[img_idx]
            stem = Path(info["path"]).stem.replace("_adv", "")
            gt = info["gt"]
            method_str = info["method"]
            category = info["category"]
            resp = record.get("response", {})
            body = resp.get("body", {})
            choices = body.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                answer = extract_cjk(content)
                correct = (answer == gt)
            else:
                answer = "?"
                correct = False
            if category == "source":
                run1_source[stem][vlm_name] = (correct, answer)
            else:
                for mkey, mstr in RUN1_METHOD_MAP.items():
                    if mstr == method_str:
                        run1_data[mkey][stem][vlm_name] = (correct, answer)
                        break

# ── Combine and score ──
all_stems = set(run2_source.keys())
for mname in METHOD_FILES:
    all_stems &= set(run2_data[mname].keys())
    if mname in run1_data:
        all_stems &= set(run1_data[mname].keys())
all_stems &= set(run1_source.keys())

print(f"Common stems: {len(all_stems)}")

scores = []
for stem in all_stems:
    gt = run2_source[stem]["gt"]
    
    # Combine source
    src_results = {}
    src_results.update(run2_source[stem]["results"])
    src_results.update(run1_source.get(stem, {}))
    
    # Collect all method results
    combined = {}
    valid = True
    for mname in METHOD_FILES:
        combined[mname] = {}
        if stem in run2_data[mname]:
            combined[mname].update(run2_data[mname][stem]["results"])
        if mname in run1_data and stem in run1_data[mname]:
            combined[mname].update(run1_data[mname][stem])
        if len(combined[mname]) < 5:
            valid = False
            break
    if not valid:
        continue
    
    # Protection rate per method (fraction of 5 VLMs confused)
    rates = {}
    for mname in METHOD_FILES:
        confused = sum(1 for v, (c, a) in combined[mname].items() if not c)
        rates[mname] = confused / len(combined[mname])
    
    rate_vals = list(rates.values())
    
    # KEY METRIC: variance of protection rates
    # We want maximum spread — ideally 0%, 20%, 40%, 60%, 80%, 100%
    variance = np.var(rate_vals)
    std_val = np.std(rate_vals)
    
    # Count unique rate levels (more = better diversity)
    unique_rates = len(set([round(r, 2) for r in rate_vals]))
    
    # Penalize cases where too many methods have the same rate
    from collections import Counter
    rate_counts = Counter([round(r, 2) for r in rate_vals])
    max_same = max(rate_counts.values())
    # Penalty: if 4+ methods have same rate, heavily penalize
    same_penalty = max(0, max_same - 2) * 0.3
    
    # Score: maximize variance + unique levels, minimize clustering
    score = variance + unique_rates * 0.2 - same_penalty
    
    scores.append({
        "score": score,
        "variance": variance,
        "std": std_val,
        "stem": stem,
        "gt": gt,
        "rates": rates,
        "combined": combined,
        "src_results": src_results,
        "unique_rates": unique_rates,
        "max_same": max_same,
    })

scores.sort(key=lambda x: x["score"], reverse=True)

method_order = ["ASPL", "Glaze", "AMP", "XTransfer", "Nightshade", "MMCoA"]

print(f"\nTotal scored: {len(scores)}")
print(f"\n{'='*110}")
print(f"Top 30 cases ranked by VARIANCE of protection rates (want max spread):")
print(f"{'='*110}")

for i, s in enumerate(scores[:30]):
    rv = [s["rates"][m] for m in method_order]
    print(f"\n#{i+1}: {s['stem']} (GT='{s['gt']}') var={s['variance']:.4f} std={s['std']:.3f} unique={s['unique_rates']} max_same={s['max_same']} score={s['score']:.4f}")
    
    print(f"  Rates: ", end="")
    for m in method_order:
        r = s["rates"].get(m, -1)
        pct = f"{r:.0%}"
        print(f"{m}={pct:>4s} ", end="")
    print()
    
    # Source
    print(f"  Source: ", end="")
    for v in ALL_VLMS:
        if v in s["src_results"]:
            c, a = s["src_results"][v]
            print(f"{v}:{'✓' if c else '✗'}({a}) ", end="")
    print()
    
    # Per-VLM details
    for v in ALL_VLMS:
        print(f"  {v:8s}: ", end="")
        for m in method_order:
            if v in s["combined"].get(m, {}):
                c, a = s["combined"][m][v]
                status = "✗" if c else "✓"  # ✓=protected, ✗=failed
                print(f"{m}:{status}({a}) ", end="")
        print()
