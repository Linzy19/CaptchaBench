# CaptchaBench ACM MM 2026 — Figure Script Mapping

All Python scripts here generate the figures used in the paper.
Script names match the figure numbers they produce.

## Main Paper Figures

| Script | Output PDF | Section | Description |
|--------|-----------|---------|-------------|
| `figmm_fig1_teaser.py` | `figures/fig1_teaser.pdf` | §1 Introduction | Teaser: 5-VLM comparison with Gemini TVR anomaly highlight |
| `figmm_fig2_pipeline_pptx.py` | `figures/fig2_pipeline.pdf` | §3.2 Benchmark | Dataset construction pipeline (ID ControlNet + SDXL) |
| `figmm_fig3_radar.py` | `figures/fig3_modality_radar.pdf` | §4.1 Baseline Eval | Per-VLM radar: CRR/1-TVR/ASR for Qwen-VL, Gemini-3.0, GPT-5.2 |
| `figmm_fig4_pareto.py` | `figures/fig4_pareto_efficiency.pdf` | §4.1 Baseline Eval | Quality–Effectiveness Pareto Frontier (ID + SDXL, 6 methods) |

## Appendix Figures

| Script | Output PDF(s) | Section | Description |
|--------|--------------|---------|-------------|
| `figmm_fig3_radar.py` | `figures/fig7_radar_appendix.pdf` | App S6 | Full 5-VLM radar appendix (Kimi 2.5 + GLM-4V extension) |
| `figmm_fig6a_stroke_line.py` | `figures/fig6a_stroke_line.pdf` | App S5 | Stroke complexity line chart (ASR vs stroke-count bin) |
| `figmm_fig6b_stroke_heatmap.py` | `figures/fig6b_stroke_heatmap.pdf` | App S5 | Stroke complexity heatmap (method × stroke-bin × VLM) |
| `figmm_fig7_radar.py` | `figures/fig7_radar_appendix.pdf` | App S6 | Alternative radar appendix (older version) |
| `figmm_fig8_vlm_bar_v2.py` | `figures/fig8_vlm_bar_ID.pdf` + `fig8_vlm_bar_SDXL.pdf` | App S3 | Per-VLM ASR grouped bar charts (ID and SDXL separately) |
| `figmm_fig9a_case_kui.py` | `figures/fig9a_case_kui.pdf` | App S8 | Case study: 揆 character (22 strokes, enclosure type) |
| `figmm_fig9b_case_jian.py` | `figures/fig9b_case_jian.pdf` | App S8 | Case study: 剑 character (9 strokes, left-right type) |
| `figmm_fig10_case_study_simple.py` | `figures/fig10_case_study_simple.pdf` | App S8 | Case study: simple characters (6+11 strokes) — all protected |
| `figmm_fig11_case_study_stroke.py` | `figures/fig11_case_study_stroke.pdf` | App S8 | Stroke-complexity case study across all 5 bins |

## Analysis Scripts (no figure output)

| Script | Purpose |
|--------|---------|
| `compute_table4mm.py` | Compute modality-group analysis tables (Tab S2) |
| `_find_diverse_case.py` | Helper: find diverse case study examples from eval results |

## Unused/Legacy Scripts

| Script | Status | Notes |
|--------|--------|-------|
| `figmm_fig5_vlm_bar.py` | Legacy | Single-panel version, replaced by figmm_fig8_vlm_bar_v2.py |
| `figmm_fig5_quality.py` | Legacy | Quality assessment figure (no longer in main paper) |
| `figmm_fig5_quality_split.py` | Legacy | Quality split version |
| `figmm_fig6_stroke.py` | Legacy | Combined stroke figure, replaced by fig6a + fig6b |
| `figmm_fig7_radar.py` | Legacy | Older radar version, superseded by figmm_fig3_radar.py |
| `fig2_pipeline_v1.py` | Legacy | First version of pipeline figure |
| `figmm_case_study_grid.py` | Legacy | Old case study grid layout |
| `figmm_case_study.py` | Legacy | Old case study script |
| `figmm1_modality_radar.py` | Legacy | Old modality radar version |

## Note on figmm_fig3_radar.py

This single script generates TWO figures:
1. `fig3_modality_radar.pdf` — 3-VLM version for main text (Qwen-VL, Gemini-3.0, GPT-5.2)
2. `fig7_radar_appendix.pdf` — full 5-VLM version for appendix
