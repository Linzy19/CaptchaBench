# 对抗攻击参数说明文档

> 本文档说明 `run_all_attacks.sh` 中每个攻击方法的默认参数来源、论文依据，以及参数的物理含义。
>
> **设计原则**：不传 `--epsilon`/`--steps` 时，各方法使用各自论文中的实验参数；
> 传入后作为全局覆盖值，用于横向比较实验。

---

## 目录

- [参数总览](#参数总览)
- [AttackMI — MI_CommonWeakness 迁移攻击](#1-attackmi--mi_commonweakness-迁移攻击)
- [AttackASPL — Anti-DreamBooth 潜在空间保护](#2-attackaspl--anti-dreambooth-潜在空间保护)
- [AttackMMCoA — 多模态联合攻击](#3-attackmmcoa--多模态联合攻击)
- [AttackNightshade — 数据投毒攻击](#4-attacknightshade--数据投毒攻击)
- [AttackXTransfer — 超迁移通用对抗扰动](#5-attackxtransfer--超迁移通用对抗扰动)
- [AttackVLM — VLM文字级攻击（Attack-Bard）](#6-attackvlm--vlm文字级攻击attack-bard)
- [AttackVLMNew — LAVIS视觉编码器迁移攻击](#7-attackvlmnew--lavis视觉编码器迁移攻击)
- [参数空间换算说明](#参数空间换算说明)
- [横向比较实验建议](#横向比较实验建议)

---

## 参数总览

| 方法 | epsilon（论文值） | steps（论文值） | step_size（论文值） | 参数空间 | 论文 |
|------|-----------------|----------------|--------------------|---------|----|
| **MI** | 16/255 ≈ **16**（0-255） | **300**（高质量） | 1/255 ≈ **1** | 0-255像素尺度 | CVPR2018 + ICLR2024 |
| **ASPL** | **0.05**（[-1,1]空间） | **200** | **0.005**（[-1,1]空间） | [-1,1]归一化空间 | ICCV2023 |
| **MMCoA** | 1/255 ≈ **1**（0-255） | **100** | 1/255 ≈ **1** | 0-255像素尺度 | arXiv2404 |
| **Nightshade** | **0.05**（[0,1]空间） | **500** | 衰减步长（初始≈0.1） | [0,1]归一化空间 | Oakland2024 |
| **XTransfer** | 12/255 ≈ **12**（0-255） | **300** | 0.5/255 ≈ **0.5** | 0-255像素尺度 | ICML2025 |
| **VLM** | 8/255 ≈ **8**（0-255） | **300** | 1/255 ≈ **1** | 0-255像素尺度 | NeurIPS2023 |
| **AttackVLMNew** | 8/255 ≈ **8**（0-255） | **300** | 1/255 ≈ **1** | 0-255像素尺度 | NeurIPS2023参考 |

> **为什么 MMCoA 的 epsilon=1 远小于其他方法的 16？**
> MMCoA 针对 CLIP 多模态空间设计。在 CLIP 的视觉-语义对齐空间中，
> 1/255 的微小扰动即可产生显著的语义偏移，使用更大扰动反而会破坏图像视觉质量。

---

## 1. AttackMI — MI_CommonWeakness 迁移攻击

**目录**：`AdversarialAttacks/`
**脚本**：`AttackMI.py`
**conda 环境**：`adv_attack`

### 论文来源

- **MI-FGSM**：*Boosting Adversarial Attacks with Momentum*, Dong et al., **CVPR 2018**
- **MI_CommonWeakness (CWA)**：*Rethinking Model Ensemble in Transfer-based Adversarial Attacks*, Chen et al., **ICLR 2024**, arXiv:2303.09105

### 论文实验参数

| 参数 | 论文值 | 代码实际使用值 | 说明 |
|------|--------|--------------|------|
| `--epsilon` | 16/255 | **16**（0-255尺度） | L∞扰动上界，图像分类标准benchmark |
| `--steps` | 10（MI-FGSM原始）| **300**（高质量CWA设置） | 步数多→攻击质量高，代码使用更高质量设置 |
| `--step_size` | ε/T ≈ 1.6/255 | **1**（0-255尺度） | 外层PGD步长 |
| `--inner_step_size` | 250 | **250** | MI内层动量步长（CWA框架参数） |
| `--reverse_step_size` | 1 | **1** | 反向扰动步长 |

### 参数选择说明

MI-FGSM 原始论文仅需 10 步，但 MI_CommonWeakness (CWA) 框架在高质量实验中使用 300 步，
可获得更好的迁移攻击效果。`epsilon=16/255` 是图像对抗攻击领域最广泛使用的标准 benchmark 值。

### run_all_attacks.sh 调用方式

```bash
# 使用论文默认参数（epsilon=16, steps=300）
run_mi "" ""

# 全局覆盖示例
run_mi 8 100
```

---

## 2. AttackASPL — Anti-DreamBooth 潜在空间保护

**目录**：`Anti-DreamBooth/`
**脚本**：`AttackASPL.py`
**conda 环境**：`anti_dreambooth`

### 论文来源

- **Anti-DreamBooth ASPL**：*Anti-DreamBooth: Protecting Users from Personalized Text-to-Image Synthesis*, Van Le et al., **ICCV 2023**, arXiv:2303.15433
- 官方代码：[VinAIResearch/Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth)，脚本 `scripts/attack_with_aspl.sh`

### 论文实验参数

| 参数 | 论文值 | 代码实际使用值 | 说明 |
|------|--------|--------------|------|
| `--pgd_eps` | **0.05** | **0.05**（[-1,1]空间） | L∞扰动上界，直接使用论文值 |
| `--pgd_steps` | **200** | **200** | PGD迭代步数 |
| `--pgd_alpha` | **0.005** | **0.005**（[-1,1]空间） | PGD步长，论文固定值 5×10⁻³ |

### 参数空间说明

ASPL 使用 **[-1, 1] 归一化图像空间**（Stable Diffusion约定），而非 [0,1] 或 0-255：
- `pgd_eps=0.05` 对应 [0,1] 空间约 0.025，对应 0-255 尺度约 **6.4 像素**
- `pgd_alpha=0.005` 对应 0-255 尺度约 **0.64 像素**

当用户传入全局 `--epsilon`（0-255尺度）时，脚本自动换算：`pgd_eps = epsilon / 255`；
**不传时直接使用论文值 0.05**，不做任何换算。

### run_all_attacks.sh 调用方式

```bash
# 使用论文默认参数（pgd_eps=0.05, pgd_steps=200, pgd_alpha=0.005）
run_aspl "" ""

# 全局覆盖示例（传入 epsilon=16，换算为 pgd_eps=16/255≈0.0627）
run_aspl 16 200
```

---

## 3. AttackMMCoA — 多模态联合攻击

**目录**：`MMCoA/`
**脚本**：`AttackMMCoA.py`
**conda 环境**：`mmcoa`

### 论文来源

- **MMCoA**：*Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective*, arXiv:2404.19287
- 官方代码：[ElleZWQ/MMCoA](https://github.com/ElleZWQ/MMCoA)

### 论文实验参数

| 参数 | 论文值 | 代码实际使用值 | 说明 |
|------|--------|--------------|------|
| `--epsilon` | **1/255** | **1**（0-255尺度） | L∞扰动上界，CLIP空间特定小扰动设置 |
| `--num_iters` | **100**（测试阶段） | **100** | 攻击迭代步数 |
| `--step_size` | 1/255 | **1**（0-255尺度） | PGD步长 |
| `--clip_model` | ViT-B/32 | **ViT-B/32** | CLIP代理模型 |

### 为什么 epsilon=1 远小于其他方法？

MMCoA 的攻击目标是 **CLIP 的多模态对齐空间**，而非传统图像分类空间。
论文发现，在 CLIP 的视觉-文本联合嵌入空间中，1/255 的微小像素扰动即可产生
显著的语义偏移效果。论文额外测试了 ε ∈ {1/255, 2/255, 4/255}。

若需要与其他方法在相同扰动强度下比较，可使用全局覆盖参数 `--epsilon 16`。

### run_all_attacks.sh 调用方式

```bash
# 使用论文默认参数（epsilon=1, num_iters=100）
run_mmcoa "" ""

# 横向比较模式（统一扰动强度）
run_mmcoa 16 300
```

---

## 4. AttackNightshade — 数据投毒攻击

**目录**：`nightshade-release/`
**脚本**：`AttackNightshade.py`
**conda 环境**：`nightshade`

### 论文来源

- **Nightshade**：*Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models*, Shan et al., **IEEE S&P (Oakland) 2024**, arXiv:2310.13828
- 官方代码：[Shawn-Shan/nightshade-release](https://github.com/Shawn-Shan/nightshade-release)

### 论文实验参数

| 参数 | 论文值 | 代码实际使用值 | 说明 |
|------|--------|--------------|------|
| `--eps` | **0.05**（Linf，[0,1]空间） | **0.05** | 扰动上界，release版本改为Linf实现 |
| `--steps` | **500** | **500** | 优化步数，500步是论文核心设置 |
| 步长 | 衰减步长（初始≈eps/0.5=0.1） | 由代码内部计算 | 从初始值线性衰减到0.001 |

### 原论文与 release 版本的差异

| 版本 | 扰动度量 | 参数值 | 说明 |
|------|---------|--------|------|
| **论文** | LPIPS 感知距离 | budget p=0.07 | 符合人眼感知 |
| **release代码** | L∞ norm | eps=0.05（[0,1]空间） | 注释明确说明 "uses Linf perturbation" |

代码使用 release 版本的 Linf 实现。`eps=0.05` 对应 0-255 尺度约 **12.75 像素**。

### 为什么 steps=500？

Nightshade 是**数据投毒**攻击，不追求实时性，追求高质量的投毒样本。
论文报告每张图像生成耗时约 94 秒（Titan RTX GPU），500步是效果与时间的最优平衡点。

### run_all_attacks.sh 调用方式

```bash
# 使用论文默认参数（eps=0.05, steps=500）
run_nightshade "" ""

# 全局覆盖示例（传入 epsilon=16，换算为 ns_eps=16/255≈0.0627）
run_nightshade 16 500
```

---

## 5. AttackXTransfer — 超迁移通用对抗扰动

**目录**：`XTransferBench/`
**脚本**：`AttackXTransfer.py`
**conda 环境**：`xtransfer`

### 论文来源

- **XTransferBench**：*X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP*, Huang et al., **ICML 2025**, arXiv:2505.05528
- 官方代码：[HanxunH/XTransferBench](https://github.com/HanxunH/XTransferBench)

### 论文实验参数

| 参数 | 论文值 | 代码实际使用值 | 说明 |
|------|--------|--------------|------|
| `--epsilon` | **12/255** | **12**（0-255尺度） | L∞扰动上界，模型名中直接体现（`_linf_eps12_`） |
| `--steps` | **300** | **300** | PGD步数 |
| `--step_size` | **0.5/255** | **0.5**（0-255尺度） | 步长较小（eps/24），保证细粒度优化 |
| `--clip_model` | ViT-B-32 | **ViT-B-32** | OpenCLIP模型，LAION-2B预训练 |
| `--clip_pretrained` | laion2b_s34b_b79k | **laion2b_s34b_b79k** | 预训练权重 |

### 为什么 epsilon=12 而非常见的 16？

XTransferBench 的 SOTA 模型（`xtransfer_large_linf_eps12_non_targeted`）使用 **12/255**。
这是该框架针对 CLIP 迁移攻击场景实验确定的最优扰动预算，略小于传统 16/255 但迁移效果更好。

### run_all_attacks.sh 调用方式

```bash
# 使用论文默认参数（epsilon=12, steps=300）
run_xtransfer "" ""

# 全局覆盖示例
run_xtransfer 16 300
```

---

## 6. AttackVLM — VLM文字级攻击（Attack-Bard）

**目录**：`Attack-Bard/`
**脚本**：`AttackVLM.py`
**conda 环境**：`attack_bard`

### 论文来源

- **AttackVLM**：*On Evaluating Adversarial Robustness of Large Vision-Language Models*, Zhao et al., **NeurIPS 2023**, arXiv:2305.16934
- 官方代码：[yunqing-me/AttackVLM](https://github.com/yunqing-me/AttackVLM)

### 论文实验参数

| 参数 | 论文值 | 代码实际使用值 | 说明 |
|------|--------|--------------|------|
| `--epsilon` | **8/255** | **8**（0-255尺度） | L∞扰动上界，白盒迁移阶段 |
| `--steps` | **300**（白盒迁移阶段） | **300** | 白盒迁移攻击步数 |
| `--step_size` | 1/255 | **1**（0-255尺度） | PGD步长 |
| 代理模型 | BLIP2 + InstructBLIP + MiniGPT4 | 三者集成 | VLM代理模型集成 |

### 攻击特殊性

AttackVLM **无 `--target_dir` 参数**，攻击目标是文字而非图像：
- 使用 `--use_json_text` 从 `source_dir` 下每张图像的同名 `.json` 文件读取目标文字
- JSON 格式：`{"prompt": "...", "annotations": [{"label": "贞"}]}`
- 目标文字格式：`"{prompt} Label: {label}"`

### 两阶段攻击流程（论文完整方法）

| 阶段 | 方法 | epsilon | steps |
|------|------|---------|-------|
| 1. 白盒迁移 | SSA_CommonWeakness（BLIP2+InstructBLIP+MiniGPT4） | 8/255 | 300 |
| 2. 黑盒精调 | RGF随机无梯度估计器 | 8/255 | 8（query×100） |

代码实现的是第1阶段（白盒迁移），这也是论文中最核心的技术贡献。

### run_all_attacks.sh 调用方式

```bash
# 使用论文默认参数（epsilon=8, steps=300，目标文字从JSON读取）
run_vlm "" ""

# 全局覆盖示例
run_vlm 16 500
```

---

## 7. AttackVLMNew — LAVIS视觉编码器迁移攻击

**目录**：`AttackVLM/`
**脚本**：`AttackVLMNew.py`
**conda 环境**：`attackvlm`

### 方法来源

- **LAVIS框架**：[salesforce/LAVIS](https://github.com/salesforce/LAVIS)，基于 BLIP/BLIP2 视觉编码器
- **攻击原理**：基于 AttackVLM 论文（NeurIPS2023）的迁移攻击策略，使用 LAVIS BLIP/BLIP2 作为代理模型

### 参数设置

| 参数 | 使用值 | 说明 |
|------|--------|------|
| `--epsilon` | **8**（0-255尺度） | 参考 AttackVLM 论文（NeurIPS2023）epsilon=8/255 |
| `--steps` | **300** | 白盒迁移攻击步数，与 AttackVLM 保持一致 |
| `--step_size` | **1**（0-255尺度） | PGD步长 |
| `--model_name` | **blip_caption** | 默认使用 BLIP（base_coco预训练），更轻量 |
| `--model_type` | **base_coco** | BLIP的MS-COCO预训练模型 |

### 与 AttackVLM（Attack-Bard）的区别

| 维度 | AttackVLM（Attack-Bard） | AttackVLMNew（本方法） |
|------|--------------------------|----------------------|
| 攻击目标 | **文字**（让VLM输出目标文字） | **图像**（让编码特征接近目标图） |
| 代理模型 | BLIP2 + InstructBLIP + MiniGPT4 | BLIP / BLIP2（LAVIS） |
| 目标输入 | 目标文字（JSON） | 目标图像（target_dir） |
| 损失函数 | 负VLM loss（文字似然） | cosine similarity（特征对齐） |
| GPU显存 | ≥ 16GB | ≥ 8GB |

### run_all_attacks.sh 调用方式

```bash
# 使用默认参数（epsilon=8, steps=300）
run_attackvlm_new "" ""

# 全局覆盖示例
run_attackvlm_new 16 300
```

---

## 参数空间换算说明

不同方法使用不同的参数空间，`run_all_attacks.sh` 中的全局 `--epsilon` 参数采用 **0-255 像素尺度**，
传给各方法时自动换算：

| 方法 | 参数空间 | 换算公式 | 全局 epsilon=16 对应值 |
|------|---------|---------|----------------------|
| MI | 0-255（直接传入） | epsilon_code = epsilon | 16 |
| ASPL | [-1,1]（SD约定） | pgd_eps = epsilon / 255 | 0.0627 |
| MMCoA | 0-255（直接传入） | epsilon_code = epsilon | 16 |
| Nightshade | [0,1] | ns_eps = epsilon / 255 | 0.0627 |
| XTransfer | 0-255（直接传入） | epsilon_code = epsilon | 16 |
| VLM | 0-255（直接传入） | epsilon_code = epsilon | 16 |
| AttackVLMNew | 0-255（直接传入），内部再 /std | epsilon_code = epsilon | 16 |

> **注意**：ASPL 和 Nightshade 的论文默认值（0.05）**不等于** 全局 epsilon=16 换算后的值（0.0627）。
> 因此，**不传全局 epsilon 时，ASPL 和 Nightshade 使用论文原始值 0.05**（更准确）。

---

## 横向比较实验建议

如果需要在**相同扰动强度**下比较各方法，建议：

```bash
# 统一使用 epsilon=16（0-255尺度），steps=300
bash run_all_attacks.sh \
    --source_dir /path/to/source \
    --target_dir /path/to/target \
    --epsilon 16 \
    --steps 300 \
    --output_base ./outputs_compare_eps16

# 统一使用 epsilon=8（更接近VLM方法的论文设置）
bash run_all_attacks.sh \
    --source_dir /path/to/source \
    --target_dir /path/to/target \
    --epsilon 8 \
    --steps 300 \
    --output_base ./outputs_compare_eps8
```

如果使用**各方法论文推荐参数**（默认，推荐用于评估各方法的原始能力）：

```bash
bash run_all_attacks.sh \
    --source_dir /path/to/source \
    --target_dir /path/to/target \
    --output_base ./outputs_paper_params
# 此时不传 --epsilon 和 --steps，各方法使用论文默认值
```

### 各方法论文默认参数汇总（快速参考）

```
MI        : epsilon=16, steps=300, step_size=1
ASPL      : pgd_eps=0.05([-1,1]空间), pgd_steps=200, pgd_alpha=0.005
MMCoA     : epsilon=1,  num_iters=100, step_size=1
Nightshade: eps=0.05([0,1]空间), steps=500
XTransfer : epsilon=12, steps=300, step_size=0.5
VLM       : epsilon=8,  steps=300, step_size=1
AttackVLMNew: epsilon=8, steps=300, step_size=1
```

---

## 参考文献

1. Dong, Y. et al. *Boosting Adversarial Attacks with Momentum*. CVPR 2018.
2. Chen, Z. et al. *Rethinking Model Ensemble in Transfer-based Adversarial Attacks*. ICLR 2024. arXiv:2303.09105
3. Van Le, T. et al. *Anti-DreamBooth: Protecting Users from Personalized Text-to-Image Synthesis*. ICCV 2023. arXiv:2303.15433
4. Zou, E. et al. *Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective*. arXiv:2404.19287
5. Shan, S. et al. *Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models*. IEEE S&P 2024. arXiv:2310.13828
6. Huang, H. et al. *X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP*. ICML 2025. arXiv:2505.05528
7. Zhao, Y. et al. *On Evaluating Adversarial Robustness of Large Vision-Language Models*. NeurIPS 2023. arXiv:2305.16934
