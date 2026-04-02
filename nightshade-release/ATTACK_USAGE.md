# 对抗攻击工具使用文档 — nightshade-release

> **AttackNightshade.py** — Nightshade VAE 潜在空间投毒攻击

---

## 目录

- [攻击原理](#攻击原理)
- [两种攻击模式对比](#两种攻击模式对比)
- [环境要求](#环境要求)
- [参数说明](#参数说明)
- [使用示例](#使用示例)
- [配对模式详解](#配对模式详解)
- [JSON 注释文件格式](#json-注释文件格式)
- [输出目录结构](#输出目录结构)
- [常见问题](#常见问题)

---

## 攻击原理

```
┌──────────────┐                    ┌──────────────────────────┐
│  原图 Source  │ ──── 输入 ────→   │  SD VAE 编码器（fp16）     │
└──────────────┘                    └──────────┬───────────────┘
                                               │ src_latent
                                               ▼
                                    ┌──────────────────────────┐
┌──────────────┐    VAE 编码         │  L2 Loss                 │
│  目标图 Target│ ── → ──→           │  (adv_latent -           │
└──────────────┘   target_latent    │   target_latent).norm()  │
  [image 模式]                      └──────────┬───────────────┘
                                               │ 符号梯度下降
                                               │ 线性步长衰减
                                               ▼
                                    ┌──────────────────────────┐
                                    │  对抗图 Adversarial        │
                                    │  (VAE 潜在表示对齐目标图)   │
                                    └──────────────────────────┘
```

**核心思路**：通过符号梯度下降（FGSM-like）在原图上施加一个修饰符 `modifier`，
使 `src + modifier` 的 VAE 潜在表示与目标图的潜在表示之间的 L2 距离最小化，
从而"投毒"训练数据，干扰 DreamBooth / 其他扩散模型的微调。

**优化循环**（完整保留自 `opt.py`）：
```
for i in range(t_size):
    actual_step_size = step_size * (1 - i / t_size * (1 - 1/100))  # 线性衰减
    adv = clamp(modifier + source, -1, 1)
    loss = (vae.encode(adv).latent - target_latent).norm()
    grad = autograd(loss, modifier)
    modifier = modifier - sign(grad) * actual_step_size
    modifier = clamp(modifier, -max_change, max_change)
```

---

## 两种攻击模式对比

| 维度 | image 模式（默认） | concept 模式 |
|------|-------------------|-------------|
| **目标来源** | 直接使用目标图的 VAE 潜在表示 | 通过 SD 文本生成目标图，再编码 |
| **输入** | 原图 + 目标图 | 原图 + 目标 JSON（prompt + label） |
| **SD 生图** | 不需要 | 需要（通过 SD 文生图） |
| **GPU 显存** | ≥ 6GB | ≥ 8GB（需要运行 SD 文生图） |
| **与原始 Nightshade 的关系** | 本项目新增的高效模式 | 与原始 Nightshade 行为一致 |

---

## 环境要求

- Python 3.8+
- PyTorch（支持 CUDA，推荐 fp16）
- diffusers（`StableDiffusionPipeline`、`AutoencoderKL`）
- Pillow、tqdm、numpy、torchvision
- nightshade-release 框架（`opt.py` 中的 `PoisonGeneration`、`img2tensor`、`tensor2img`）

---

## 环境准备

### 第 1 步：创建 conda 环境

```bash
conda create -n nightshade python=3.10 -y
conda activate nightshade
```

### 第 2 步：安装 PyTorch（CUDA 11.8，推荐 fp16 支持）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y
```

### 第 3 步：安装 Python 依赖

```bash
cd nightshade-release
pip install -r requirements.txt
```

### 第 4 步：验证安装

```bash
# 验证 image 模式（必须）
python -c "from diffusers import AutoencoderKL; from einops import rearrange; print('image 模式 OK')"

# 验证 concept 模式（可选，需 >=8GB GPU）
python -c "from diffusers import StableDiffusionPipeline; from transformers import CLIPTokenizer; print('concept 模式 OK')"
```

### 模式与显存要求

| 模式 | 额外依赖 | GPU 显存 |
|------|----------|----------|
| `image`（默认） | 只需 VAE（AutoencoderKL） | ≥ 6GB |
| `concept` | 完整 SD Pipeline（transformers + accelerate）| ≥ 8GB |

### 依赖说明

| 包 | 用途 | 模式 |
|----|------|------|
| `diffusers>=0.14.0` | SD VAE / 完整 SD Pipeline | 所有 |
| `einops` | `opt.py` 内部 `from einops import rearrange` | 所有 |
| `Pillow` | 图像 I/O | 所有 |
| `numpy` | 数值计算 | 所有 |
| `tqdm` | 进度条 | 所有 |
| `transformers>=4.26.0` | CLIPTextModel / tokenizer | concept |
| `accelerate>=0.16.0` | 模型加载加速 | concept |
| `ftfy` | transformers 文本预处理 | concept |

---

## 参数说明

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `--source_dir` | str | — | ✅ | 原图目录（待投毒的图像） |
| `--target_dir` | str | — | ✅ | 目标目录（潜在空间对齐目标） |
| `--output_dir` | str | `./output_nightshade` | ❌ | 输出目录 |
| `--sd_model` | str | `stabilityai/stable-diffusion-2-1` | ❌ | SD 模型名称或本地路径 |
| `--eps` | float | 0.05 | ❌ | [0,1] 尺度的扰动预算 |
| `--steps` | int | 500 | ❌ | 优化步数 |
| `--mode` | str | `image` | ❌ | 攻击模式：`image` / `concept` |
| `--match_mode` | str | `auto` | ❌ | 配对模式：`auto` / `single` / `random` |
| `--resolution` | int | 512 | ❌ | 处理分辨率（SD 约定 512） |
| `--seed` | int | 2023 | ❌ | 随机种子 |

### eps 参数说明

`--eps` 以 [0,1] 图像空间为单位，内部转换为 [-1,1] 空间（`max_change = eps / 0.5`）：
- `0.05` → [-1,1] 空间的 `0.1` → 约 12.75 / 255 的像素变化
- 视觉上基本不可察觉

---

## 使用示例

### 1. image 模式（默认）— 直接使用目标图

```bash
python AttackNightshade.py \
    --source_dir ../Data_source/sample_source \
    --target_dir ../Data_source/sample_target \
    --output_dir ./output_nightshade
```

### 2. concept 模式 — 通过目标 JSON 文本生成目标图

```bash
python AttackNightshade.py \
    --source_dir ../Data_source/sample_source \
    --target_dir ../Data_source/sample_target \
    --output_dir ./output_nightshade_concept \
    --mode concept
```

### 3. 自定义参数

```bash
python AttackNightshade.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --eps 0.05 \
    --steps 500 \
    --match_mode random \
    --seed 2023
```

### 4. 使用本地 SD 模型

```bash
python AttackNightshade.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --sd_model /path/to/local/stable-diffusion-2-1
```

### 5. 更强投毒效果（增大扰动 + 更多步数）

```bash
python AttackNightshade.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --eps 0.1 \
    --steps 1000
```

---

## 配对模式详解

| 模式 | 说明 |
|------|------|
| `auto`（默认） | 自动检测：目标图只有 1 张 → `single`；多张 → `random` |
| `single` | 所有原图都攻击向**同一张**目标图（取目标列表的第 1 张） |
| `random` | 每张原图**随机选择**一张目标图进行配对 |

---

## JSON 注释文件格式

`concept` 模式下，程序从目标图片的同名 `.json` 文件提取概念文本：

```json
{
    "prompt": "Rocky mountain terrain with ancient Chinese character inscriptions",
    "annotations": [
        {"label": "镶"}
    ]
}
```

提取后的概念文本为：
`"Rocky mountain terrain with ancient Chinese character inscriptions Label: 镶"`

该文本随后传入 SD 生图管线，生成目标图像再进行 VAE 编码。

---

## 输出目录结构

```
output_dir/
├── image_001_adv.png        # 投毒后的对抗图片
├── image_001_adv.json       # 对应注释文件（原图有 JSON 时自动拷贝）
├── image_002_adv.png
├── ...
└── match_info.json          # 攻击配对信息 + 参数 + 耗时统计
```

### match_info.json 示例

```json
{
  "attack_type": "nightshade_vae_latent_poisoning",
  "method": "VAE_LatentOptimization",
  "sd_model": "stabilityai/stable-diffusion-2-1",
  "parameters": {
    "eps": 0.05,
    "steps": 500,
    "mode": "image"
  },
  "pairs": [
    {
      "source": "image_001.png",
      "target": "target_001.png",
      "output": "image_001_adv.png"
    }
  ],
  "timing": {
    "model_load_time_sec": 12.0,
    "attack_time_sec": 300.0,
    "total_time_sec": 312.0,
    "avg_time_per_image_sec": 30.0
  }
}
```

---

## 常见问题

### Q: image 模式和 concept 模式应该怎么选？

- **image 模式**（推荐）：直接提供目标图，更灵活，无需运行 SD 文生图，
  适合所有情况
- **concept 模式**：适合需要与原始 Nightshade 行为完全对齐的场景，
  要求目标目录中有对应的 `.json` 注释文件

### Q: 投毒后的图像视觉质量如何保证？

通过 `--eps` 控制最大扰动幅度：
- `0.05`（默认）：肉眼几乎不可见
- `0.1`：轻微可见但不影响观感
- 超过 `0.2`：可能产生明显噪点

### Q: 需要 GPU 吗？

是的，VAE 编码器使用 fp16 推理，建议使用 GPU。
CPU 也可运行但速度极慢（每张图可能需要数分钟）。

### Q: steps 设多少合适？

默认 500 步与原始 Nightshade 论文一致，是质量与速度的平衡点。
如需更快速度可降至 200~300 步，如需更强效果可增至 1000 步。
