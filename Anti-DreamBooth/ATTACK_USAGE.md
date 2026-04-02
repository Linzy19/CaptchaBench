# 对抗攻击工具使用文档 — Anti-DreamBooth

> **AttackASPL.py** — Anti-DreamBooth ASPL 攻击（SD VAE 潜在空间 PGD）

---

## 目录

- [攻击原理](#攻击原理)
- [两种模式对比](#两种模式对比)
- [环境要求](#环境要求)
- [参数说明](#参数说明)
- [使用示例](#使用示例)
- [配对模式详解](#配对模式详解)
- [输出目录结构](#输出目录结构)
- [常见问题](#常见问题)

---

## 攻击原理

```
┌──────────────┐                    ┌──────────────────────────┐
│  原图 Source  │ ──── 输入 ────→   │   SD VAE 编码器（冻结）    │
└──────────────┘                    └──────────┬───────────────┘
                                               │ encode → latent
                                               ▼
┌──────────────┐    VAE 编码         ┌──────────────────────────┐
│  目标图 Target│ ── → ──→           │  MSE(src_latent,         │
└──────────────┘   固定目标潜在       │       target_latent)     │
                                    └──────────┬───────────────┘
                                               │ PGD 梯度更新
                                               ▼
                                    ┌──────────────────────────┐
                                    │  对抗图 Adversarial        │
                                    │  (VAE 潜在表示接近目标图)   │
                                    └──────────────────────────┘
```

**核心思路**：通过 Stable Diffusion 的 VAE 编码器，将原图的潜在表示驱动到与目标图
潜在表示相同的位置，从而保护原图不被 DreamBooth 微调正确学习。

**损失函数**：`MSE(vae.encode(src + delta).latent_dist.mean, target_latent)`

> 使用 `.latent_dist.mean`（而非 `.sample()`）确保梯度计算稳定，无随机噪声干扰。

---

## 两种模式对比

| 维度 | fast 模式（默认） | full 模式 |
|------|-----------------|-----------|
| **攻击方法** | 仅 VAE-PGD | 完整 ASPL（代理 UNet 迭代训练 + PGD） |
| **加载模型** | 仅 VAE 编码器 | 完整 SD 管线（VAE + UNet + TextEncoder） |
| **GPU 显存** | ≥ 4GB | ≥ 16GB |
| **攻击效果** | 快速、有效 | 更强，与原论文一致 |
| **适用场景** | 日常批量保护 | 需要最强保护强度时 |

---

## 环境要求

- Python 3.8+
- PyTorch（支持 CUDA）
- diffusers（`AutoencoderKL`）
- Pillow、tqdm、numpy
- **full 模式额外需要**：accelerate、transformers，以及原始 `attacks/aspl.py`

---

## 环境准备

> **注意**：本框架有两个 requirements 文件：
> - `requirements.txt`：原始 Anti-DreamBooth 训练环境（torch==1.13.1，用于完整 DreamBooth 训练流程）
> - `requirements_attack.txt`：`AttackASPL.py` 专用依赖（推荐，兼容 Python 3.10 + PyTorch 2.x）

### 推荐：使用 requirements_attack.txt（AttackASPL.py 专用）

#### 第 1 步：创建 conda 环境

```bash
conda create -n anti_dreambooth python=3.10 -y
conda activate anti_dreambooth
```

#### 第 2 步：安装 PyTorch（CUDA 11.8）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y
```

#### 第 3 步：安装 Python 依赖

```bash
cd Anti-DreamBooth
pip install -r requirements_attack.txt
```

#### 第 4 步：验证安装

```bash
python -c "import torch; import diffusers; print('OK, CUDA:', torch.cuda.is_available())"
```

### 依赖说明（requirements_attack.txt）

| 包 | 用途 | 模式 |
|----|------|------|
| `diffusers>=0.14.0` | SD VAE（AutoencoderKL）| fast + full |
| `Pillow` | 图像 I/O | 所有 |
| `numpy` | 数值计算 | 所有 |
| `tqdm` | 进度条 | 所有 |
| `accelerate>=0.16.0` | 分布式加速（aspl.py）| full 模式 |
| `transformers>=4.26.0` | CLIP text encoder | full 模式 |
| `ftfy` | 文本预处理 | full 模式 |
| `datasets>=2.10.0` | attacks/aspl.py 内部导入 | full 模式 |

---

## 参数说明

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `--source_dir` | str | — | ✅ | 原图目录（待保护的图像） |
| `--target_dir` | str | — | ✅ | 目标图目录（潜在空间对齐目标） |
| `--output_dir` | str | `./output_aspl` | ❌ | 输出目录 |
| `--sd_model` | str | `stabilityai/stable-diffusion-2-1` | ❌ | SD 模型名称或本地路径 |
| `--pgd_alpha` | float | `1/255` | ❌ | PGD 步长，[-1,1] 空间 |
| `--pgd_eps` | float | `0.05` | ❌ | L-inf 扰动上限，[-1,1] 空间 |
| `--pgd_steps` | int | 200 | ❌ | PGD 迭代步数 |
| `--mode` | str | `fast` | ❌ | 攻击模式：`fast` / `full` |
| `--match_mode` | str | `auto` | ❌ | 配对模式：`auto` / `single` / `random` |
| `--mixed_precision` | str | `fp16` | ❌ | 推理精度：`no` / `fp16` / `bf16` |
| `--seed` | int | 42 | ❌ | 随机种子 |

### 图像范围约定

Stable Diffusion 使用 [-1, 1] 范围（`pixel / 127.5 - 1.0`），与通常的 [0, 1] 不同。

- `--pgd_alpha` 和 `--pgd_eps` 均以 [-1, 1] 空间为单位
- 对应 [0, 1] 空间的换算：`eps_01 = pgd_eps / 2`（`0.05` ≈ `[0,1]` 空间的 `0.025`）

---

## 使用示例

### 1. fast 模式（默认）— 快速 VAE-PGD 保护

```bash
python AttackASPL.py \
    --source_dir ../Data_source/sample_source \
    --target_dir ../Data_source/sample_target \
    --output_dir ./output_aspl_fast
```

### 2. full 模式 — 完整 ASPL（最强保护）

```bash
python AttackASPL.py \
    --source_dir ../Data_source/sample_source \
    --target_dir ../Data_source/sample_target \
    --output_dir ./output_aspl_full \
    --mode full \
    --pgd_steps 200
```

### 3. 自定义参数

```bash
python AttackASPL.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --sd_model stabilityai/stable-diffusion-2-1 \
    --pgd_alpha 0.004 \
    --pgd_eps 0.05 \
    --pgd_steps 200 \
    --match_mode random \
    --seed 42
```

### 4. 使用本地 SD 模型

```bash
python AttackASPL.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --sd_model /path/to/local/stable-diffusion-2-1
```

### 5. BFloat16 精度（适合 A100/H100）

```bash
python AttackASPL.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --mixed_precision bf16
```

---

## 配对模式详解

| 模式 | 说明 |
|------|------|
| `auto`（默认） | 自动检测：目标图只有 1 张 → `single`；多张 → `random` |
| `single` | 所有原图都攻击向**同一张**目标图（取目标列表的第 1 张） |
| `random` | 每张原图**随机选择**一张目标图进行配对 |

---

## 输出目录结构

```
output_dir/
├── image_001_adv.png        # 对抗图片（fast 模式）
├── image_001_adv.json       # 对应注释文件（原图有 JSON 时自动拷贝）
├── image_002_adv.png
├── ...
├── noise-ckpt/              # full 模式产生的 ASPL checkpoint（中间文件）
└── match_info.json          # 攻击配对信息 + 参数 + 耗时统计
```

### match_info.json 示例（fast 模式）

```json
{
  "attack_type": "aspl_fast_vae_pgd",
  "method": "VAE_PGD",
  "sd_model": "stabilityai/stable-diffusion-2-1",
  "parameters": {
    "pgd_alpha": 0.00392156862745098,
    "pgd_eps": 0.05,
    "pgd_steps": 200,
    "mixed_precision": "fp16",
    "mode": "fast"
  },
  "pairs": [
    {
      "source": "image_001.png",
      "target": "target_001.png",
      "output": "image_001_adv.png"
    }
  ],
  "timing": {
    "vae_load_time_sec": 5.2,
    "attack_time_sec": 180.0,
    "total_time_sec": 185.2,
    "avg_time_per_image_sec": 18.0
  }
}
```

---

## 常见问题

### Q: pgd_eps = 0.05 对应多大的视觉扰动？

`pgd_eps` 以 [-1, 1] 图像空间为单位：
- `0.05` 对应约 6.375 / 255 的像素变化幅度（`0.05 × 0.5 × 255 ≈ 6.4`）
- 视觉上几乎不可察觉

### Q: fast 模式和 full 模式应该怎么选？

- 日常批量保护、快速实验 → 使用 **fast 模式**
- 需要最强保护效果、与原始 Anti-DreamBooth 论文对齐 → 使用 **full 模式**

### Q: full 模式提示 "Cannot import aspl.py" 怎么办？

确保以下文件存在且依赖已安装：
```
Anti-DreamBooth/attacks/aspl.py
```
```bash
pip install diffusers accelerate transformers
```

### Q: 显存不够怎么办？

- fast 模式下可改用 `--mixed_precision fp16` 降低显存
- 或减少 `--pgd_steps`（如从 200 降到 100）
- full 模式显存需求较大（≥ 16GB），建议优先使用 fast 模式
