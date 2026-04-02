# 对抗攻击工具使用文档 — XTransferBench

> **AttackXTransfer.py** — 基于 CLIP 的对抗攻击（逐图 PGD + 通用对抗扰动 UAP）

---

## 目录

- [攻击原理](#攻击原理)
- [两种攻击模式对比](#两种攻击模式对比)
- [两种攻击类型对比](#两种攻击类型对比)
- [环境要求](#环境要求)
- [参数说明](#参数说明)
- [使用示例](#使用示例)
- [配对模式详解](#配对模式详解)
- [JSON 注释文件格式](#json-注释文件格式)
- [输出目录结构](#输出目录结构)
- [常见问题](#常见问题)

---

## 攻击原理

### per_image 模式（逐图 PGD）

```
┌──────────────┐                    ┌──────────────────────────┐
│  原图 Source  │ ──── 输入 ────→   │  OpenCLIP ViT-B-32        │
└──────────────┘                    │  encode_image()           │
                                    │                          │
┌───────────────────────┐    CLIP   │  F.normalize(..., dim=-1) │
│  目标 JSON 文本         │ 文本编码→ │  encode_text()            │
│  (prompt + label)     │          └──────────────┬───────────┘
└───────────────────────┘                         │
                                                  │ 最大化余弦相似度
                                                  │（定向攻击）
                                                  ▼
                                       ┌──────────────────────┐
                                       │  对抗图 Adversarial    │
                                       │  (CLIP 视觉特征        │
                                       │   对齐目标文本嵌入)     │
                                       └──────────────────────┘
```

**逐图 PGD 核心公式**：
```
target_emb  = F.normalize(encode_text(target_text), dim=-1)
delta = random_init(-ε, ε)
for t in range(steps):
    adv_emb = F.normalize(encode_image(normalize(img + delta)), dim=-1)
    loss    = -cosine_sim(adv_emb, target_emb)   # 定向：最大化余弦相似度
    delta   = delta - step_size * sign(grad(loss))
    delta   = clamp(delta, -ε, ε)
```

### uap 模式（通用对抗扰动）

```
┌───────────────┐                   ┌──────────────────────────┐
│ 所有原图       │   遍历每张图       │  共享 delta（通用扰动）    │
│ (N 张图像)     │ ─────────────→   │  shape: [1, 3, H, W]     │
└───────────────┘                   └──────────────┬───────────┘
                                                   │
                                    ΣLoss / N 再反向传播
                                                   │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │  所有图像共享同一 delta    │
                                    │  {name}_uap_adv.png      │
                                    └──────────────────────────┘
```

**UAP 外层循环**（与 `generate_universal_perturbation.py` 对齐）：
```
delta = random_init(-ε, ε)     # 全局共享，shape [1,3,H,W]
for epoch in range(uap_epochs):
    for step in range(steps_per_epoch):
        total_loss = Σ_i -cos_sim(encode_image(img_i + delta), target_emb_i)
        delta = delta - step_size * sign(grad(total_loss / N))
        delta = clamp(delta, -ε, ε)
```

---

## 两种攻击模式对比

| 维度 | per_image 模式（默认） | uap 模式 |
|------|-----------------------|---------|
| **delta 范围** | 每张图独立的 delta | 所有图共享一个通用 delta |
| **训练策略** | 逐图单独 PGD 优化 | 跨图联合 PGD 训练 |
| **输出文件名** | `{name}_adv.png` | `{name}_uap_adv.png` |
| **额外输出** | — | `uap_delta.pth`（通用扰动张量） |
| **攻击效果** | 每张图最优 | 单一扰动对所有图通用 |
| **GPU 显存** | ≥ 4GB | ≥ 4GB（但一次前向传播所有图） |
| **适用场景** | 精准单图保护 | 批量发布前统一添加保护 |

---

## 两种攻击类型对比

| 维度 | targeted（定向，默认） | untargeted（非定向） |
|------|-----------------------|---------------------|
| **目标** | 目标 JSON 文本嵌入 | 原始图像嵌入（推离） |
| **损失函数** | `-cos_sim(adv_img, text_target)` | `+cos_sim(adv_img, clean_img)` |
| **JSON 文件** | 必须提供 | 不需要（仅需源图） |
| **效果** | CLIP 视觉特征对齐目标文本 | CLIP 视觉特征远离原图 |

---

## 环境要求

- Python 3.8+
- PyTorch（支持 CUDA，推荐）
- open_clip_torch：`pip install open-clip-torch`
- torchvision、Pillow、tqdm、numpy

> **兼容性说明**：本脚本对 `encode_text / encode_image` 的输出使用 `F.normalize(..., dim=-1)` 手动归一化，**不使用** `normalize=True` 关键字参数，因此兼容所有版本的 open_clip_torch（旧版本不支持该参数）。

---

## 环境准备

### 第 1 步：创建 conda 环境

```bash
conda create -n xtransfer python=3.10 -y
conda activate xtransfer
```

### 第 2 步：安装 PyTorch（CUDA 11.8）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y
```

### 第 3 步：安装 Python 依赖

```bash
cd XTransferBench
pip install -r requirements.txt
```

> **说明**：`requirements.txt` 与根目录的 `pyproject.toml` 并存。
> `pyproject.toml` 用于包发布，`requirements.txt` 用于开发环境快速复现，
> 运行 `AttackXTransfer.py` 只需使用 `requirements.txt`。

### 第 4 步：验证安装

```bash
python -c "import open_clip; print('open_clip OK, version:', open_clip.__version__)"
```

### 依赖说明

| 包 | 用途 |
|----|------|
| `open_clip_torch>=2.20.0` | OpenCLIP ViT-B-32 等模型 |
| `Pillow` | 图像 I/O |
| `numpy` | 数值计算 |
| `tqdm` | 进度条 |

> **兼容性说明**：本脚本使用 `F.normalize` 手动归一化，不依赖 `encode_text(normalize=True)` 参数，兼容所有版本的 `open_clip_torch`。

---

## 参数说明

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `--source_dir` | str | — | ✅ | 原图目录（待扰动的图像） |
| `--target_dir` | str | — | ✅ | 目标目录（包含 `.json` 注释文件） |
| `--output_dir` | str | `./output_xtransfer` | ❌ | 输出目录 |
| `--mode` | str | `per_image` | ❌ | 攻击模式：`per_image` / `uap` |
| `--attack_type` | str | `targeted` | ❌ | 攻击类型：`targeted` / `untargeted` |
| `--clip_model` | str | `ViT-B-32` | ❌ | OpenCLIP 模型名称 |
| `--clip_pretrained` | str | `laion2b_s34b_b79k` | ❌ | OpenCLIP 预训练权重名称 |
| `--epsilon` | float | `16` | ❌ | L-inf 扰动上限（0-255 尺度） |
| `--step_size` | float | `1` | ❌ | PGD 步长（0-255 尺度） |
| `--steps` | int | `300` | ❌ | PGD 迭代步数 / UAP 每轮步数 |
| `--uap_epochs` | int | `5` | ❌ | UAP 训练轮数（仅 `uap` 模式） |
| `--image_size` | int | `224` | ❌ | 输入图像尺寸（宽=高） |
| `--match_mode` | str | `auto` | ❌ | 配对模式：`auto` / `name` / `order` |
| `--seed` | int | `7` | ❌ | 随机种子 |
| `--log_interval` | int | `50` | ❌ | 每隔 N 步打印一次损失 |

### epsilon 参数说明

`--epsilon` 以 [0, 255] 像素尺度为单位，内部自动转换为 [0, 1] 空间（`epsilon_01 = epsilon / 255`）：
- `16`（默认）→ 16/255 ≈ 0.063，视觉上可见但通常可接受
- `8` → 8/255 ≈ 0.031，较隐蔽
- `32` → 32/255 ≈ 0.125，更强攻击效果

---

## 使用示例

### 1. 逐图定向攻击（默认）

```bash
python AttackXTransfer.py \
    --source_dir ../Data_source/sample_source \
    --target_dir ../Data_source/sample_target \
    --output_dir ./output_xtransfer
```

### 2. UAP 模式 — 通用对抗扰动

```bash
python AttackXTransfer.py \
    --source_dir ../Data_source/sample_source \
    --target_dir ../Data_source/sample_target \
    --output_dir ./output_xtransfer_uap \
    --mode uap \
    --uap_epochs 5
```

### 3. 非定向攻击（推离原图特征）

```bash
python AttackXTransfer.py \
    --source_dir ../Data_source/sample_source \
    --target_dir ../Data_source/sample_target \
    --output_dir ./output_xtransfer_untargeted \
    --attack_type untargeted
```

### 4. 更换 CLIP 模型（更强特征）

```bash
python AttackXTransfer.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --clip_model ViT-L-14 \
    --clip_pretrained laion2b_s32b_b82k
```

### 5. 自定义攻击强度

```bash
python AttackXTransfer.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --epsilon 16 \
    --step_size 1 \
    --steps 300 \
    --seed 7
```

### 6. 按文件名严格匹配 + 高强度攻击

```bash
python AttackXTransfer.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --match_mode name \
    --epsilon 32 \
    --steps 500 \
    --step_size 2
```

---

## 配对模式详解

与其他攻击脚本不同，XTransfer 攻击目标为**文本**（来自 JSON 文件），配对逻辑如下：

| 模式 | 说明 |
|------|------|
| `auto`（默认） | 先按**文件名**匹配（`image_001.png` → `image_001.json`）；无法匹配的原图按**顺序位置**回退到剩余 JSON |
| `name` | 严格**文件名**匹配；找不到同名 JSON 的原图直接跳过 |
| `order` | 按**排序后的位置**逐一对齐（第 1 张原图配第 1 个 JSON，以此类推） |

---

## JSON 注释文件格式

`--target_dir` 中的 `.json` 文件提供目标文本，格式如下：

```json
{
    "prompt": "Rocky mountain terrain with ancient Chinese character inscriptions",
    "annotations": [
        {"label": "水"},
        {"label": "山"}
    ]
}
```

提取后的目标文本为：
`"Rocky mountain terrain with ancient Chinese character inscriptions Label: 水, 山"`

该文本随后由 CLIP 文本编码器编码为目标嵌入，用于定向攻击。

---

## 输出目录结构

### per_image 模式

```
output_dir/
├── image_001_adv.png        # 逐图对抗图片
├── image_002_adv.png
├── ...
└── match_info.json          # 攻击配对信息 + 参数 + 耗时统计
```

### uap 模式

```
output_dir/
├── image_001_uap_adv.png    # 应用通用扰动后的图片
├── image_002_uap_adv.png
├── ...
├── uap_delta.pth            # 训练完成的通用扰动张量（可复用）
└── match_info.json          # 攻击配对信息 + 参数 + 耗时统计
```

### match_info.json 示例（per_image 定向攻击）

```json
{
  "attack_type": "xtransfer_clip_attack",
  "method": "PGD_CLIP_定向",
  "parameters": {
    "mode": "per_image",
    "attack_type": "targeted",
    "epsilon_255": 16.0,
    "step_size_255": 1.0,
    "steps": 300
  },
  "pairs": [
    {
      "source": "image_001.png",
      "target_text": "Rocky mountain terrain with ancient Chinese character inscriptions Label: 水",
      "output": "image_001_adv.png",
      "time_sec": 24.5
    }
  ],
  "timing": {
    "total_time_sec": 245.0,
    "total_images": 10,
    "success": 10,
    "failed": 0,
    "avg_time_per_image_sec": 24.5
  }
}
```

### match_info.json 示例（uap 模式）

```json
{
  "attack_type": "xtransfer_clip_attack",
  "method": "UAP_CLIP_定向",
  "parameters": {
    "mode": "uap",
    "attack_type": "targeted",
    "epsilon_255": 16.0,
    "step_size_255": 1.0,
    "steps_per_epoch": 300,
    "uap_epochs": 5,
    "image_size": 224
  },
  "pairs": [
    {
      "source": "image_001.png",
      "target_text": "Rocky mountain terrain Label: 水",
      "output": "image_001_uap_adv.png"
    }
  ],
  "timing": {
    "total_time_sec": 820.0,
    "uap_train_time_sec": 810.0,
    "apply_time_sec": 10.0,
    "total_images": 10,
    "success": 10,
    "failed": 0
  }
}
```

---

## 常见问题

### Q: per_image 模式和 uap 模式应该怎么选？

- **per_image 模式**（推荐）：每张图独立优化，攻击效果最强，适合需要对每张图精确保护的场景
- **uap 模式**：训练一个通用扰动，对整个图像集合有效，适合需要统一批量添加保护的场景（如发布整个数据集前），攻击效果略弱于逐图模式

### Q: targeted 和 untargeted 攻击分别适用于什么场景？

- **targeted（定向）**：将图像的 CLIP 特征拉向指定文本语义，干扰模型将图像与原始标签关联，是数据投毒保护的主要手段
- **untargeted（非定向）**：将图像的 CLIP 特征推离原图语义，破坏特征一致性，适合作为通用保护手段

### Q: open_clip 版本兼容性问题如何处理？

旧版本 `open_clip` 不支持 `encode_text(..., normalize=True)` 关键字参数。本脚本已通过手动归一化解决此问题：

```python
# 本脚本使用的方式（兼容所有版本）
F.normalize(model.encode_text(tokens).float(), dim=-1)
F.normalize(model.encode_image(normalize(image)).float(), dim=-1)
```

### Q: uap_delta.pth 文件可以复用吗？

可以。UAP 扰动张量保存为标准 PyTorch 文件，可通过以下方式加载复用：

```python
import torch
delta = torch.load("output_dir/uap_delta.pth")
adv_img = torch.clamp(clean_img + delta.to(device), 0.0, 1.0)
```

### Q: CLIP 模型可以替换吗？

可以，通过 `--clip_model` 和 `--clip_pretrained` 指定 OpenCLIP 支持的模型，例如：
- `ViT-B-32` + `laion2b_s34b_b79k`（默认，速度较快）
- `ViT-B-16` + `laion2b_s34b_b88k`（更精细的特征）
- `ViT-L-14` + `laion2b_s32b_b82k`（更强的特征，需要更多显存）

### Q: 如何调整攻击强度？

| 调整方向 | 参数建议 |
|----------|----------|
| **更强攻击**（更明显的扰动） | 增大 `--epsilon`（如 32、64） |
| **更精细攻击**（更多迭代） | 增大 `--steps`（如 500、1000） |
| **更快收敛** | 增大 `--step_size`（如 2、4） |
| **UAP 更强泛化** | 增大 `--uap_epochs`（如 10、20） |
| **更隐蔽攻击** | 减小 `--epsilon`（如 8、4） |
