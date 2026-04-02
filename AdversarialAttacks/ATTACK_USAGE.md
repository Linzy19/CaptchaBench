# 对抗攻击工具使用文档 — AdversarialAttacks

> **AttackMI.py** — MI_CommonWeakness 迁移攻击（图像级特征对齐）

---

## 目录

- [攻击原理](#攻击原理)
- [环境要求](#环境要求)
- [参数说明](#参数说明)
- [使用示例](#使用示例)
- [配对模式详解](#配对模式详解)
- [目标图片加载逻辑](#目标图片加载逻辑)
- [输出目录结构](#输出目录结构)
- [常见问题](#常见问题)

---

## 攻击原理

```
┌──────────────┐                    ┌─────────────────────────────┐
│  原图 Source  │ ──── 输入 ────→   │     MI_CommonWeakness        │
└──────────────┘                    │          攻击器               │
                                    │                              │
┌──────────────┐    特征提取         │  ResNet-18 / ResNet-50       │
│  目标图 Target│ ── → ──→          │  / ViT-B-16                  │
└──────────────┘   Ground Truth     │  (统一投影到 512 维)           │
                                    └──────────────┬───────────────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │  对抗图 Adversarial        │
                                    │  (特征空间接近目标图)        │
                                    └──────────────────────────┘
```

**核心思路**：使用 ResNet-18、ResNet-50、ViT-B/16 三个视觉编码器（分类头替换为线性投影，
统一到 512 维特征空间）提取目标图的特征之和作为 Ground Truth。通过 MI_CommonWeakness
（动量迭代 + 公共弱点）方法对原图施加对抗扰动，使其在特征空间中逼近目标图。

**损失函数**：`-MSE(adv_features_sum, target_features_sum)`

**攻击标志**：`targeted_attack=True`（攻击器最大化 criterion，即最小化 MSE，
即将对抗特征拉向目标特征）。

---

## 环境要求

- Python 3.8+
- PyTorch（支持 CUDA）
- torchvision
- Pillow、tqdm、numpy
- AdversarialAttacks 框架（`attacks/AdversarialInput/CommonWeakness.py`）
- GPU 显存要求：≥ 4GB（仅加载 ResNet + ViT 特征提取器）

---

## 环境准备

### 第 1 步：创建 conda 环境

```bash
conda create -n adv_attack python=3.10 -y
conda activate adv_attack
```

### 第 2 步：安装 PyTorch（CUDA 11.8）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y
```

> 如需其他 CUDA 版本，请参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。

### 第 3 步：安装 Python 依赖

```bash
cd AdversarialAttacks
pip install -r requirements.txt
```

### 第 4 步：验证安装

```bash
python -c "import torch; import timm; import scipy; print('OK, CUDA:', torch.cuda.is_available())"
```

### 依赖说明

| 包 | 用途 |
|----|------|
| `timm>=0.6.12` | ViT-B/16 模型加载 |
| `numpy` | 数值计算 |
| `Pillow` | 图像 I/O |
| `scipy` | MI 高斯核（`attacks/AdversarialInput/CommonWeakness.py`） |
| `tqdm` | 进度条显示 |

---

## 参数说明

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `--source_dir` | str | — | ✅ | 原图目录（待扰动的图像） |
| `--target_dir` | str | — | ✅ | 目标图目录（特征对齐目标） |
| `--output_dir` | str | `./output_mi_attack` | ❌ | 输出目录 |
| `--epsilon` | float | 16 | ❌ | L-inf 扰动上限（0-255 尺度） |
| `--step_size` | float | 1 | ❌ | 外层 PGD 步长（0-255 尺度） |
| `--steps` | int | 300 | ❌ | 攻击总迭代步数 |
| `--inner_step_size` | float | 250 | ❌ | MI 内层动量步长 |
| `--reverse_step_size` | float | 1 | ❌ | 反向扰动步长（0-255 尺度） |
| `--match_mode` | str | `auto` | ❌ | 配对模式：`auto` / `single` / `random` |
| `--input_res` | int | 224 | ❌ | 输入分辨率（宽=高） |
| `--seed` | int | 2023 | ❌ | 随机种子 |

---

## 使用示例

### 1. 基础用法

```bash
python AttackMI.py \
    --source_dir ../Data_source/sample_source \
    --target_dir ../Data_source/sample_target \
    --output_dir ./output_mi_attack
```

### 2. 自定义攻击强度

```bash
python AttackMI.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --epsilon 16 \
    --step_size 1 \
    --steps 300 \
    --inner_step_size 250 \
    --reverse_step_size 1
```

### 3. 单目标模式 — 所有原图攻击向同一张目标图

```bash
python AttackMI.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --match_mode single
```

### 4. 随机配对 + 固定种子（可复现）

```bash
python AttackMI.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --match_mode random \
    --seed 2023
```

### 5. 高强度攻击（更大扰动 + 更多步数）

```bash
python AttackMI.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --epsilon 32 \
    --steps 500 \
    --step_size 2
```

---

## 配对模式详解

| 模式 | 说明 |
|------|------|
| `auto`（默认） | 自动检测：目标图只有 1 张 → `single`；多张 → `random` |
| `single` | 所有原图都攻击向**同一张**目标图（取目标列表的第 1 张） |
| `random` | 每张原图**随机选择**一张目标图进行配对 |

---

## 目标图片加载逻辑

加载优先级（按顺序尝试，成功即停止）：

1. 若 `--target_dir` 下存在 `target_paths.json`，从中读取路径列表
2. 以上不存在时，自动扫描 `--target_dir` 目录下的所有图片文件

---

## 输出目录结构

```
output_dir/
├── image_001_adv.png        # 对抗图片
├── image_001_adv.json       # 对应注释文件（原图有 JSON 时自动拷贝）
├── image_002_adv.png
├── ...
└── match_info.json          # 攻击配对信息 + 参数 + 耗时统计
```

### match_info.json 示例

```json
{
  "attack_type": "mi_transfer_feature_alignment",
  "method": "MI_CommonWeakness",
  "models": ["ResNet-18", "ResNet-50", "ViT-B-16"],
  "parameters": {
    "epsilon": 16,
    "step_size": 1,
    "total_step": 300,
    "inner_step_size": 250,
    "reverse_step_size": 1,
    "input_res": 224,
    "unified_feature_dim": 512
  },
  "pairs": [
    {
      "source": "image_001.png",
      "target": "target_005.png",
      "output": "image_001_adv.png"
    }
  ],
  "timing": {
    "model_load_time_sec": 8.5,
    "attack_time_sec": 240.0,
    "total_time_sec": 248.5,
    "avg_time_per_image_sec": 24.0
  }
}
```

---

## 常见问题

### Q: inner_step_size 和 reverse_step_size 的作用是什么？

MI_CommonWeakness 攻击分两步：
- **reverse_step_size**：第1步对汇总 logit 执行快速反向扰动，用于快速逃离当前局部最优
- **inner_step_size**：第2步对每个模型分别执行内层动量迭代，用于精细寻找公共弱点

通常保持默认值即可，如需更激进的攻击可适当增大 `inner_step_size`（如 500）。

### Q: 如何调整攻击强度？

| 调整方向 | 参数建议 |
|----------|----------|
| **更强攻击**（更明显的扰动） | 增大 `--epsilon`（如 32、64） |
| **更精细攻击**（更多迭代） | 增大 `--steps`（如 500、1000） |
| **更快收敛** | 增大 `--step_size`（如 2、4） |
| **更隐蔽攻击** | 减小 `--epsilon`（如 8、4） |

### Q: 随机配对的结果可复现吗？

可以。通过 `--seed` 参数固定随机种子（默认 2023），相同种子 + 相同输入 = 相同配对结果。
