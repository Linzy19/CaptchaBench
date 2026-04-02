# 🛡️ 对抗攻击工具使用文档

> 本文档包含两套对抗攻击方案的完整使用说明：
> - **AttackImage.py** — 图像编码器级别攻击（图片 → 图片）
> - **AttackVLM.py** — VLM 文字级别攻击（图片 → 文字）

---

## 目录

- [环境要求](#环境要求)
- [方案一：AttackImage.py — 图像特征攻击](#方案一attackimagepy--图像特征攻击)
  - [攻击原理](#攻击原理)
  - [参数说明](#参数说明)
  - [使用示例](#使用示例)
  - [配对模式详解](#配对模式详解)
  - [目标图片加载逻辑](#目标图片加载逻辑)
- [方案二：AttackVLM.py — VLM 文字攻击](#方案二attackvlmpy--vlm-文字攻击)
  - [攻击原理](#攻击原理-1)
  - [参数说明](#参数说明-1)
  - [使用示例](#使用示例-1)
  - [目标文字来源详解](#目标文字来源详解)
  - [JSON 注释文件格式](#json-注释文件格式)
- [两套方案对比](#两套方案对比)
- [输出目录结构](#输出目录结构)
- [常见问题](#常见问题)

---

## 环境要求

- Python 3.8+
- PyTorch（支持 CUDA）
- 项目依赖（surrogates / attacks / utils 模块，来自 Attack-Bard）
- GPU 显存要求：
  - AttackImage.py：≥ 8GB（仅加载编码器）
  - AttackVLM.py：≥ 16GB（加载完整 VLM 模型），不用 MiniGPT4 时 ≥ 12GB

---

## 环境准备

> **重要**：Attack-Bard 依赖重量级 VLM 模型（BLIP2、InstructBLIP、MiniGPT4），
> 安装步骤比其他框架更复杂。

### 第 1 步：创建 conda 环境

```bash
conda create -n attack_bard python=3.10 -y
conda activate attack_bard
```

### 第 2 步：安装 PyTorch（CUDA 11.8）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y
```

### 第 3 步：安装 Python 依赖

```bash
cd Attack-Bard
pip install -r requirements.txt
```

### 第 4 步：配置 MiniGPT4（使用 `--model_name minigpt4` 时需要）

```bash
# 下载 Vicuna 权重后，修改配置文件中的路径
# eval_configs/minigpt4_eval.yaml → llama_model: /path/to/vicuna-weights
```

> 如不需要 MiniGPT4，在运行参数中不指定 `--model_name minigpt4` 即可跳过。

### 第 5 步：验证安装

```bash
# 验证核心依赖
python -c "import torch; from transformers import Blip2Processor; import timm; print('OK, CUDA:', torch.cuda.is_available())"
```

### 依赖说明

| 包 | 用途 | 适用脚本 |
|----|------|---------|
| `transformers>=4.26.0` | BLIP2、InstructBLIP、CLIP 模型加载 | 两者 |
| `accelerate>=0.16.0` | 大模型加载加速 | 两者 |
| `timm>=0.6.12` | minigpt4 内部依赖（`eva_vit.py`、`dist_utils.py`）；注意 `ViT.py` 实际使用 `transformers.ViTModel` | AttackVLM.py（minigpt4 模式） |
| `Pillow` | 图像 I/O | 两者 |
| `numpy` | 数值计算 | 两者 |
| `scipy` | SSA 攻击高斯核 | 两者 |
| `einops` | SSA 频谱模拟攻击 | 两者 |
| `tqdm` | 进度条 | 两者 |
| `ftfy` | transformers 文本预处理 | 两者 |
| `gradio` | MiniGPT4 Web UI（可选）| AttackVLM.py |
| `omegaconf` | MiniGPT4 配置管理 | AttackVLM.py |

---

## 方案一：AttackImage.py — 图像特征攻击

### 攻击原理

```
┌──────────────┐                    ┌───────────────────┐
│  原图 Source  │ ──── 输入 ────→   │  SSA_CommonWeakness │
└──────────────┘                    │     攻击器          │
                                    │                     │
┌──────────────┐     特征提取       │  ViT + BLIP + CLIP  │
│  目标图 Target│ ──── → ────→      │  (EnsembleFeatureLoss)│
└──────────────┘   Ground Truth     └─────────┬───────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  对抗图 Adversarial │
                                    │  (特征接近目标图)    │
                                    └──────────────────┘
```

**核心思路**：使用 ViT、BLIP、CLIP 三个视觉编码器提取目标图片的特征作为 Ground Truth，通过 SSA（频谱模拟攻击）+ CommonWeakness（公共弱点）方法，对原图施加对抗扰动，使其在特征空间上逼近目标图。

### 参数说明

| 参数 | 缩写 | 类型 | 默认值 | 必填 | 说明 |
|------|------|------|--------|------|------|
| `--source_dir` | — | str | — | ✅ | 原图目录（要施加扰动的图片） |
| `--target_dir` | — | str | None | ⚠️ | 目标图目录（与 `--target_json` 二选一） |
| `--target_json` | — | str | None | ⚠️ | 目标图路径 JSON 文件（与 `--target_dir` 二选一） |
| `--output_dir` | — | str | `./output_attack_image` | ❌ | 输出目录 |
| `--epsilon` | `-e` | float | 16 | ❌ | 最大扰动幅度（0-255 尺度） |
| `--step_size` | — | float | 1 | ❌ | PGD 步长（0-255 尺度） |
| `--steps` | `-s` | int | 500 | ❌ | 攻击总迭代步数 |
| `--input_res` | `-r` | int | 224 | ❌ | 输入分辨率（宽=高） |
| `--match_mode` | `-m` | str | `auto` | ❌ | 配对模式：`auto` / `single` / `random` |
| `--seed` | — | int | 2023 | ❌ | 随机种子 |

> ⚠️ `--target_dir` 和 `--target_json` 必须至少指定一个。

### 使用示例

#### 1. 基础用法 — 随机配对攻击

```bash
python AttackImage.py \
    --source_dir ./dataset/NIPS17 \
    --target_dir ./target_images \
    --output_dir ./output_attack_img
```

#### 2. 自定义攻击参数

```bash
python AttackImage.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --epsilon 16 \
    --steps 500 \
    --step_size 1 \
    --match_mode random
```

#### 3. 单目标模式 — 所有原图攻击向同一张目标图

```bash
python AttackImage.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --match_mode single
```

#### 4. 使用 target_paths.json 指定目标图

```bash
python AttackImage.py \
    --source_dir ./source \
    --target_json /path/to/target_paths.json \
    --output_dir ./output
```

#### 5. 高强度攻击（大扰动 + 多步数）

```bash
python AttackImage.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    -e 32 -s 1000 --step_size 2
```

### 配对模式详解

| 模式 | 说明 |
|------|------|
| `auto`（默认） | 自动检测：目标图只有 1 张 → `single`；多张 → `random` |
| `single` | 所有原图都攻击向**同一张**目标图（取目标列表的第 1 张） |
| `random` | 每张原图**随机选择**一张目标图进行配对 |

### 目标图片加载逻辑

加载优先级（按顺序尝试，成功即停止）：

1. 若指定了 `--target_json`，优先从该 JSON 文件读取路径列表
2. 若目标目录下存在 `target_paths.json`，从中读取路径列表
3. 以上都不存在时，自动扫描 `--target_dir` 目录下的所有图片文件

---

## 方案二：AttackVLM.py — VLM 文字攻击

### 攻击原理

```
┌──────────────┐                    ┌────────────────────────┐
│  原图 Source  │ ──── 输入 ────→   │   SSA_CommonWeakness    │
└──────────────┘                    │       攻击器            │
                                    │                        │
┌──────────────────┐   文字嵌入     │  BLIP2 + InstructBLIP   │
│ 目标文字          │ ── → ──→     │    + MiniGPT4           │
│ (prompt + label) │  VLM Loss     └───────────┬────────────┘
└──────────────────┘                           │
                                               ▼
                                    ┌──────────────────┐
                                    │  对抗图 Adversarial │
                                    │  (VLM 生成目标文字) │
                                    └──────────────────┘
```

**核心思路**：使用 BLIP2、InstructBLIP、MiniGPT4 三个 VLM 作为代理模型，通过最大化目标文字的似然概率（取负 VLM loss），迫使 VLM 在看到对抗图时生成指定的目标文字。

### 参数说明

| 参数 | 缩写 | 类型 | 默认值 | 必填 | 说明 |
|------|------|------|--------|------|------|
| `--source_dir` | — | str | — | ✅ | 原图目录（要施加扰动的图片） |
| `--output_dir` | — | str | `./output_attack_vlm` | ❌ | 输出目录 |
| `--target_text` | — | str | None | ⚠️ | 统一的目标文字（三选一） |
| `--target_prompt` | — | str | None | ⚠️ | 目标 prompt（需配合 `--target_label`） |
| `--target_label` | — | str | None | ⚠️ | 目标 label（需配合 `--target_prompt`） |
| `--use_json_text` | — | flag | False | ⚠️ | 从每张图的同名 JSON 文件读取目标文字 |
| `--epsilon` | `-e` | float | 16 | ❌ | 最大扰动幅度（0-255 尺度） |
| `--step_size` | — | float | 1 | ❌ | PGD 步长（0-255 尺度） |
| `--steps` | `-s` | int | 500 | ❌ | 攻击总迭代步数 |
| `--input_res` | `-r` | int | 224 | ❌ | 输入分辨率（宽=高） |
| `--no_gpt4` | — | flag | False | ❌ | 禁用 MiniGPT4（仅用 BLIP2 + InstructBLIP） |
| `--log_interval` | — | int | 120 | ❌ | 每隔 N 步打印一次 loss |
| `--seed` | — | int | 2023 | ❌ | 随机种子 |
| `--max_images` | — | int | None | ❌ | 最多处理图片数量（默认全部） |

> ⚠️ 目标文字来源必须三选一：`--target_text`、`--use_json_text`、或 `--target_prompt` + `--target_label` 组合。

### 使用示例

#### 1. 直接指定目标文字

```bash
python AttackVLM.py \
    --source_dir ./dataset/NIPS17 \
    --target_text "This is a cat sitting on a sofa" \
    --output_dir ./output_attack_vlm
```

#### 2. 从 JSON 注释文件读取目标文字（prompt + label）

```bash
python AttackVLM.py \
    --source_dir ./source_with_json \
    --output_dir ./output \
    --use_json_text
```

#### 3. 手动指定 prompt + label 组合

```bash
python AttackVLM.py \
    --source_dir ./source \
    --target_prompt "Rocky mountain terrain with ancient Chinese character" \
    --target_label "贞" \
    --output_dir ./output
```

#### 4. 不使用 MiniGPT4（节省显存）

```bash
python AttackVLM.py \
    --source_dir ./source \
    --target_text "A beautiful sunset over the ocean" \
    --output_dir ./output \
    --no_gpt4
```

#### 5. 限制处理数量 + 自定义参数

```bash
python AttackVLM.py \
    --source_dir ./dataset/NIPS17 \
    --target_text "Harmful content description" \
    --output_dir ./output \
    --epsilon 32 \
    --steps 1000 \
    --max_images 200 \
    --log_interval 60
```

#### 6. 大规模批量攻击

```bash
python AttackVLM.py \
    --source_dir /data/images/all_sources \
    --output_dir /data/output/vlm_attack \
    --use_json_text \
    -e 16 -s 500 \
    --seed 42
```

### 目标文字来源详解

目标文字的确定遵循以下优先级（从高到低）：

| 优先级 | 来源 | 触发条件 | 说明 |
|--------|------|----------|------|
| 1 | JSON 注释文件 | `--use_json_text` | 从每张图片同名的 `.json` 文件中提取 `prompt` + `annotations[].label` |
| 2 | Prompt + Label 组合 | `--target_prompt` + `--target_label` | 拼接为 `"{prompt} Label: {label}"` |
| 3 | 直接指定文字 | `--target_text` | 所有图片使用相同的目标文字 |

> 💡 如果 `--use_json_text` 模式下某张图片找不到对应 JSON 文件，会自动 fallback 到优先级 2 或 3 的方式。

### JSON 注释文件格式

当使用 `--use_json_text` 时，程序会查找与图片同名的 `.json` 文件（如 `image_001.png` → `image_001.json`），格式如下：

```json
{
    "prompt": "Rocky mountain terrain with ancient inscriptions",
    "annotations": [
        {"label": "贞"},
        {"label": "卜"}
    ]
}
```

生成的目标文字为：`"Rocky mountain terrain with ancient inscriptions Label: 贞, 卜"`

---

## 两套方案对比

| 维度 | AttackImage.py | AttackVLM.py |
|------|---------------|-------------|
| **攻击层面** | 图像特征空间 | VLM 文字生成空间 |
| **攻击目标** | 让原图特征逼近目标图特征 | 让 VLM 对原图生成指定目标文字 |
| **输入** | 原图 + 目标图 | 原图 + 目标文字 (prompt + label) |
| **代理模型** | ViT + BLIP + CLIP（特征提取器） | BLIP2 + InstructBLIP + MiniGPT4（VLM） |
| **损失函数** | `EnsembleFeatureLoss` (MSE) | `VLMAttackCriterion` (负 VLM loss) |
| **攻击方法** | SSA_CommonWeakness | SSA_CommonWeakness |
| **GPU 显存** | ≥ 8GB | ≥ 16GB（不用 MiniGPT4 时 ≥ 12GB） |
| **单图耗时** | 较快（~30s/张） | 较慢（~60s/张） |
| **适用场景** | 图像检索欺骗、特征空间投毒 | VLM 输出篡改、文字内容注入 |

### 选择建议

- 如果你的目标是**让图片在视觉模型中被误识别为另一张图片**  → 使用 **AttackImage.py**
- 如果你的目标是**让 VLM 对图片输出特定的文字描述**  → 使用 **AttackVLM.py**
- 如果你需要**同时攻击图像特征和文字输出**  → 先用 AttackImage.py 生成图片级对抗样本，再用 AttackVLM.py 进一步叠加文字攻击

---

## 输出目录结构

两个脚本运行后，输出目录结构一致：

```
output_dir/
├── image_001_adv.png        # 对抗图片
├── image_001_adv.json       # 对应的注释文件（如果原图有 JSON 则自动拷贝）
├── image_002_adv.png
├── image_002_adv.json
├── ...
└── match_info.json           # 攻击配对信息 + 运行参数 + 耗时统计
```

### match_info.json 示例

**AttackImage.py 的 match_info.json：**

```json
{
  "attack_type": "image_encoder_feature_attack",
  "method": "SSA_CommonWeakness",
  "models": ["ViT", "BLIP", "CLIP"],
  "parameters": {
    "epsilon": 16,
    "step_size": 1,
    "total_step": 500,
    "input_res": 224
  },
  "pairs": [
    {
      "source": "image_001.png",
      "target": "target_005.png",
      "output": "image_001_adv.png"
    }
  ],
  "timing": {
    "model_load_time_sec": 12.5,
    "attack_time_sec": 300.0,
    "total_time_sec": 312.5,
    "avg_time_per_image_sec": 30.0
  }
}
```

**AttackVLM.py 的 match_info.json：**

```json
{
  "attack_type": "vlm_text_attack",
  "method": "SSA_CommonWeakness",
  "parameters": {
    "epsilon": 16,
    "step_size": 1,
    "total_step": 500,
    "use_gpt4": true
  },
  "text_groups": [
    {
      "target_text": "Rocky mountain terrain ... Label: 贞",
      "num_images": 10,
      "attack_time_sec": 600.0
    }
  ],
  "pairs": [
    {
      "source": "image_001.png",
      "target_text": "Rocky mountain terrain ... Label: 贞",
      "output": "image_001_adv.png"
    }
  ],
  "timing": {
    "total_time_sec": 600.0,
    "total_images": 10,
    "avg_time_per_image_sec": 60.0
  }
}
```

---

## 常见问题

### Q: 两个脚本可以搭配使用吗？

可以。你可以先用 `AttackImage.py` 生成图像特征级别的对抗样本，然后把输出作为 `AttackVLM.py` 的输入，进行二次文字攻击：

```bash
# 第一步：图像特征攻击
python AttackImage.py \
    --source_dir ./original_images \
    --target_dir ./target_images \
    --output_dir ./step1_image_attack

# 第二步：VLM 文字攻击（在已攻击图片上叠加）
python AttackVLM.py \
    --source_dir ./step1_image_attack \
    --target_text "Target description" \
    --output_dir ./step2_vlm_attack
```

### Q: 显存不够怎么办？

- 使用 `--no_gpt4` 参数禁用 MiniGPT4，可节省约 4GB 显存
- 降低 `--input_res`（如改为 128）
- 使用 `--max_images` 限制单次处理数量

### Q: 如何调整攻击强度？

| 调整方向 | 参数建议 |
|----------|----------|
| **更强攻击**（更明显的扰动） | 增大 `--epsilon`（如 32、64） |
| **更精细攻击**（更多迭代） | 增大 `--steps`（如 1000、2000） |
| **更快收敛** | 增大 `--step_size`（如 2、4） |
| **更隐蔽攻击**（更小扰动） | 减小 `--epsilon`（如 8、4） |

### Q: 随机配对的结果可复现吗？

可以。通过 `--seed` 参数固定随机种子（默认 2023），相同种子 + 相同输入 = 相同的配对结果和攻击输出。

### Q: match_info.json 有什么用？

记录了每张图片的完整配对关系、攻击参数和耗时统计，方便后续分析和复现实验。
