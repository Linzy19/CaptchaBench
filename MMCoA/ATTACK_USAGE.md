# 对抗攻击工具使用文档 — MMCoA

> **AttackMMCoA.py** — 联合图像级 + 文本级对抗攻击

---

## 目录

- [攻击原理](#攻击原理)
- [三种攻击域对比](#三种攻击域对比)
- [环境要求](#环境要求)
- [参数说明](#参数说明)
- [使用示例](#使用示例)
- [JSON 注释文件格式](#json-注释文件格式)
- [输出目录结构](#输出目录结构)
- [常见问题](#常见问题)

---

## 攻击原理

### 图像攻击（image 模式）

```
┌──────────────┐                    ┌──────────────────────────┐
│  原图 Source  │ ──── 输入 ────→   │   ImageAttack_MI          │
└──────────────┘                    │   (动量迭代生成器)          │
                                    │                          │
┌──────────────┐    CLIP 特征        │   CLIP ViT-B/32           │
│  目标图 Target│ ── → ──→           │   encode_image()         │
└──────────────┘   Ground Truth     └──────────┬───────────────┘
                                               │  -cosine_similarity
                                               ▼
                                    ┌──────────────────────────┐
                                    │  对抗图 Adversarial        │
                                    │  (CLIP 特征接近目标图)      │
                                    └──────────────────────────┘
```

**图像攻击核心思路**：使用 CLIP ViT-B/32 作为代理模型，通过 ImageAttack_MI
（动量迭代 PGD 生成器接口）对原图施加对抗扰动，使其 CLIP 视觉特征余弦相似度
向目标图特征靠拢。

**生成器接口**：`ImageAttack_MI.attack()` 是 Python 生成器，使用方式：
```python
gen = attacker.attack(src_img, num_iters)
for _ in range(num_iters):
    adv = next(gen)          # 获取带梯度的当前对抗图像
    loss = ...               # 计算损失
    loss.backward()          # 反向传播
adv_final = next(gen)        # 最终对抗图像（已去梯度）
```

### 文本攻击（text 模式）

```
┌─────────────────────────┐
│  目标 JSON 的文本          │  prompt + label
│  (prompt + label)        │ ─────────────────→  BertAttack
└─────────────────────────┘                     词重要性排序
                                                     │
                                            BERT MLM 词替换
                                                     │
                                                     ▼
                                    ┌──────────────────────────┐
                                    │  对抗文本 Adversarial Text  │
                                    │  ({name}_adv.json)         │
                                    └──────────────────────────┘
```

**文本攻击核心思路**：对目标 JSON 的 `prompt + label` 文本，通过 BertAttack
（基于 BERT MLM 的词重要性排序 + 候选词替换）生成对抗文本，使 CLIP 文本嵌入
偏离原始值，存储到输出 JSON 的 `prompt_adv` 字段。

---

## 三种攻击域对比

| 维度 | image | text | both |
|------|-------|------|------|
| **攻击目标** | 原图视觉特征 | 目标文本嵌入 | 两者同时 |
| **输入** | 原图 + 目标图 | 原图 + 目标 JSON | 两者都需要 |
| **代理模型** | CLIP ViT-B/32（图像编码器） | CLIP + BERT MLM | 两者都加载 |
| **输出** | `{name}_adv.png` | `{name}_adv.json` | 图像 + JSON |
| **GPU 显存** | ≥ 4GB | ≥ 6GB（加载 BERT） | ≥ 6GB |

---

## 环境要求

- Python 3.8+
- PyTorch（支持 CUDA）
- OpenAI CLIP：`pip install git+https://github.com/openai/CLIP.git`
- torchvision
- Pillow、tqdm、numpy
- **text / both 模式额外需要**：transformers（`pip install transformers`）
- MMCoA 框架（`attack/imageAttack.py`、`attack/bert_attack.py`）

---

## 环境准备

### 第 1 步：创建 conda 环境

```bash
conda create -n mmcoa python=3.10 -y
conda activate mmcoa
```

### 第 2 步：安装 PyTorch（CUDA 11.8）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y
```

### 第 3 步：安装 Python 依赖

```bash
cd MMCoA
pip install -r requirements.txt
```

> **说明**：`requirements.txt` 中包含 `git+https://github.com/openai/CLIP.git`，
> pip 会直接从 GitHub 安装 OpenAI CLIP，需要网络连接。

### 第 4 步：验证安装

```bash
# 验证 image 模式依赖
python -c "import clip; import scipy; print('image 模式 OK')"

# 验证 text / both 模式依赖
python -c "from transformers import BertTokenizer; print('text 模式 OK')"
```

### 依赖说明

| 包 | 用途 | 模式 |
|----|------|------|
| `openai/CLIP`（GitHub） | CLIP 图像编码器 | image + both |
| `scipy` | MI 高斯核（`attack/imageAttack.py`）| image + both |
| `matplotlib` | `attack/imageAttack.py` 内部导入 | image + both |
| `Pillow` | 图像 I/O | 所有 |
| `numpy` | 数值计算 | 所有 |
| `tqdm` | 进度条 | 所有 |
| `transformers>=4.26.0` | BertTokenizer + BertForMaskedLM | text + both |
| `ftfy` | transformers 文本预处理 | text + both |

---

## 参数说明

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `--source_dir` | str | — | ✅ | 原图目录（待扰动的图像） |
| `--target_dir` | str | — | ✅ | 目标目录（图像对齐目标 / 文本 JSON 来源） |
| `--output_dir` | str | `./output_mmcoa` | ❌ | 输出目录 |
| `--attack_domain` | str | `image` | ❌ | 攻击域：`image` / `text` / `both` |
| `--clip_model` | str | `ViT-B/32` | ❌ | CLIP 模型（OpenAI CLIP 格式） |
| `--epsilon` | float | 16 | ❌ | L-inf 扰动上限（0-255 尺度，图像攻击） |
| `--step_size` | float | 1 | ❌ | PGD 步长（0-255 尺度，图像攻击） |
| `--num_iters` | int | 300 | ❌ | 攻击迭代步数（图像攻击） |
| `--bert_model` | str | `bert-base-uncased` | ❌ | BERT MLM 模型（文本攻击） |
| `--match_mode` | str | `auto` | ❌ | 配对模式：`auto` / `single` / `random` |
| `--input_res` | int | 224 | ❌ | 输入分辨率（图像攻击） |
| `--seed` | int | 42 | ❌ | 随机种子 |

---

## 使用示例

### 1. 图像攻击（默认）

```bash
python AttackMMCoA.py \
    --source_dir ../Data_source/sample_source \
    --target_dir ../Data_source/sample_target \
    --output_dir ./output_mmcoa \
    --attack_domain image
```

### 2. 文本攻击

```bash
python AttackMMCoA.py \
    --source_dir ../Data_source/sample_source \
    --target_dir ../Data_source/sample_target \
    --output_dir ./output_mmcoa \
    --attack_domain text
```

### 3. 联合攻击（图像 + 文本同时）

```bash
python AttackMMCoA.py \
    --source_dir ../Data_source/sample_source \
    --target_dir ../Data_source/sample_target \
    --output_dir ./output_mmcoa \
    --attack_domain both
```

### 4. 自定义图像攻击参数

```bash
python AttackMMCoA.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --attack_domain image \
    --epsilon 16 \
    --step_size 1 \
    --num_iters 300 \
    --match_mode random \
    --seed 42
```

### 5. 使用单目标图

```bash
python AttackMMCoA.py \
    --source_dir ./source \
    --target_dir ./target \
    --output_dir ./output \
    --attack_domain image \
    --match_mode single
```

---

## JSON 注释文件格式

文本攻击模式下，程序扫描 `--target_dir` 中的 `.json` 文件并提取目标文本，格式如下：

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

文本攻击的输出 JSON（`{name}_adv.json`）在原始结构基础上追加 `prompt_adv` 字段：

```json
{
    "prompt": "Rocky mountain terrain with ancient Chinese character inscriptions",
    "annotations": [{"label": "水"}],
    "prompt_adv": "Rocky terrain mountain with ancient Chinese character inscriptions"
}
```

---

## 输出目录结构

```
output_dir/
├── image_001_adv.png          # 对抗图片（image / both 模式）
├── image_001_adv.json         # 对抗文本 JSON（text / both 模式）
├── image_002_adv.png
├── ...
├── match_info.json            # 图像攻击配对信息
└── match_info_text.json       # 文本攻击配对信息（text / both 模式）
```

### match_info.json 示例（图像攻击）

```json
{
  "attack_type": "mmcoa_image_attack",
  "method": "ImageAttack_MI_CLIP",
  "parameters": {
    "epsilon": 16,
    "step_size": 1,
    "num_iters": 300,
    "clip_model": "ViT-B/32"
  },
  "pairs": [
    {
      "source": "image_001.png",
      "target": "target_001.png",
      "output": "image_001_adv.png"
    }
  ],
  "timing": {
    "attack_time_sec": 90.0,
    "avg_time_per_image_sec": 9.0
  }
}
```

---

## 常见问题

### Q: 图像攻击和文本攻击可以搭配使用吗？

可以，直接使用 `--attack_domain both`，两种攻击会依次执行，输出对应的图像和 JSON 文件。

### Q: 文本攻击提示找不到 JSON 文件怎么办？

确保 `--target_dir` 中包含 `.json` 文件（与目标图同名或独立存放均可）。
文本攻击仅读取目标目录的 JSON，不需要目标目录中有图像文件。

### Q: CLIP 模型可以替换吗？

可以，通过 `--clip_model` 指定 OpenAI CLIP 支持的模型名称，例如：
- `ViT-B/32`（默认，速度较快）
- `ViT-B/16`（更精细的特征）
- `ViT-L/14`（更强的特征，需要更多显存）
