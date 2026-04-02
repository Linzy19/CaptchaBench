"""
AttackMMCoA.py - MMCoA 联合图像级 + 文本级对抗攻击

攻击策略：
    - image 模式（默认）：CLIP + ImageAttack_MI 梯度扰动
        扰动原图使其 CLIP 视觉特征向目标图的 CLIP 视觉特征对齐。
        损失函数：-cosine_similarity(clip.encode_image(adv), clip.encode_image(target))
        ImageAttack_MI.attack() 是 Python 生成器：循环中每次调用 next() 获取带梯度的
        图像，执行 loss.backward()，最后再调用一次 next() 取得最终去梯度结果。
    - text 模式：CLIP 文本编码器 + 词重要性替换
        对目标 JSON 的 prompt + label 文本执行对抗扰动，替换关键词以最大化文本
        嵌入与原始值的 KL 散度。输出对抗文本，存储为 {name}_adv.json。
    - both 模式：同时执行图像攻击和文本攻击，输出对抗图像和对抗 JSON。

框架：MMCoA
参考：
    - MMCoA/attack/imageAttack.py  -> ImageAttack_MI 生成器接口
    - MMCoA/attack/bert_attack.py  -> BertAttack 词替换
    - MMCoA/MMCoA.py               -> CLIP 加载模式

图像攻击（默认）：
    python AttackMMCoA.py \\
        --source_dir /path/to/source --target_dir /path/to/target \\
        --output_dir ./output_mmcoa --attack_domain image

文本攻击：
    python AttackMMCoA.py \\
        --source_dir /path/to/source --target_dir /path/to/target \\
        --output_dir ./output_mmcoa --attack_domain text

联合攻击：
    python AttackMMCoA.py \\
        --source_dir /path/to/source --target_dir /path/to/target \\
        --output_dir ./output_mmcoa --attack_domain both

自定义参数：
    python AttackMMCoA.py \\
        --source_dir ./source --target_dir ./target \\
        --output_dir ./output --attack_domain image \\
        --epsilon 16 --step_size 1 --num_iters 300 --match_mode random
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import glob
import time
import copy
import shutil
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# ---------------------------------------------------------------------------
# 路径配置：将 MMCoA 根目录加入 sys.path，使 attack 包可导入
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from attack.imageAttack import ImageAttack_MI, NormType

DEFAULT_RANDOM_SEED = 42

# CLIP 图像归一化常数
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# 通用辅助函数
# ---------------------------------------------------------------------------

def seed_everything(seed=DEFAULT_RANDOM_SEED):
    """固定所有随机种子，确保实验可复现。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_image_files(directory):
    """递归扫描目录，返回所有支持格式的图像路径（已排序）。"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp',
                  '*.JPEG', '*.JPG', '*.PNG']
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def build_target_text_from_json(json_path):
    """从 JSON 注释文件提取 'prompt Label: label1, label2' 字符串。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    parts = []
    prompt = data.get("prompt", "")
    if prompt:
        parts.append(prompt)
    labels = [ann.get("label", "") for ann in data.get("annotations", [])
              if ann.get("label")]
    if labels:
        parts.append("Label: " + ", ".join(labels))
    return " ".join(parts) if parts else None


def load_clip(clip_model_name="ViT-B/32", device="cuda"):
    """加载 OpenAI CLIP 模型，返回 (model, preprocess)。"""
    try:
        import clip
        model, preprocess = clip.load(clip_model_name, device=device)
        model.eval()
        return model, preprocess
    except ImportError:
        raise ImportError(
            "需要安装 OpenAI CLIP：pip install git+https://github.com/openai/CLIP.git"
        )


def clip_normalize(x):
    """将 [0,1] 图像张量用 CLIP 均值/标准差归一化。"""
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(CLIP_STD,  device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# 数据加载与配对
# ---------------------------------------------------------------------------

def match_source_target_pairs(source_dir, target_dir, match_mode="auto",
                               input_res=224):
    """
    构建 (src_path, tgt_path, src_tensor[0,1], tgt_tensor[0,1]) 配对列表。

    配对模式：
        auto   —— 目标图只有1张时用 single，否则用 random
        single —— 所有原图攻击向同一张目标图（取第1张）
        random —— 每张原图随机选择一张目标图
    """
    source_files = get_image_files(source_dir)
    target_files = get_image_files(target_dir)

    if not source_files:
        raise ValueError(f"原图目录中未找到图像：{source_dir}")
    if not target_files:
        raise ValueError(f"目标目录中未找到图像：{target_dir}")

    if match_mode == "auto":
        match_mode = "single" if len(target_files) == 1 else "random"
        print(f"[自动] {len(target_files)} 张目标 -> '{match_mode}' 模式")

    resizer   = transforms.Resize((input_res, input_res))
    to_tensor = transforms.ToTensor()

    pairs = []

    if match_mode == "single":
        tgt_path = target_files[0]
        tgt_img  = resizer(to_tensor(Image.open(tgt_path).convert("RGB")))
        for src_path in tqdm(source_files, desc="加载原图"):
            try:
                src_img = resizer(to_tensor(Image.open(src_path).convert("RGB")))
                pairs.append((src_path, tgt_path, src_img, tgt_img.clone()))
            except Exception as e:
                print(f"  ✗ {os.path.basename(src_path)}：{e}")
    elif match_mode == "random":
        for src_path in tqdm(source_files, desc="加载图像对"):
            tgt_path = random.choice(target_files)
            try:
                src_img = resizer(to_tensor(Image.open(src_path).convert("RGB")))
                tgt_img = resizer(to_tensor(Image.open(tgt_path).convert("RGB")))
                pairs.append((src_path, tgt_path, src_img, tgt_img))
            except Exception as e:
                print(f"  ✗ {os.path.basename(src_path)}：{e}")

    if not pairs:
        raise ValueError("未找到有效的原图-目标图对！")
    print(f"[完成] 共配对 {len(pairs)} 对")
    return pairs


def load_target_texts(target_dir, source_files, match_mode="auto"):
    """
    文本攻击模式：为每张原图配对目标 JSON 中的文本。
    返回 (src_path, target_text) 列表。
    """
    target_json_files = sorted(glob.glob(os.path.join(target_dir, "*.json")))
    if not target_json_files:
        raise ValueError(f"目标目录中未找到 JSON 文件：{target_dir}")

    if match_mode in ("auto", "random"):
        pairs = []
        for src_path in source_files:
            tgt_json = random.choice(target_json_files)
            text = build_target_text_from_json(tgt_json)
            if text:
                pairs.append((src_path, text))
            else:
                print(f"  ✗ {os.path.basename(tgt_json)} 中无有效文本")
    else:  # single 模式
        text = build_target_text_from_json(target_json_files[0])
        pairs = [(src, text) for src in source_files if text]

    return pairs


# ---------------------------------------------------------------------------
# 图像攻击
# ---------------------------------------------------------------------------

def run_image_attack(pairs, output_dir, clip_model_name="ViT-B/32",
                     epsilon=16, step_size=1, num_iters=300):
    """
    图像级 CLIP 特征对齐攻击。

    使用 ImageAttack_MI 生成器接口：
        1. gen = attacker.attack(src_img, num_iters)
        2. 循环 num_iters 次：adv = next(gen) → 计算 CLIP 损失 → loss.backward()
        3. adv_result = next(gen)  —— 最终去梯度结果
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n加载 CLIP 模型 ...")
    clip_model, _ = load_clip(clip_model_name, device)
    print(f"  ✓ CLIP {clip_model_name} 加载完成")

    eps   = epsilon  / 255.0
    s_sz  = step_size / 255.0

    attacker = ImageAttack_MI(
        epsilon=eps,
        norm_type=NormType.Linf,
        random_init=True,
        cls=True,
        step_size=s_sz,
        bounding=(0.0, 1.0),
    )

    match_info = {
        "attack_type": "mmcoa_image_attack",
        "method": "ImageAttack_MI_CLIP",
        "parameters": {
            "epsilon": epsilon,
            "step_size": step_size,
            "num_iters": num_iters,
            "clip_model": clip_model_name,
        },
        "pairs": [],
    }

    start = time.time()
    per_image_times = []
    n_img_success = 0
    n_img_failed = 0

    for i, (src_path, tgt_path, src_tensor, tgt_tensor) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {os.path.basename(src_path)}"
              f" -> {os.path.basename(tgt_path)}")
        img_start = time.time()

        src_img = src_tensor.unsqueeze(0).to(device)
        tgt_img = tgt_tensor.unsqueeze(0).to(device)

        # 预计算目标 CLIP 特征（固定，不参与梯度计算）
        with torch.no_grad():
            tgt_feat = clip_model.encode_image(clip_normalize(tgt_img))
            tgt_feat = F.normalize(tgt_feat.float(), dim=-1)

        # 生成器驱动的攻击循环
        gen = attacker.attack(src_img, num_iters)

        for _ in range(num_iters):
            adv_img = next(gen)                          # 获取带梯度的当前对抗图像
            adv_feat = clip_model.encode_image(clip_normalize(adv_img))
            adv_feat = F.normalize(adv_feat.float(), dim=-1)
            # 定向攻击：最小化余弦相似度的负值，即拉近对抗特征与目标特征
            loss = -F.cosine_similarity(adv_feat, tgt_feat.detach()).mean()
            loss.backward()

        adv_result = next(gen)                           # 最终对抗图像（已去梯度）

        # 保存对抗图像
        src_name  = os.path.splitext(os.path.basename(src_path))[0]
        out_path  = os.path.join(output_dir, f"{src_name}_adv.png")
        adv_pil   = transforms.ToPILImage()(adv_result.squeeze(0).clamp(0, 1).cpu())
        adv_pil.save(out_path)
        print(f"  ✓ 已保存：{out_path}")
        img_elapsed = time.time() - img_start
        per_image_times.append(img_elapsed)
        n_img_success += 1

        # 如果原图有同名 JSON，一并复制到输出目录
        src_json = os.path.splitext(src_path)[0] + ".json"
        if os.path.exists(src_json):
            shutil.copy2(src_json, os.path.join(output_dir, f"{src_name}_adv.json"))

        match_info["pairs"].append({
            "source": os.path.basename(src_path),
            "target": os.path.basename(tgt_path),
            "output": f"{src_name}_adv.png",
        })

    elapsed = time.time() - start
    match_info["timing"] = {
        "attack_time_sec": elapsed,
        "avg_time_per_image_sec": elapsed / len(pairs) if pairs else 0,
        "per_image_times_sec": per_image_times,
        "min_time_sec": min(per_image_times) if per_image_times else 0,
        "max_time_sec": max(per_image_times) if per_image_times else 0,
        "success": n_img_success,
        "failed": n_img_failed,
    }
    _save_match_info(output_dir, match_info)

    write_attack_log(
        output_dir=output_dir,
        script_name="AttackMMCoA.py (image)",
        total=len(pairs),
        success=n_img_success,
        failed=n_img_failed,
        total_time=elapsed,
        per_image_times=per_image_times,
    )

    print(f"\n{'='*60}")
    print(f"图像攻击完成！")
    print(f"  生成数量：{len(pairs)} 张对抗图像")
    print(f"  输出目录：{output_dir}")
    print(f"{'='*60}")
    return match_info


# ---------------------------------------------------------------------------
# 文本攻击（基于 CLIP 词重要性 + BERT MLM 词替换）
# ---------------------------------------------------------------------------

class _CLIPTextWrapper(nn.Module):
    """
    将 CLIP 文本编码器封装为 BertAttack 期望的接口：
        net.module.inference_text(tokenizer_output) -> {'text_embed': Tensor}
    """
    def __init__(self, clip_model):
        super().__init__()
        self._clip = clip_model

    def inference_text(self, text_inputs):
        # text_inputs 是 HuggingFace tokenizer 的 BatchEncoding
        # BertAttack 将 input_ids 传给 BERT MLM 参考网络，
        # 并调用 net.module.inference_text(text_inputs) 获取嵌入
        tokens = text_inputs.input_ids  # [B, L]
        # 截断/填充到 CLIP 的上下文长度（77）
        ctx = self._clip.context_length
        if tokens.shape[1] > ctx:
            tokens = tokens[:, :ctx]
        elif tokens.shape[1] < ctx:
            pad = torch.zeros(tokens.shape[0], ctx - tokens.shape[1],
                              dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, pad], dim=1)
        with torch.no_grad():
            embed = self._clip.encode_text(tokens)   # [B, D]
        # BertAttack 读取 ['text_embed'][:, 0, :]（cls token）
        # 通过 unsqueeze dim=1 模拟序列维度
        return {'text_embed': embed.unsqueeze(1).float()}


class _CLIPTextWrapperModule(nn.Module):
    """BertAttack 需要的额外 .module 封装（调用 net.module.inference_text）。"""
    def __init__(self, clip_model):
        super().__init__()
        self.module = _CLIPTextWrapper(clip_model)

    def forward(self, *args, **kwargs):
        return self.module.inference_text(*args, **kwargs)


def run_text_attack(source_files, target_dir, output_dir,
                    clip_model_name="ViT-B/32",
                    bert_model="bert-base-uncased",
                    match_mode="auto"):
    """
    文本级对抗攻击。

    为每张原图配对一条目标 JSON 文本，通过 BertAttack 对该文本执行词替换，
    产生对抗文本，保存为 {src_name}_adv.json（在原始 JSON 结构中新增 prompt_adv 字段）。
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n加载 CLIP 模型（文本攻击）...")
    clip_model, _ = load_clip(clip_model_name, device)
    print(f"  ✓ CLIP {clip_model_name} 加载完成")

    print("加载 BERT MLM 模型（BertAttack）...")
    try:
        from transformers import BertForMaskedLM, BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        bert_mlm = BertForMaskedLM.from_pretrained(bert_model).to(device).eval()
        print(f"  ✓ BERT {bert_model} 加载完成")
    except ImportError:
        raise ImportError("需要安装 transformers：pip install transformers")

    from attack.bert_attack import BertAttack

    net_wrapper = _CLIPTextWrapperModule(clip_model)
    bert_attacker = BertAttack(ref_net=bert_mlm, tokenizer=bert_tokenizer, cls=True)

    text_pairs = load_target_texts(target_dir, source_files, match_mode)

    match_info = {
        "attack_type": "mmcoa_text_attack",
        "method": "BertAttack_CLIP",
        "parameters": {
            "clip_model": clip_model_name,
            "bert_model": bert_model,
        },
        "pairs": [],
    }

    start = time.time()
    per_text_times = []
    n_txt_success = 0
    n_txt_failed = 0

    for i, (src_path, target_text) in enumerate(text_pairs):
        print(f"\n[{i+1}/{len(text_pairs)}] {os.path.basename(src_path)}")
        print(f"  目标文本：{target_text[:80]}...")
        txt_start = time.time()
        try:
            adv_texts = bert_attacker.attack(net_wrapper, [target_text], k=10,
                                             num_perturbation=1)
            adv_text = adv_texts[0] if adv_texts else target_text
            print(f"  对抗文本：{adv_text[:80]}...")

            src_name = os.path.splitext(os.path.basename(src_path))[0]

            # 构建对抗 JSON（保留原始结构，追加 prompt_adv 字段）
            src_json_path = os.path.splitext(src_path)[0] + ".json"
            if os.path.exists(src_json_path):
                with open(src_json_path, 'r', encoding='utf-8') as f:
                    orig_data = json.load(f)
            else:
                orig_data = {}
            orig_data["prompt_adv"] = adv_text   # 保留原始 prompt，追加对抗版本
            out_json = os.path.join(output_dir, f"{src_name}_adv.json")
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(orig_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ 已保存：{out_json}")
            txt_elapsed = time.time() - txt_start
            per_text_times.append(txt_elapsed)
            n_txt_success += 1

            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "original_text": target_text[:200],
                "adversarial_text": adv_text[:200],
                "output_json": f"{src_name}_adv.json",
            })
        except Exception as e:
            print(f"  ✗ 失败：{e}")
            n_txt_failed += 1

    elapsed = time.time() - start
    match_info["timing"] = {
        "attack_time_sec": elapsed,
        "avg_time_per_text_sec": elapsed / len(text_pairs) if text_pairs else 0,
        "per_item_times_sec": per_text_times,
        "min_time_sec": min(per_text_times) if per_text_times else 0,
        "max_time_sec": max(per_text_times) if per_text_times else 0,
        "success": n_txt_success,
        "failed": n_txt_failed,
    }
    _save_match_info(output_dir, match_info, suffix="text")

    write_attack_log(
        output_dir=output_dir,
        script_name="AttackMMCoA.py (text)",
        total=len(text_pairs),
        success=n_txt_success,
        failed=n_txt_failed,
        total_time=elapsed,
        per_image_times=per_text_times,
    )

    print(f"\n{'='*60}")
    print(f"文本攻击完成！")
    print(f"  处理数量：{len(text_pairs)} 条文本")
    print(f"  输出目录：{output_dir}")
    print(f"{'='*60}")
    return match_info


# ---------------------------------------------------------------------------
# 攻击日志写入工具
# ---------------------------------------------------------------------------

def write_attack_log(output_dir, script_name, total, success, failed,
                     total_time, per_image_times):
    """
    将攻击统计信息追加写入 output_dir/attack_log.txt。
    文件不存在时自动创建，存在时追加一条记录。
    """
    import datetime
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "attack_log.txt")

    avg_time = total_time / total if total > 0 else 0.0
    min_time = min(per_image_times) if per_image_times else 0.0
    max_time = max(per_image_times) if per_image_times else 0.0
    run_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "========================================\n",
        f"运行时间: {run_time_str}\n",
        f"脚本: {script_name}\n",
        f"输出目录: {output_dir}\n",
        "----------------------------------------\n",
        f"处理文件总数: {total}\n",
        f"成功: {success} | 失败: {failed}\n",
        f"总耗时: {total_time:.1f} 秒\n",
        f"平均每文件耗时: {avg_time:.1f} 秒\n",
        f"最快: {min_time:.1f} 秒 | 最慢: {max_time:.1f} 秒\n",
        "========================================\n",
        "\n",
    ]

    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"[日志] 已写入：{log_path}")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _save_match_info(output_dir, info, suffix=""):
    """将攻击配对信息保存为 JSON 文件。"""
    fname = f"match_info{'_' + suffix if suffix else ''}.json"
    path  = os.path.join(output_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"[已保存] {path}")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AttackMMCoA - 联合图像级 + 文本级对抗攻击"
    )
    parser.add_argument("--source_dir",    type=str, required=True,
                        help="原图目录（待扰动的图像）")
    parser.add_argument("--target_dir",    type=str, required=True,
                        help="目标目录（图像对齐目标 / 文本 JSON 来源）")
    parser.add_argument("--output_dir",    type=str, default="./output_mmcoa",
                        help="输出目录（默认：./output_mmcoa）")
    parser.add_argument("--attack_domain", type=str, default="image",
                        choices=["image", "text", "both"],
                        help="攻击域：image（图像）/ text（文本）/ both（联合，默认：image）")
    parser.add_argument("--clip_model",    type=str, default="ViT-B/32",
                        help="CLIP 模型（默认：ViT-B/32）")
    parser.add_argument("--epsilon",       type=float, default=16,
                        help="L-inf 扰动上限（0-255 尺度，默认：16）")
    parser.add_argument("--step_size",     type=float, default=1,
                        help="PGD 步长（0-255 尺度，默认：1）")
    parser.add_argument("--num_iters",     type=int,   default=300,
                        help="攻击迭代步数（默认：300）")
    parser.add_argument("--bert_model",    type=str, default="bert-base-uncased",
                        help="BERT MLM 模型（文本攻击，默认：bert-base-uncased）")
    parser.add_argument("--match_mode",    type=str, default="auto",
                        choices=["auto", "single", "random"],
                        help="配对模式：auto/single/random（默认：auto）")
    parser.add_argument("--input_res",     type=int,   default=224,
                        help="输入分辨率（默认：224）")
    parser.add_argument("--seed",          type=int,   default=DEFAULT_RANDOM_SEED,
                        help=f"随机种子（默认：{DEFAULT_RANDOM_SEED}）")

    args = parser.parse_args()
    seed_everything(args.seed)

    print(f"\n{'='*60}")
    print(f"AttackMMCoA - 联合图像 + 文本对抗攻击")
    print(f"{'='*60}")
    print(f"原图目录：    {args.source_dir}")
    print(f"目标目录：    {args.target_dir}")
    print(f"输出目录：    {args.output_dir}")
    print(f"攻击域：      {args.attack_domain}")
    print(f"CLIP 模型：   {args.clip_model}")
    print(f"Epsilon：     {args.epsilon}")
    print(f"步长：        {args.step_size}")
    print(f"迭代步数：    {args.num_iters}")
    print(f"配对模式：    {args.match_mode}")
    print(f"随机种子：    {args.seed}")
    print(f"{'='*60}")

    source_files = get_image_files(args.source_dir)
    if not source_files:
        raise ValueError(f"原图目录中未找到图像：{args.source_dir}")

    if args.attack_domain in ("image", "both"):
        pairs = match_source_target_pairs(
            args.source_dir, args.target_dir,
            match_mode=args.match_mode,
            input_res=args.input_res,
        )
        run_image_attack(
            pairs,
            args.output_dir,
            clip_model_name=args.clip_model,
            epsilon=args.epsilon,
            step_size=args.step_size,
            num_iters=args.num_iters,
        )

    if args.attack_domain in ("text", "both"):
        run_text_attack(
            source_files,
            args.target_dir,
            args.output_dir,
            clip_model_name=args.clip_model,
            bert_model=args.bert_model,
            match_mode=args.match_mode,
        )


if __name__ == "__main__":
    main()
