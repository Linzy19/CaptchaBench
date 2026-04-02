"""
AttackVLMNew.py - LAVIS-based VLM Transfer Attack（图像-图像特征对齐）

攻击策略：
    - 以原图（source）作为待扰动的图像
    - 以目标图（target）作为攻击目标（BLIP/BLIP2 视觉编码器特征对齐）
    - 通过 PGD 驱动原图的对抗特征向目标图特征靠拢（最大化 cosine similarity）
    - 支持 BLIP (blip_caption/base_coco) 和 BLIP2 (blip2_opt/pretrain_opt2.7b) 两种模型

与项目其他攻击脚本的对应关系：
    AttackMI.py       → AdversarialAttacks (ResNet/ViT 特征对齐)
    AttackASPL.py     → Anti-DreamBooth
    AttackMMCoA.py    → MMCoA
    AttackNightshade.py → nightshade-release
    AttackXTransfer.py  → XTransferBench
    AttackVLM.py (Attack-Bard) → BLIP2/InstructBLIP/MiniGPT4 文字攻击
    AttackVLMNew.py   → AttackVLM/LAVIS_tool (BLIP/BLIP2 图像特征迁移攻击)

基础用法：
    python AttackVLMNew.py \\
        --source_dir ../Data_source/sample_source \\
        --target_dir ../Data_source/sample_target \\
        --output_dir ./output_attackvlm_new

自定义参数：
    python AttackVLMNew.py \\
        --source_dir ./source --target_dir ./target \\
        --output_dir ./output \\
        --epsilon 16 --step_size 1 --steps 300 \\
        --model_name blip_caption --model_type base_coco \\
        --match_mode pair

BLIP2 模型：
    python AttackVLMNew.py \\
        --source_dir ./source --target_dir ./target \\
        --output_dir ./output \\
        --model_name blip2_opt --model_type pretrain_opt2.7b
"""

import argparse
import os
import sys
import json
import glob
import time
import random
import datetime
import shutil

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

# LAVIS 是本攻击的核心依赖
try:
    from lavis.models import load_model_and_preprocess
except ImportError as e:
    print(f"[错误] 无法导入 lavis：{e}")
    print("[提示] 请先激活 attackvlm 环境：conda activate attackvlm")
    print("       安装命令：pip install salesforce-lavis")
    sys.exit(1)

DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# 随机种子工具
# ===========================================================================

def seed_everything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ===========================================================================
# 图像文件工具
# ===========================================================================

def get_image_files(directory):
    """获取目录下所有图像文件路径（排序后返回）"""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
                  "*.JPEG", "*.JPG", "*.PNG"]
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def match_pairs(source_files, target_files, match_mode="pair"):
    """
    将 source 和 target 文件配对。

    match_mode:
        pair   - 按排序顺序逐一配对（source[i] ↔ target[i]），
                 数量不等时以较短的为准
        single - 所有 source 图片对同一张 target（取 target[0]）
    """
    if not source_files:
        raise ValueError(f"source 目录下没有图像文件")
    if not target_files:
        raise ValueError(f"target 目录下没有图像文件")

    if match_mode == "single":
        target_file = target_files[0]
        pairs = [(src, target_file) for src in source_files]
        print(f"[配对] single 模式：所有 {len(source_files)} 张 source → target[0]: "
              f"{os.path.basename(target_file)}")
    else:
        # pair 模式
        n = min(len(source_files), len(target_files))
        pairs = list(zip(source_files[:n], target_files[:n]))
        if len(source_files) != len(target_files):
            print(f"[警告] source({len(source_files)}) 与 target({len(target_files)}) "
                  f"数量不等，按较少的 {n} 张配对")
        print(f"[配对] pair 模式：{n} 对 source/target")

    return pairs


# ===========================================================================
# LAVIS 归一化常量（BLIP/BLIP2 的标准归一化参数）
# ===========================================================================

LAVIS_MEAN = (0.48145466, 0.4578275, 0.40821073)
LAVIS_STD  = (0.26862954, 0.26130258, 0.27577711)

# 反归一化（将 LAVIS 归一化后的张量还原到 [0, 1]）
inverse_normalize = torchvision.transforms.Normalize(
    mean=[-m / s for m, s in zip(LAVIS_MEAN, LAVIS_STD)],
    std=[1.0 / s for s in LAVIS_STD],
)


def load_image_for_lavis(path, vis_processors):
    """
    用 LAVIS vis_processors 加载单张图像。
    返回形状为 (C, H, W) 的 float tensor（已归一化）。
    """
    img = Image.open(path).convert("RGB")
    return vis_processors["eval"](img)


# ===========================================================================
# 核心攻击函数
# ===========================================================================

def extract_features(model, image_tensor, model_name):
    """
    从 BLIP/BLIP2 视觉编码器提取 CLS token 特征。

    返回形状：(B, D) 的 L2 归一化特征向量
    """
    sample = {"image": image_tensor}
    if "blip2" in model_name:
        feats = model.forward_encoder_image(sample)
    else:
        feats = model.forward_encoder(sample)
    feats = feats[:, 0, :]                                    # CLS token
    feats = feats / feats.norm(dim=1, keepdim=True)           # L2 归一化
    return feats


def pgd_attack_single(model, src_tensor, tgt_tensor, model_name,
                      epsilon, step_size, total_step):
    """
    对单张图像对执行 PGD 迁移攻击。

    目标：最大化 adv 特征与 target 特征的 cosine similarity。

    参数：
        src_tensor  - 已 LAVIS 归一化的 source 张量，形状 (1, C, H, W)
        tgt_tensor  - 已 LAVIS 归一化的 target 张量，形状 (1, C, H, W)
        epsilon     - 扰动上界（LAVIS 归一化空间，由像素空间 eps/255 除以 std 得到）
        step_size   - PGD 步长（同上）

    返回：
        adv_image_raw - 反归一化后 [0,1] 的对抗图像张量，形状 (1, C, H, W)
        final_sim     - 最终 cosine similarity（float）
    """
    src_tensor = src_tensor.to(device)
    tgt_tensor = tgt_tensor.to(device)

    # 预计算目标特征（不需要梯度）
    with torch.no_grad():
        tgt_feats = extract_features(model, tgt_tensor, model_name)

    # 初始化扰动
    delta = torch.zeros_like(src_tensor, requires_grad=True)

    final_sim = 0.0
    for step in range(total_step):
        adv_image = src_tensor + delta
        sample_adv = {"image": adv_image}

        adv_feats = extract_features(model, adv_image, model_name)
        cos_sim = torch.mean(torch.sum(adv_feats * tgt_feats, dim=1))
        cos_sim.backward()

        grad = delta.grad.detach()
        delta_data = torch.clamp(
            delta + step_size * torch.sign(grad),
            min=-epsilon,
            max=epsilon,
        )
        delta.data = delta_data
        delta.grad.zero_()

        if step % 50 == 0 or step == total_step - 1:
            print(f"    step {step:3d}/{total_step}  cos_sim={cos_sim.item():.5f}  "
                  f"max_delta={torch.max(torch.abs(delta_data)).item():.4f}")
        final_sim = cos_sim.item()

    # 反归一化回 [0,1] 空间
    adv_image_raw = torch.clamp(
        inverse_normalize(src_tensor + delta.detach()), 0.0, 1.0
    )
    return adv_image_raw, final_sim


# ===========================================================================
# 攻击日志写入工具
# ===========================================================================

def write_attack_log(output_dir, script_name, total, success, failed,
                     total_time, per_image_times):
    """将攻击统计信息追加写入 output_dir/attack_log.txt"""
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

    with open(log_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[日志] 已写入：{log_path}")


# ===========================================================================
# 主攻击流程
# ===========================================================================

def run_attack(pairs, output_dir, model_name, model_type,
               epsilon_px, step_size_px, total_step):
    """
    批量攻击所有 (source, target) 图像对。

    参数：
        pairs         - [(src_path, tgt_path), ...]
        epsilon_px    - 像素空间扰动上界（0-255）
        step_size_px  - 像素空间 PGD 步长（0-255）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载 LAVIS 模型和预处理器
    print(f"\n[模型] 加载 LAVIS 模型：{model_name} / {model_type}  (device={device})")
    model_load_start = time.time()
    blip_model, vis_processors, _ = load_model_and_preprocess(
        name=model_name,
        model_type=model_type,
        is_eval=True,
        device=device,
    )
    print(f"[模型] 加载完成，耗时 {time.time() - model_load_start:.1f}s")

    # 将像素空间 epsilon/step_size 转换到 LAVIS 归一化空间
    # LAVIS 归一化：x_norm = (x_raw - mean) / std
    # 扰动约束：delta_norm = delta_raw / std（mean 在加减中抵消）
    scaling_tensor = torch.tensor(LAVIS_STD, device=device).reshape(3, 1, 1).unsqueeze(0)
    epsilon   = epsilon_px   / 255.0 / scaling_tensor
    step_size = step_size_px / 255.0 / scaling_tensor

    match_info = {
        "attack_type": "vlm_image_transfer_attack",
        "method": "PGD_LAVIS_FeatureAlignment",
        "model_name": model_name,
        "model_type": model_type,
        "parameters": {
            "epsilon_px": epsilon_px,
            "step_size_px": step_size_px,
            "total_step": total_step,
        },
        "pairs": [],
    }

    total_start = time.time()
    per_image_times = []
    n_success = 0
    n_failed = 0

    for i, (src_path, tgt_path) in enumerate(tqdm(pairs, desc="Attacking")):
        src_name = os.path.splitext(os.path.basename(src_path))[0]
        print(f"\n[{i+1}/{len(pairs)}] source: {os.path.basename(src_path)}"
              f"  target: {os.path.basename(tgt_path)}")

        img_start = time.time()
        try:
            # 加载图像
            src_tensor = load_image_for_lavis(src_path, vis_processors).unsqueeze(0)
            tgt_tensor = load_image_for_lavis(tgt_path, vis_processors).unsqueeze(0)

            # 执行 PGD 攻击
            adv_image, final_sim = pgd_attack_single(
                blip_model, src_tensor, tgt_tensor, model_name,
                epsilon, step_size, total_step,
            )

            # 保存对抗图像
            output_path = os.path.join(output_dir, f"{src_name}_adv.png")
            torchvision.utils.save_image(adv_image.squeeze(0), output_path)
            print(f"    ✓ 已保存：{output_path}  final_cos_sim={final_sim:.5f}")

            # 同步复制 source JSON（若存在）
            src_json = os.path.splitext(src_path)[0] + ".json"
            if os.path.exists(src_json):
                shutil.copy2(src_json, os.path.join(output_dir, f"{src_name}_adv.json"))

            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "target": os.path.basename(tgt_path),
                "output": f"{src_name}_adv.png",
                "final_cos_sim": final_sim,
            })
            n_success += 1

        except Exception as e:
            print(f"    ✗ 失败：{e}")
            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "target": os.path.basename(tgt_path),
                "output": None,
                "error": str(e),
            })
            n_failed += 1

        img_elapsed = time.time() - img_start
        per_image_times.append(img_elapsed)

        # 释放 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # 写入 match_info.json
    match_info["timing"] = {
        "total_time_sec": total_time,
        "total_images": len(pairs),
        "avg_time_per_image_sec": total_time / len(pairs) if pairs else 0,
        "per_image_times_sec": per_image_times,
        "min_time_sec": min(per_image_times) if per_image_times else 0,
        "max_time_sec": max(per_image_times) if per_image_times else 0,
        "success": n_success,
        "failed": n_failed,
    }
    with open(os.path.join(output_dir, "match_info.json"), "w", encoding="utf-8") as f:
        json.dump(match_info, f, indent=2, ensure_ascii=False)

    write_attack_log(
        output_dir=output_dir,
        script_name="AttackVLMNew.py",
        total=len(pairs),
        success=n_success,
        failed=n_failed,
        total_time=total_time,
        per_image_times=per_image_times,
    )

    # 打印汇总
    print(f"\n{'=' * 60}")
    print(f"AttackVLMNew 完成！")
    print(f"{'=' * 60}")
    print(f"  生成对抗图像：{n_success}")
    print(f"  失败：{n_failed}")
    print(f"  输出目录：{output_dir}")
    print(f"  总耗时：{total_time:.1f}s")
    if len(pairs):
        print(f"  平均每张：{total_time/len(pairs):.1f}s")
    print(f"{'=' * 60}")

    return match_info


# ===========================================================================
# 命令行入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AttackVLMNew - LAVIS-based VLM Transfer Attack（图像-图像特征对齐）"
    )

    # 必需参数
    parser.add_argument("--source_dir", type=str, required=True,
                        help="source 图像目录（待扰动的原始图像）")
    parser.add_argument("--target_dir", type=str, required=True,
                        help="target 图像目录（攻击目标图像）")
    parser.add_argument("--output_dir", type=str, default="./output_attackvlm_new",
                        help="输出目录（存放对抗图像和日志）")

    # 攻击参数
    parser.add_argument("--epsilon", type=float, default=16,
                        help="最大扰动量（0-255 像素空间，默认 16）")
    parser.add_argument("--step_size", type=float, default=1,
                        help="PGD 步长（0-255 像素空间，默认 1）")
    parser.add_argument("--steps", type=int, default=300,
                        help="PGD 迭代步数（默认 300）")

    # 模型参数
    parser.add_argument("--model_name", type=str, default="blip_caption",
                        choices=["blip_caption", "blip2_opt", "blip2_t5"],
                        help="LAVIS 模型名称（默认 blip_caption）")
    parser.add_argument("--model_type", type=str, default="base_coco",
                        help="LAVIS 模型类型（默认 base_coco）")

    # 配对模式
    parser.add_argument("--match_mode", type=str, default="pair",
                        choices=["pair", "single"],
                        help="source/target 配对模式：pair=逐一配对，single=所有source→同一target（默认 pair）")

    # 其他
    parser.add_argument("--seed", type=int, default=2023, help="随机种子")
    parser.add_argument("--max_images", type=int, default=None,
                        help="最多处理图像数（默认全部）")

    args = parser.parse_args()

    seed_everything(args.seed)

    print(f"\n{'=' * 60}")
    print(f"AttackVLMNew - LAVIS VLM Transfer Attack")
    print(f"{'=' * 60}")
    print(f"Source Dir:   {args.source_dir}")
    print(f"Target Dir:   {args.target_dir}")
    print(f"Output Dir:   {args.output_dir}")
    print(f"Epsilon:      {args.epsilon} px")
    print(f"Step Size:    {args.step_size} px")
    print(f"Steps:        {args.steps}")
    print(f"Model:        {args.model_name} / {args.model_type}")
    print(f"Match Mode:   {args.match_mode}")
    print(f"Device:       {device}")
    print(f"Seed:         {args.seed}")
    print(f"{'=' * 60}")

    # 获取图像文件列表
    source_files = get_image_files(args.source_dir)
    target_files = get_image_files(args.target_dir)

    if not source_files:
        print(f"[错误] source 目录下没有图像：{args.source_dir}")
        sys.exit(1)
    if not target_files:
        print(f"[错误] target 目录下没有图像：{args.target_dir}")
        sys.exit(1)

    print(f"[数据] source 图像数：{len(source_files)}")
    print(f"[数据] target 图像数：{len(target_files)}")

    # 配对
    pairs = match_pairs(source_files, target_files, args.match_mode)

    # 限制数量
    if args.max_images and len(pairs) > args.max_images:
        print(f"[限制] 只处理前 {args.max_images} 对（共 {len(pairs)} 对）")
        pairs = pairs[:args.max_images]

    # 执行攻击
    run_attack(
        pairs=pairs,
        output_dir=args.output_dir,
        model_name=args.model_name,
        model_type=args.model_type,
        epsilon_px=args.epsilon,
        step_size_px=args.step_size,
        total_step=args.steps,
    )


if __name__ == "__main__":
    main()
