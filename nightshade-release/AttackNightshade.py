"""
AttackNightshade.py - Nightshade VAE 潜在空间投毒攻击

攻击策略：
    - 以原图（source）作为待扰动的图像
    - image 模式（默认）：直接使用目标图的 VAE 潜在表示作为优化目标
        → 扰动原图使其 VAE 潜在表示向目标图的潜在表示对齐
        → 目标图的 VAE 潜在表示直接编码（无需 SD 生图）
    - concept 模式：使用目标 JSON 的 prompt + label 文本通过 SD2.1 生成目标图，
        再以生成图的 VAE 潜在表示作为优化目标（原始 Nightshade 行为）

优化循环（完整保留自 opt.py）：
    - 500 步，步长线性衰减
    - 对 modifier 执行符号梯度下降，约束在 [-max_change, max_change]
    - 损失函数：(vae.encode(src + modifier) - target_latent).norm()
    - 图像范围 [-1, 1]（SD 约定：pixel / 127.5 - 1.0）

参考：
    nightshade-release/opt.py (PoisonGeneration 类)
    nightshade-release/gen_poison.py（数据接口参考）

image 模式（默认）：
    python AttackNightshade.py \\
        --source_dir ../Data_source/sample_source \\
        --target_dir ../Data_source/sample_target \\
        --output_dir ./output_nightshade

concept 模式（目标由目标 JSON 文本通过 SD 生成）：
    python AttackNightshade.py \\
        --source_dir ../Data_source/sample_source \\
        --target_dir ../Data_source/sample_target \\
        --output_dir ./output_nightshade_concept \\
        --mode concept

自定义参数：
    python AttackNightshade.py \\
        --source_dir ./source --target_dir ./target \\
        --output_dir ./output \\
        --eps 0.05 --steps 500 --match_mode random --seed 2023
"""

import torch
import os
import sys
import json
import glob
import time
import random
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# ---------------------------------------------------------------------------
# 路径配置：将 nightshade-release 根目录加入 sys.path，使 opt.py 可导入
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from opt import PoisonGeneration, img2tensor, tensor2img
from diffusers import StableDiffusionPipeline

DEFAULT_RANDOM_SEED = 2023


# ===========================================================================
# 扩展版 PoisonGeneration，支持直接传入目标图像
# ===========================================================================

class ExtendedPoisonGeneration(PoisonGeneration):
    """
    在 PoisonGeneration 基础上新增 generate_one_with_target_image()。

    原始 generate_one() 调用 generate_target(concept_text) 通过完整 SD 管线生成目标图。
    本子类新增方法，直接接收目标 PIL 图像，跳过 SD 生图步骤，
    同时完整保留原始优化循环不变。

    同时重写 load_model()，使其使用传入的 sd_model_path 而非硬编码的
    "stabilityai/stable-diffusion-2-1"。
    """

    def __init__(self, target_concept, device, eps=0.05,
                 sd_model_path="stabilityai/stable-diffusion-2-1"):
        # 在调用父类 __init__ 之前保存模型路径（父类 __init__ 会调用 load_model）
        self._sd_model_path = sd_model_path
        super().__init__(target_concept, device, eps)

    def load_model(self):
        """重写父类方法，使用外部传入的 sd_model_path 而非硬编码路径。"""
        pipeline = StableDiffusionPipeline.from_pretrained(
            self._sd_model_path,
            safety_checker=None,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        return pipeline.to(self.device)

    def generate_one_with_target_image(self, source_pil: Image.Image,
                                       target_pil: Image.Image,
                                       t_size: int = 500) -> Image.Image:
        """
        在 VAE 潜在空间中将 source_pil 向 target_pil 方向扰动。

        参数
        ----
        source_pil : 待扰动的原始 PIL RGB 图像
        target_pil : 固定的优化目标 PIL RGB 图像
        t_size     : 优化步数（默认 500，与原始实现一致）

        返回
        ----
        对抗 PIL 图像（大小为 SD 分辨率 512×512）
        """
        # 转换为 SD 张量，范围 [-1, 1]
        source_tensor = img2tensor(source_pil).to(self.device)
        target_tensor = img2tensor(target_pil).to(self.device)

        source_tensor = source_tensor.half()
        target_tensor = target_tensor.half()

        # 预计算固定目标潜在表示
        with torch.no_grad():
            target_latent = self.get_latent(target_tensor)

        modifier = torch.clone(source_tensor) * 0.0

        max_change = self.eps / 0.5   # 从 [0,1] 缩放到 [-1,1] 空间
        step_size = max_change

        for i in range(t_size):
            # 线性衰减步长（与原始 opt.py 保持一致）
            actual_step_size = step_size - (step_size - step_size / 100) / t_size * i
            modifier.requires_grad_(True)

            adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
            adv_latent = self.get_latent(adv_tensor)

            loss = (adv_latent - target_latent).norm()
            tot_loss = loss.sum()

            grad = torch.autograd.grad(tot_loss, modifier)[0]

            modifier = modifier - torch.sign(grad) * actual_step_size
            modifier = torch.clamp(modifier, -max_change, max_change)
            modifier = modifier.detach()

            if i % 50 == 0:
                print(f"    [步骤 {i:>4}/{t_size}] 损失：{loss.mean().item():.4f}")

        final_adv = torch.clamp(modifier + source_tensor, -1.0, 1.0)
        return tensor2img(final_adv)


# ===========================================================================
# 辅助函数
# ===========================================================================

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


def get_image_files(directory: str):
    """递归扫描目录，返回所有支持格式的图像路径（已排序）。"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp',
                  '*.JPEG', '*.JPG', '*.PNG']
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def build_target_text_from_json(json_path: str):
    """从 JSON 注释文件提取 'prompt Label: label' 字符串。"""
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


def load_target_paths(target_dir: str):
    """
    加载目标图像路径，优先读取 target_paths.json，否则自动扫描目录。
    """
    json_path = os.path.join(target_dir, "target_paths.json")
    if os.path.exists(json_path):
        try:
            print(f"[加载] 从 JSON 读取目标路径：{json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            target_files = data if isinstance(data, list) else data.get("target_files", [])
            if target_files:
                print(f"[加载] 共加载 {len(target_files)} 条目标路径")
                return target_files
        except Exception as e:
            print(f"[警告] 读取 {json_path} 失败：{e}")
    print(f"[自动扫描] 扫描目录：{target_dir}")
    target_files = get_image_files(target_dir)
    if not target_files:
        raise ValueError(f"目标目录中未找到图像文件：{target_dir}")
    print(f"[自动扫描] 找到 {len(target_files)} 张图像")
    return target_files


# ===========================================================================
# 原图-目标配对逻辑
# ===========================================================================

def match_source_target_pairs(source_dir: str, target_dir: str,
                               match_mode: str = "auto", mode: str = "image"):
    """
    构建 (src_path, tgt_path_or_text) 配对列表。

    image 模式  : tgt 为目标图像路径。
    concept 模式: tgt 为概念文字（从目标 JSON 的 prompt + label 提取）。

    配对模式：
        auto   —— 目标只有1张时用 single，否则用 random
        single —— 所有原图攻击向第1个目标
        random —— 每张原图随机选择一个目标
    """
    source_files = get_image_files(source_dir)
    target_files = load_target_paths(target_dir)

    if not source_files:
        raise ValueError(f"原图目录中未找到图像：{source_dir}")
    if not target_files:
        raise ValueError(f"未找到目标图像：{target_dir}")

    if match_mode == "auto":
        match_mode = "single" if len(target_files) == 1 else "random"
        print(f"\n[自动] {len(target_files)} 个目标 -> '{match_mode}' 模式")

    print(f"[信息] 配对模式：{match_mode.upper()} | 原图：{len(source_files)} 张 | "
          f"目标：{len(target_files)} 张")

    pairs = []

    if match_mode == "single":
        tgt = target_files[0]
        if mode == "concept":
            tgt_json = os.path.splitext(tgt)[0] + ".json"
            tgt = build_target_text_from_json(tgt_json) if os.path.exists(tgt_json) else tgt
        print(f"\n[单目标] 所有原图 -> {os.path.basename(str(tgt))}")
        for src in source_files:
            pairs.append((src, tgt))

    elif match_mode == "random":
        for src in tqdm(source_files, desc="构建配对"):
            tgt = random.choice(target_files)
            if mode == "concept":
                tgt_json = os.path.splitext(tgt)[0] + ".json"
                tgt = build_target_text_from_json(tgt_json) if os.path.exists(tgt_json) else tgt
            pairs.append((src, tgt))

    if not pairs:
        raise ValueError("未找到有效的原图-目标对！")
    print(f"[完成] 共配对 {len(pairs)} 对")
    return pairs


# ===========================================================================
# 核心攻击执行器
# ===========================================================================

def run_attack(
    pairs,
    output_dir: str,
    sd_model: str,
    eps: float,
    steps: int,
    mode: str,
):
    """
    对所有原图-目标对执行 Nightshade 投毒攻击。

    参数
    ----
    pairs      : image 模式为 (src_path, tgt_path) 列表，
                 concept 模式为 (src_path, concept_str) 列表
    output_dir : 对抗图像和 match_info.json 的保存目录
    sd_model   : SD 模型名称或本地路径（用于 VAE）
    eps        : [0,1] 尺度的扰动预算
    steps      : 优化步数
    mode       : "image" 或 "concept"
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print("加载 Nightshade PoisonGeneration 模型 ...")
    print(f"  SD 模型：{sd_model}")
    print(f"  设备：   {device}")
    print(f"  Eps：    {eps}")
    print(f"  步数：   {steps}")
    print(f"  模式：   {mode}")
    print("=" * 60)

    t0 = time.time()
    # image 模式下 target_concept 不会被使用，传入占位符
    poison_gen = ExtendedPoisonGeneration(
        target_concept="__placeholder__",
        device=device,
        eps=eps,
        sd_model_path=sd_model,
    )
    model_load_time = time.time() - t0
    print(f"  模型加载耗时：{model_load_time:.2f}s")

    print(f"\n{'=' * 60}")
    print("攻击参数：")
    print(f"  Epsilon：{eps}（最大像素变化量 ÷ 0.5，[-1,1] 空间）")
    print(f"  步数：   {steps}")
    print(f"  模式：   {mode}")
    print(f"  配对数量：{len(pairs)}")
    print(f"{'=' * 60}\n")

    match_info = {
        "attack_type": "nightshade_vae_latent_poisoning",
        "method": "VAE_LatentOptimization",
        "sd_model": sd_model,
        "parameters": {
            "eps": eps,
            "steps": steps,
            "mode": mode,
        },
        "pairs": [],
    }

    attack_start = time.time()
    per_image_times = []
    n_success = 0
    n_failed = 0
    json_copied = 0

    for i, (src_path, tgt) in enumerate(pairs):
        tgt_label = tgt if isinstance(tgt, str) and not os.path.exists(tgt) else os.path.basename(str(tgt))
        print(f"\n[{i+1}/{len(pairs)}] 攻击：{os.path.basename(src_path)}"
              f" -> {tgt_label[:60]}")
        img_start = time.time()
        try:
            src_pil = Image.open(src_path).convert("RGB")

            if mode == "image":
                tgt_pil = Image.open(tgt).convert("RGB")
                adv_pil = poison_gen.generate_one_with_target_image(
                    source_pil=src_pil,
                    target_pil=tgt_pil,
                    t_size=steps,
                )
            else:
                # concept 模式：调用原始 generate_one()，通过 SD 生成目标图
                adv_pil = poison_gen.generate_one(
                    pil_image=src_pil,
                    target_concept=tgt,
                )

            src_name = os.path.splitext(os.path.basename(src_path))[0]
            output_path = os.path.join(output_dir, f"{src_name}_adv.png")
            adv_pil.save(output_path)
            print(f"  ✓ 已保存：{output_path}")
            img_elapsed = time.time() - img_start
            per_image_times.append(img_elapsed)
            n_success += 1

            # 复制同名 JSON 到输出目录
            src_json = os.path.splitext(src_path)[0] + ".json"
            if os.path.exists(src_json):
                shutil.copy2(src_json, os.path.join(output_dir, f"{src_name}_adv.json"))
                json_copied += 1

            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "target": str(tgt)[:200],
                "output": f"{src_name}_adv.png",
            })

        except Exception as e:
            print(f"  ✗ 失败：{e}")
            n_failed += 1
            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "target": str(tgt)[:200],
                "output": None,
                "error": str(e),
            })

    attack_time = time.time() - attack_start

    match_info["timing"] = {
        "model_load_time_sec": model_load_time,
        "attack_time_sec": attack_time,
        "total_time_sec": model_load_time + attack_time,
        "avg_time_per_image_sec": attack_time / len(pairs) if pairs else 0,
        "per_image_times_sec": per_image_times,
        "min_time_sec": min(per_image_times) if per_image_times else 0,
        "max_time_sec": max(per_image_times) if per_image_times else 0,
        "success": n_success,
        "failed": n_failed,
    }

    with open(os.path.join(output_dir, "match_info.json"), 'w', encoding='utf-8') as f:
        json.dump(match_info, f, indent=2, ensure_ascii=False)

    write_attack_log(
        output_dir=output_dir,
        script_name="AttackNightshade.py",
        total=len(pairs),
        success=n_success,
        failed=n_failed,
        total_time=model_load_time + attack_time,
        per_image_times=per_image_times,
    )

    success = sum(1 for p in match_info["pairs"] if p.get("output"))
    print(f"\n{'=' * 60}")
    print("Nightshade 攻击完成！")
    print(f"{'=' * 60}")
    print(f"  生成数量：     {success}/{len(pairs)} 张对抗图像")
    print(f"  JSON 复制：    {json_copied}/{len(pairs)}")
    print(f"  输出目录：     {output_dir}")
    print(f"  模型加载：     {model_load_time:.2f}s")
    print(f"  攻击耗时：     {attack_time:.2f}s")
    print(f"  总耗时：       {model_load_time + attack_time:.2f}s")
    print(f"  每张平均：     {attack_time / len(pairs):.2f}s")
    print(f"{'=' * 60}")

    return match_info


# ===========================================================================
# 攻击日志写入工具
# ===========================================================================

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


# ===========================================================================
# 命令行入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AttackNightshade - VAE 潜在空间投毒攻击"
    )

    parser.add_argument("--source_dir", type=str, required=True,
                        help="原图目录（待投毒的图像）")
    parser.add_argument("--target_dir", type=str, required=True,
                        help="目标目录（潜在空间对齐目标）")
    parser.add_argument("--output_dir", type=str, default="./output_nightshade",
                        help="输出目录（默认：./output_nightshade）")
    parser.add_argument("--sd_model", type=str,
                        default="stabilityai/stable-diffusion-2-1",
                        help="Stable Diffusion 模型名称或本地路径")
    parser.add_argument("--eps", type=float, default=0.05,
                        help="[0,1] 尺度的扰动预算（默认：0.05）")
    parser.add_argument("--steps", type=int, default=500,
                        help="优化步数（默认：500）")
    parser.add_argument("--mode", type=str, default="image",
                        choices=["image", "concept"],
                        help="image：直接使用目标图；concept：通过 SD 文本生成目标（默认：image）")
    parser.add_argument("--match_mode", type=str, default="auto",
                        choices=["auto", "single", "random"],
                        help="配对模式：auto/single/random（默认：auto）")
    parser.add_argument("--resolution", type=int, default=512,
                        help="处理分辨率（默认：512，SD 约定）")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"随机种子（默认：{DEFAULT_RANDOM_SEED}）")

    args = parser.parse_args()
    seed_everything(args.seed)

    print(f"\n{'=' * 60}")
    print("AttackNightshade - VAE 潜在空间投毒攻击")
    print("=" * 60)
    print(f"  原图目录：  {args.source_dir}")
    print(f"  目标目录：  {args.target_dir}")
    print(f"  输出目录：  {args.output_dir}")
    print(f"  攻击模式：  {args.mode}")
    print(f"  SD 模型：   {args.sd_model}")
    print(f"  Eps：       {args.eps}")
    print(f"  步数：      {args.steps}")
    print(f"  配对模式：  {args.match_mode}")
    print(f"  随机种子：  {args.seed}")
    print("=" * 60)

    pairs = match_source_target_pairs(
        args.source_dir,
        args.target_dir,
        match_mode=args.match_mode,
        mode=args.mode,
    )

    run_attack(
        pairs=pairs,
        output_dir=args.output_dir,
        sd_model=args.sd_model,
        eps=args.eps,
        steps=args.steps,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
