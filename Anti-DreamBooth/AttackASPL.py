"""
AttackASPL.py - Anti-DreamBooth ASPL 攻击（SD VAE 潜在空间 PGD）

攻击策略：
    - 保护原图不被 DreamBooth 微调所学习
    - fast 模式（默认）：仅使用 VAE 的 PGD
        扰动原图使其 SD VAE 潜在表示向目标图的潜在表示对齐。
        只需 VAE 编码器（fast 模式无需 UNet / TextEncoder）。
        损失函数：MSE(vae.encode(src + delta).latent_dist.mean, target_latent)
        更新规则：delta += alpha * sign(grad)；delta = clamp(delta, -eps, +eps)
    - full 模式：完整 ASPL（代理 UNet 迭代训练 + PGD，与 aspl.py 一致）
        委托给 aspl.py 的原始 pgd_attack() + train_one_epoch() 逻辑。

图像范围约定：Stable Diffusion 使用 [-1, 1]（像素 / 127.5 - 1.0）。

参考：Anti-DreamBooth/attacks/aspl.py

fast 模式（默认）：
    python AttackASPL.py --source_dir ../Data_source/sample_source \\
                         --target_dir ../Data_source/sample_target \\
                         --output_dir ./output_aspl_fast

full ASPL 模式：
    python AttackASPL.py --source_dir ../Data_source/sample_source \\
                         --target_dir ../Data_source/sample_target \\
                         --output_dir ./output_aspl_full \\
                         --mode full --pgd_steps 200

自定义参数：
    python AttackASPL.py --source_dir ./source --target_dir ./target \\
                         --output_dir ./output \\
                         --sd_model stabilityai/stable-diffusion-2-1 \\
                         --pgd_alpha 0.004 --pgd_eps 0.05 --pgd_steps 200 \\
                         --match_mode random --seed 42
"""

import torch
import torch.nn.functional as F
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

# ---------------------------------------------------------------------------
# 路径配置：将 Anti-DreamBooth 根目录加入 sys.path，使 attacks 包可导入
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

DEFAULT_RANDOM_SEED = 42
SD_IMAGE_SIZE = 512   # Stable Diffusion 标准输入尺寸


# ===========================================================================
# 随机种子工具
# ===========================================================================

def seed_everything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ===========================================================================
# 图像 I/O 辅助函数（SD [-1,1] 范围约定）
# ===========================================================================

def get_image_files(directory: str):
    """递归扫描目录，返回所有支持格式的图像路径（已排序）。"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp',
                  '*.JPEG', '*.JPG', '*.PNG']
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def load_target_paths(target_dir: str):
    """
    加载目标图像路径，优先读取 target_paths.json，否则自动扫描目录。
    """
    json_path = os.path.join(target_dir, "target_paths.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            target_files = data if isinstance(data, list) else data.get("target_files", [])
            if target_files:
                print(f"[加载] 从 JSON 读取了 {len(target_files)} 条目标路径")
                return target_files
        except Exception as e:
            print(f"[警告] 读取 {json_path} 失败：{e}")
    print(f"[自动扫描] 扫描目录：{target_dir}")
    target_files = get_image_files(target_dir)
    if not target_files:
        raise ValueError(f"目标目录中未找到图像文件：{target_dir}")
    print(f"[自动扫描] 找到 {len(target_files)} 张图像")
    return target_files


def load_json_prompt(image_path: str) -> str:
    """从同名 JSON 文件中读取 'prompt' 字段。"""
    json_path = os.path.splitext(image_path)[0] + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("prompt", "")
        except Exception:
            pass
    return ""


def pil_to_sd_tensor(img_pil: Image.Image,
                     size: int = SD_IMAGE_SIZE) -> torch.Tensor:
    """PIL 图像转换为 (1,3,size,size) float32，范围 [-1,1]。"""
    img_pil = img_pil.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img_pil, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0
    return tensor.unsqueeze(0)


def sd_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """(1,3,H,W) [-1,1] 张量转换为 PIL 图像。"""
    arr = ((tensor.squeeze(0).detach().cpu().clamp(-1, 1) + 1.0) * 127.5)
    return Image.fromarray(arr.permute(1, 2, 0).to(torch.uint8).numpy())


# ===========================================================================
# 原图-目标图配对逻辑
# ===========================================================================

def match_source_target_pairs(source_dir: str, target_dir: str,
                               match_mode: str = "auto"):
    """
    构建 (src_path, tgt_path) 配对列表。

    配对模式：
        auto   —— 目标图只有1张时用 single，否则用 random
        single —— 所有原图攻击向同一张目标图（取第1张）
        random —— 每张原图随机选择一张目标图
    """
    source_files = get_image_files(source_dir)
    target_files = load_target_paths(target_dir)

    if not source_files:
        raise ValueError(f"原图目录中未找到图像：{source_dir}")
    if not target_files:
        raise ValueError(f"未找到目标图像：{target_dir}")

    if match_mode == "auto":
        match_mode = "single" if len(target_files) == 1 else "random"
        print(f"\n[自动] {len(target_files)} 张目标 -> '{match_mode}' 模式")

    print(f"[信息] 配对模式：{match_mode.upper()} | "
          f"原图：{len(source_files)} 张 | 目标：{len(target_files)} 张")

    pairs = []
    if match_mode == "single":
        tgt_path = target_files[0]
        print(f"\n[单目标] 所有原图 -> {os.path.basename(tgt_path)}")
        for src in source_files:
            pairs.append((src, tgt_path))
    elif match_mode == "random":
        for src in tqdm(source_files, desc="构建配对"):
            pairs.append((src, random.choice(target_files)))
    else:
        raise ValueError(f"未知的 match_mode：{match_mode!r}")

    print(f"[完成] 共配对 {len(pairs)} 对")
    return pairs


# ===========================================================================
# fast 模式：仅使用 VAE 的 PGD
# ===========================================================================

def fast_vae_pgd_attack(
    vae,
    src_tensor: torch.Tensor,
    tgt_tensor: torch.Tensor,
    pgd_alpha: float,
    pgd_eps: float,
    pgd_steps: int,
    weight_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    仅使用 VAE 的 PGD，驱动原图潜在表示向目标潜在表示靠拢。

    参数
    ----
    vae         : AutoencoderKL（冻结参数）
    src_tensor  : (1,3,512,512)，范围 [-1,1]
    tgt_tensor  : (1,3,512,512)，范围 [-1,1]（固定参考目标）
    pgd_alpha   : [-1,1] 图像空间的步长
    pgd_eps     : [-1,1] 图像空间的 L-inf 上限
    weight_dtype: VAE 推理精度（fp16 / bf16 / float32）
    device      : cuda / cpu

    返回
    ----
    perturbed: (1,3,512,512) 对抗图像，范围 [-1,1]
    """
    vae.to(device, dtype=weight_dtype)
    vae.eval()
    vae.requires_grad_(False)

    src_orig = src_tensor.to(device, dtype=torch.float32)

    # 预计算固定目标潜在表示（使用 mean 而非 sample，避免随机噪声干扰梯度）
    with torch.no_grad():
        target_latent = (
            vae.encode(tgt_tensor.to(device, dtype=weight_dtype))
            .latent_dist.mean
            .to(dtype=torch.float32)
            * vae.config.scaling_factor
        )

    perturbed = src_orig.clone().detach()

    for step in range(pgd_steps):
        perturbed.requires_grad_(True)

        # 计算当前对抗样本的 VAE 潜在表示（使用 mean，梯度稳定）
        src_latent = (
            vae.encode(perturbed.to(dtype=weight_dtype))
            .latent_dist.mean
            .to(dtype=torch.float32)
            * vae.config.scaling_factor
        )

        loss = F.mse_loss(src_latent, target_latent)
        loss.backward()

        with torch.no_grad():
            adv = perturbed + pgd_alpha * perturbed.grad.sign()
            eta = torch.clamp(adv - src_orig, -pgd_eps, pgd_eps)
            perturbed = torch.clamp(src_orig + eta, -1.0, 1.0).detach_()

        if (step + 1) % 20 == 0 or step == pgd_steps - 1:
            print(f"    PGD [{step+1}/{pgd_steps}]  loss：{loss.item():.6f}")

    return perturbed


# ===========================================================================
# 主攻击执行器
# ===========================================================================

def run_attack(
    pairs,
    output_dir: str,
    sd_model: str,
    pgd_alpha: float,
    pgd_eps: float,
    pgd_steps: int,
    mode: str,
    mixed_precision: str,
    source_dir: str,
    target_dir: str,
    seed: int,
):
    os.makedirs(output_dir, exist_ok=True)

    if mode == "full":
        print("\n[模式] FULL ASPL —— 代理模型迭代训练 + PGD")
        _run_full_aspl(
            source_dir=source_dir, target_dir=target_dir,
            output_dir=output_dir, sd_model=sd_model,
            pgd_alpha=pgd_alpha, pgd_eps=pgd_eps, pgd_steps=pgd_steps,
            mixed_precision=mixed_precision, seed=seed,
        )
        match_info = {
            "attack_type": "aspl_full",
            "method": "ASPL_SurrogateTraining_PGD",
            "sd_model": sd_model,
            "parameters": {"pgd_alpha": pgd_alpha, "pgd_eps": pgd_eps,
                           "pgd_steps": pgd_steps, "mode": mode},
            "pairs": [{"source": os.path.basename(s), "target": os.path.basename(t)}
                      for s, t in pairs],
        }
        with open(os.path.join(output_dir, "match_info.json"), 'w', encoding='utf-8') as f:
            json.dump(match_info, f, indent=2, ensure_ascii=False)
        return match_info

    # ---- fast 模式 ----
    print("\n[模式] FAST —— 仅 VAE-PGD（无代理模型训练）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = (
        torch.float16  if mixed_precision == "fp16"
        else torch.bfloat16 if mixed_precision == "bf16"
        else torch.float32
    )

    print(f"\n{'=' * 60}")
    print(f"加载 SD VAE：{sd_model}")
    print(f"  设备：{device} | 精度：{weight_dtype}")
    print("=" * 60)

    t0 = time.time()
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(sd_model, subfolder="vae")
    vae.requires_grad_(False)
    vae_load_time = time.time() - t0
    print(f"  VAE 加载耗时：{vae_load_time:.2f}s")

    print(f"\n{'=' * 60}")
    print("攻击参数（Fast VAE-PGD）：")
    print(f"  PGD Alpha：{pgd_alpha:.6f} | Epsilon：{pgd_eps:.4f} | 步数：{pgd_steps}")
    print(f"  配对数量： {len(pairs)}")
    print(f"{'=' * 60}\n")

    match_info = {
        "attack_type": "aspl_fast_vae_pgd",
        "method": "VAE_PGD",
        "sd_model": sd_model,
        "parameters": {
            "pgd_alpha": pgd_alpha, "pgd_eps": pgd_eps,
            "pgd_steps": pgd_steps, "mixed_precision": mixed_precision, "mode": mode,
        },
        "pairs": [],
    }

    attack_start = time.time()
    per_image_times = []
    n_success = 0
    n_failed = 0
    json_copied = 0

    for i, (src_path, tgt_path) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {os.path.basename(src_path)}"
              f" -> {os.path.basename(tgt_path)}")
        img_start = time.time()
        try:
            src_tensor = pil_to_sd_tensor(Image.open(src_path).convert("RGB"))
            tgt_tensor = pil_to_sd_tensor(Image.open(tgt_path).convert("RGB"))

            adv_tensor = fast_vae_pgd_attack(
                vae=vae, src_tensor=src_tensor, tgt_tensor=tgt_tensor,
                pgd_alpha=pgd_alpha, pgd_eps=pgd_eps, pgd_steps=pgd_steps,
                weight_dtype=weight_dtype, device=device,
            )

            src_name = os.path.splitext(os.path.basename(src_path))[0]
            output_path = os.path.join(output_dir, f"{src_name}_adv.png")
            sd_tensor_to_pil(adv_tensor).save(output_path)
            print(f"  ✓ 已保存：{output_path}")
            img_elapsed = time.time() - img_start
            per_image_times.append(img_elapsed)
            n_success += 1

            # 如果原图有同名 JSON，一并复制到输出目录
            src_json = os.path.splitext(src_path)[0] + ".json"
            if os.path.exists(src_json):
                shutil.copy2(src_json, os.path.join(output_dir, f"{src_name}_adv.json"))
                json_copied += 1

            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "target": os.path.basename(tgt_path),
                "output": f"{src_name}_adv.png",
            })
        except Exception as e:
            print(f"  ✗ 失败：{e}")
            n_failed += 1
            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "target": os.path.basename(tgt_path),
                "output": None, "error": str(e),
            })

    attack_time = time.time() - attack_start
    match_info["timing"] = {
        "vae_load_time_sec": vae_load_time,
        "attack_time_sec": attack_time,
        "total_time_sec": vae_load_time + attack_time,
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
        script_name="AttackASPL.py",
        total=len(pairs),
        success=n_success,
        failed=n_failed,
        total_time=vae_load_time + attack_time,
        per_image_times=per_image_times,
    )

    success = sum(1 for p in match_info["pairs"] if p.get("output"))
    print(f"\n{'=' * 60}")
    print("攻击完成！")
    print(f"{'=' * 60}")
    print(f"  生成数量：     {success}/{len(pairs)} 张对抗图像")
    print(f"  JSON 复制：    {json_copied}/{len(pairs)}")
    print(f"  输出目录：     {output_dir}")
    print(f"  VAE 加载：     {vae_load_time:.2f}s")
    print(f"  攻击耗时：     {attack_time:.2f}s")
    print(f"  每张平均：     {attack_time / len(pairs):.2f}s")
    print(f"{'=' * 60}")
    return match_info


def _run_full_aspl(source_dir, target_dir, output_dir, sd_model,
                   pgd_alpha, pgd_eps, pgd_steps, mixed_precision, seed):
    """委托给原始 aspl.py 的 main()，按需创建临时目录。"""
    try:
        from attacks.aspl import parse_args as aspl_parse_args, main as aspl_main
    except ImportError:
        raise ImportError(
            "无法导入 aspl.py。请确保 Anti-DreamBooth/attacks/aspl.py 存在，"
            "且已安装 diffusers、accelerate、transformers 等依赖。"
        )

    source_files = get_image_files(source_dir)
    target_files = load_target_paths(target_dir)

    # 从第一张原图的同名 JSON 读取 instance_prompt
    instance_prompt = load_json_prompt(source_files[0]) if source_files else ""
    if not instance_prompt:
        instance_prompt = "a photo of sks person"
        print(f"[信息] 使用默认 instance_prompt：'{instance_prompt}'")
    else:
        print(f"[信息] instance_prompt：'{instance_prompt[:80]}'")

    import tempfile
    tmp_src = tempfile.mkdtemp(prefix="aspl_src_")
    tmp_adv = tempfile.mkdtemp(prefix="aspl_adv_")
    try:
        for p in source_files:
            shutil.copy2(p, tmp_src)
            shutil.copy2(p, tmp_adv)

        argv = [
            "--pretrained_model_name_or_path", sd_model,
            "--instance_data_dir_for_train", tmp_src,
            "--instance_data_dir_for_adversarial", tmp_adv,
            "--instance_prompt", instance_prompt,
            "--output_dir", output_dir,
            "--mixed_precision", mixed_precision,
            "--pgd_alpha", str(pgd_alpha),
            "--pgd_eps", str(pgd_eps),
            "--target_image_path", target_files[0],
        ]
        if seed is not None:
            argv += ["--seed", str(seed)]

        args = aspl_parse_args(argv)
        aspl_main(args)

        # 将 aspl checkpoint 输出文件重命名为 {name}_adv.png 格式
        ckpt_dirs = sorted(glob.glob(os.path.join(output_dir, "noise-ckpt", "*")))
        if ckpt_dirs:
            for adv_path in get_image_files(ckpt_dirs[-1]):
                base = os.path.basename(adv_path)
                parts = base.split("_noise_", 1)
                stem = os.path.splitext(parts[1] if len(parts) == 2 else base)[0]
                dst = os.path.join(output_dir, f"{stem}_adv.png")
                shutil.copy2(adv_path, dst)
                print(f"  ✓ 已保存：{dst}")
    finally:
        shutil.rmtree(tmp_src, ignore_errors=True)
        shutil.rmtree(tmp_adv, ignore_errors=True)


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
        description="AttackASPL - Anti-DreamBooth SD VAE 潜在空间 PGD 攻击"
    )

    parser.add_argument("--source_dir",      type=str, required=True,
                        help="原图目录（待保护的图像）")
    parser.add_argument("--target_dir",      type=str, required=True,
                        help="目标图目录（潜在空间对齐目标）")
    parser.add_argument("--output_dir",      type=str, default="./output_aspl",
                        help="输出目录（默认：./output_aspl）")
    parser.add_argument("--sd_model",        type=str,
                        default="stabilityai/stable-diffusion-2-1",
                        help="SD 模型名称或本地路径")
    parser.add_argument("--pgd_alpha",       type=float, default=1.0 / 255,
                        help="PGD 步长，[-1,1] 空间（默认：1/255）")
    parser.add_argument("--pgd_eps",         type=float, default=0.05,
                        help="L-inf 扰动上限，[-1,1] 空间（默认：0.05）")
    parser.add_argument("--pgd_steps",       type=int,   default=200,
                        help="PGD 迭代步数（默认：200）")
    parser.add_argument("--mode",            type=str,   default="fast",
                        choices=["fast", "full"],
                        help="攻击模式：fast（仅 VAE）/ full（完整 ASPL，默认：fast）")
    parser.add_argument("--match_mode",      type=str,   default="auto",
                        choices=["auto", "single", "random"],
                        help="配对模式：auto/single/random（默认：auto）")
    parser.add_argument("--seed",            type=int,   default=DEFAULT_RANDOM_SEED,
                        help=f"随机种子（默认：{DEFAULT_RANDOM_SEED}）")
    parser.add_argument("--mixed_precision", type=str,   default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="混合精度：no/fp16/bf16（默认：fp16）")

    args = parser.parse_args()
    seed_everything(args.seed)

    print(f"\n{'=' * 60}")
    print("AttackASPL - Anti-DreamBooth SD VAE 潜在空间 PGD 攻击")
    print("=" * 60)
    print(f"  原图目录：     {args.source_dir}")
    print(f"  目标目录：     {args.target_dir}")
    print(f"  输出目录：     {args.output_dir}")
    print(f"  攻击模式：     {args.mode}")
    print(f"  SD 模型：      {args.sd_model}")
    print(f"  PGD Alpha：    {args.pgd_alpha}")
    print(f"  PGD Epsilon：  {args.pgd_eps}")
    print(f"  PGD 步数：     {args.pgd_steps}")
    print(f"  混合精度：     {args.mixed_precision}")
    print(f"  配对模式：     {args.match_mode}")
    print(f"  随机种子：     {args.seed}")
    print("=" * 60)

    pairs = match_source_target_pairs(
        args.source_dir, args.target_dir, match_mode=args.match_mode)

    run_attack(
        pairs=pairs,
        output_dir=args.output_dir,
        sd_model=args.sd_model,
        pgd_alpha=args.pgd_alpha,
        pgd_eps=args.pgd_eps,
        pgd_steps=args.pgd_steps,
        mode=args.mode,
        mixed_precision=args.mixed_precision,
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
