"""
AttackMI.py - MI_CommonWeakness 迁移攻击（图像级特征对齐）

攻击策略：
    - 以原图（source）作为待扰动的图像
    - 以目标图（target）作为攻击目标（特征对齐）
    - 通过 MI_CommonWeakness 驱动原图的对抗特征向目标图特征靠拢
    - 代理模型：ResNet-18 / ResNet-50 / ViT-B-16 三个特征提取器（移除分类头，
      通过线性投影统一到相同维度）

关键设计说明：
    MI_CommonWeakness 调用 criterion 的方式如下：
        第1步：criterion(sum_i model_i(x), y)   —— 汇总 logit
        第2步：criterion(model_i(aug_x), y)     —— 每个模型单独 logit
    两步必须兼容同一个 y，因此：
        1. 将三个模型的输出统一投影到 512 维空间。
        2. 第1步中汇总 logit 形状为 (B, 512)；y 也为 (B, 512)，即目标特征之和。
        3. 第2步每个模型输出形状为 (B, 512)；MI_CommonWeakness 对所有模型使用同一个 y，
           因此 y 仍设为目标特征之和——单个模型输出对该 sum 的 MSE 仍是有效代理损失。
    FeatureAlignCriterion 返回 -MSE(logit, target_feat)，MI_CommonWeakness 最小化该值
    （targeted_attack=True 会让攻击器增大 criterion，即最大化 -MSE，即最小化 MSE，
    即把对抗特征拉向目标特征）。

参考：
    AdversarialAttacks/attacks/AdversarialInput/CommonWeakness.py
    AdversarialAttacks/attacks/AdversarialInput/AdversarialInputBase.py

基础用法：
    python AttackMI.py --source_dir ../Data_source/sample_source \\
                       --target_dir ../Data_source/sample_target \\
                       --output_dir ./output_mi_attack

自定义参数：
    python AttackMI.py --source_dir ./source --target_dir ./target \\
                       --output_dir ./output \\
                       --epsilon 16 --step_size 1 --steps 300 \\
                       --inner_step_size 250 --reverse_step_size 1 \\
                       --match_mode random --seed 2023

单目标模式（所有原图攻击向同一张目标图）：
    python AttackMI.py --source_dir ./source --target_dir ./target \\
                       --output_dir ./output --match_mode single
"""

import torch
import torch.nn as nn
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
from torchvision import transforms, models

# ---------------------------------------------------------------------------
# 路径配置：将 AdversarialAttacks 根目录加入 sys.path，使 attacks 包可导入
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from attacks.AdversarialInput import MI_CommonWeakness

DEFAULT_RANDOM_SEED = 2023
UNIFIED_DIM = 512   # 所有特征提取器统一投影到该维度


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
# 特征对齐损失函数
# ===========================================================================

class FeatureAlignCriterion(nn.Module):
    """
    自定义特征对齐损失函数。

    MI_CommonWeakness 的调用方式：
        第1步：criterion(sum_i model_i(x), y)    -> y 必须与输出兼容
        第2步：criterion(model_i(aug_x), y)      -> 同一个 y，逐模型调用

    在每次攻击前，通过 set_target_feature() 注册目标特征之和。
    criterion 返回 -MSE(logit, _target_feature)，使得：
        maximize criterion (targeted_attack=True)
        == minimize MSE
        == 将对抗特征拉向目标特征
    """

    def __init__(self):
        super().__init__()
        self._target_feature = None

    def set_target_feature(self, target_feature: torch.Tensor):
        """注册目标特征（攻击前调用一次）。"""
        self._target_feature = target_feature.detach()

    def forward(self, logit: torch.Tensor, y=None):
        if self._target_feature is None:
            raise RuntimeError("攻击前请先调用 set_target_feature()")
        return -F.mse_loss(logit, self._target_feature.to(logit.device))


# ===========================================================================
# 代理特征提取模型（输出统一维度 UNIFIED_DIM 的向量）
# ===========================================================================

class ResNet18FeatureExtractor(nn.Module):
    """ResNet-18 主干 + 线性投影到 UNIFIED_DIM。"""

    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])   # 输出 (B,512,1,1)
        self.proj = nn.Linear(512, UNIFIED_DIM, bias=False)
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(self._normalize(x)).flatten(1)   # (B,512)
        return self.proj(feat)                                 # (B,UNIFIED_DIM)


class ResNet50FeatureExtractor(nn.Module):
    """ResNet-50 主干 + 线性投影到 UNIFIED_DIM。"""

    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-1])   # 输出 (B,2048,1,1)
        self.proj = nn.Linear(2048, UNIFIED_DIM, bias=False)
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(self._normalize(x)).flatten(1)   # (B,2048)
        return self.proj(feat)                                 # (B,UNIFIED_DIM)


class ViTB16FeatureExtractor(nn.Module):
    """ViT-B/16 替换分类头为 Identity + 线性投影到 UNIFIED_DIM。"""

    def __init__(self):
        super().__init__()
        base = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        base.heads = nn.Identity()                             # 输出 (B,768)
        self.model = base
        self.proj = nn.Linear(768, UNIFIED_DIM, bias=False)
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model(self._normalize(x))                  # (B,768)
        return self.proj(feat)                                 # (B,UNIFIED_DIM)


# ===========================================================================
# 图像 I/O 辅助函数
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


def save_image_tensor(tensor: torch.Tensor, path: str):
    """将 [0,1] 浮点张量 (C,H,W) 保存为 PNG 文件。"""
    arr = (tensor.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    Image.fromarray(arr.permute(1, 2, 0).numpy()).save(path)


# ===========================================================================
# 原图-目标图配对逻辑
# ===========================================================================

def match_source_target_pairs(source_dir: str, target_dir: str,
                               match_mode: str = "auto", input_res: int = 224):
    """
    构建 (src_path, tgt_path, src_tensor, tgt_tensor) 配对列表。

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

    print(f"[信息] 配对模式：{match_mode.upper()} | 原图：{len(source_files)} 张 | "
          f"目标：{len(target_files)} 张")

    resizer   = transforms.Resize((input_res, input_res))
    to_tensor = transforms.ToTensor()
    pairs = []

    if match_mode == "single":
        tgt_path = target_files[0]
        tgt_img  = resizer(to_tensor(Image.open(tgt_path).convert("RGB")))
        print(f"\n[单目标] 所有原图 -> {os.path.basename(tgt_path)}")
        for src_path in tqdm(source_files, desc="加载原图"):
            try:
                src_img = resizer(to_tensor(Image.open(src_path).convert("RGB")))
                pairs.append((src_path, tgt_path, src_img, tgt_img.clone()))
            except Exception as e:
                print(f"  ✗ {os.path.basename(src_path)}：{e}")

    elif match_mode == "random":
        print(f"\n[随机] 从 {len(target_files)} 张目标中随机配对 {len(source_files)} 张原图 ...")
        for src_path in tqdm(source_files, desc="加载图像对"):
            tgt_path = random.choice(target_files)
            try:
                src_img = resizer(to_tensor(Image.open(src_path).convert("RGB")))
                tgt_img = resizer(to_tensor(Image.open(tgt_path).convert("RGB")))
                pairs.append((src_path, tgt_path, src_img, tgt_img))
            except Exception as e:
                print(f"  ✗ 配对失败：{e}")
    else:
        raise ValueError(f"未知的 match_mode：{match_mode!r}")

    if not pairs:
        raise ValueError("未找到有效的原图-目标图对！")
    print(f"[完成] 共配对 {len(pairs)} 对")
    return pairs


# ===========================================================================
# 核心攻击执行器
# ===========================================================================

def run_attack(
    pairs,
    output_dir: str,
    epsilon: float = 16,
    step_size: float = 1,
    total_step: int = 300,
    inner_step_size: float = 250,
    reverse_step_size: float = 1,
    input_res: int = 224,
):
    """
    对所有原图-目标图对执行 MI_CommonWeakness 特征对齐攻击。

    MI_CommonWeakness.attack(x, y) 执行流程：
        第1步（反向扰动）：logit = sum_i model_i(x)；loss = criterion(logit, y)
                          x += reverse_step_size * grad.sign()  [定向攻击]
        第2步（内层 MI）：对每个模型：
                          loss = criterion(model_i(aug_x), y)
                          内层动量 = mu * momentum - grad / ||grad||
                          x += inner_step_size * momentum
        外层更新：x = x + ksi * outer_momentum.sign()

    y = 目标特征之和（预计算，通过 criterion 注册）。
    targeted_attack=True 表示攻击器添加（而非减去）梯度，
    结合 -MSE 损失，将对抗样本拉向目标。
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print("初始化代理特征提取器 ...")
    print(f"  设备：{device}")
    print("=" * 60)

    t0 = time.time()
    resnet18 = ResNet18FeatureExtractor().eval().requires_grad_(False)
    print("  ✓ ResNet-18 特征提取器已加载")
    resnet50 = ResNet50FeatureExtractor().eval().requires_grad_(False)
    print("  ✓ ResNet-50 特征提取器已加载")
    vit = ViTB16FeatureExtractor().eval().requires_grad_(False)
    print("  ✓ ViT-B/16 特征提取器已加载")
    model_list = [resnet18, resnet50, vit]
    model_load_time = time.time() - t0
    print(f"  模型加载耗时：{model_load_time:.2f}s")

    criterion = FeatureAlignCriterion()

    attacker = MI_CommonWeakness(
        model_list,
        epsilon=epsilon / 255.0,
        step_size=step_size / 255.0,
        total_step=total_step,
        criterion=criterion,
        targeted_attack=True,   # 定向攻击：将对抗特征拉向目标特征
        mu=1,
        inner_step_size=inner_step_size,
        reverse_step_size=reverse_step_size / 255.0,
    )

    print(f"\n{'=' * 60}")
    print("攻击参数：")
    print(f"  方法：           MI_CommonWeakness")
    print(f"  代理模型：       ResNet-18 / ResNet-50 / ViT-B-16 (→{UNIFIED_DIM}维)")
    print(f"  Epsilon：        {epsilon}/255 = {epsilon/255:.4f}")
    print(f"  步长：           {step_size}/255 = {step_size/255:.4f}")
    print(f"  总步数：         {total_step}")
    print(f"  内层步长：       {inner_step_size}")
    print(f"  反向步长：       {reverse_step_size}/255")
    print(f"  输入分辨率：     {input_res}")
    print(f"  配对数量：       {len(pairs)}")
    print(f"{'=' * 60}\n")

    match_info = {
        "attack_type": "mi_transfer_feature_alignment",
        "method": "MI_CommonWeakness",
        "models": ["ResNet-18", "ResNet-50", "ViT-B-16"],
        "parameters": {
            "epsilon": epsilon,
            "step_size": step_size,
            "total_step": total_step,
            "inner_step_size": inner_step_size,
            "reverse_step_size": reverse_step_size,
            "input_res": input_res,
            "unified_feature_dim": UNIFIED_DIM,
        },
        "pairs": [],
    }

    attack_start = time.time()
    per_image_times = []
    n_success = 0
    n_failed = 0
    json_copied = 0

    for i, (src_path, tgt_path, src_tensor, tgt_tensor) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {os.path.basename(src_path)}"
              f" -> {os.path.basename(tgt_path)}")
        img_start = time.time()

        src_img = src_tensor.unsqueeze(0).to(device)
        tgt_img = tgt_tensor.unsqueeze(0).to(device)

        # 预计算目标特征之和（对应 MI_CommonWeakness 第1步的求和操作）
        with torch.no_grad():
            target_feat_sum = None
            for model in model_list:
                feat = model(tgt_img).to(device)
                target_feat_sum = feat if target_feat_sum is None else target_feat_sum + feat

        criterion.set_target_feature(target_feat_sum)

        adv_img = attacker(src_img, y=target_feat_sum)

        src_name = os.path.splitext(os.path.basename(src_path))[0]
        output_path = os.path.join(output_dir, f"{src_name}_adv.png")
        save_image_tensor(adv_img.squeeze(0), output_path)
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
        script_name="AttackMI.py",
        total=len(pairs),
        success=n_success,
        failed=n_failed,
        total_time=model_load_time + attack_time,
        per_image_times=per_image_times,
    )

    print(f"\n{'=' * 60}")
    print("攻击完成！")
    print(f"{'=' * 60}")
    print(f"  生成数量：      {len(pairs)} 张对抗图像")
    print(f"  JSON 复制：     {json_copied}/{len(pairs)}")
    print(f"  输出目录：      {output_dir}")
    print(f"  模型加载：      {model_load_time:.2f}s")
    print(f"  攻击耗时：      {attack_time:.2f}s")
    print(f"  总耗时：        {model_load_time + attack_time:.2f}s")
    print(f"  每张平均：      {attack_time / len(pairs):.2f}s")
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
        description="AttackMI - MI_CommonWeakness 迁移攻击（特征对齐）"
    )

    parser.add_argument("--source_dir",        type=str, required=True,
                        help="原图目录（待扰动的图像）")
    parser.add_argument("--target_dir",        type=str, required=True,
                        help="目标图目录（特征对齐目标）")
    parser.add_argument("--output_dir",        type=str, default="./output_mi_attack",
                        help="输出目录（默认：./output_mi_attack）")
    parser.add_argument("--epsilon",           type=float, default=16,
                        help="L-inf 扰动上限（0-255 尺度，默认：16）")
    parser.add_argument("--step_size",         type=float, default=1,
                        help="PGD 步长（0-255 尺度，默认：1）")
    parser.add_argument("--steps",             type=int,   default=300,
                        help="攻击总迭代步数（默认：300）")
    parser.add_argument("--inner_step_size",   type=float, default=250,
                        help="MI 内层步长（默认：250）")
    parser.add_argument("--reverse_step_size", type=float, default=1,
                        help="反向扰动步长（0-255 尺度，默认：1）")
    parser.add_argument("--match_mode",        type=str,   default="auto",
                        choices=["auto", "single", "random"],
                        help="配对模式：auto/single/random（默认：auto）")
    parser.add_argument("--input_res",         type=int,   default=224,
                        help="输入分辨率（默认：224）")
    parser.add_argument("--seed",              type=int,   default=DEFAULT_RANDOM_SEED,
                        help=f"随机种子（默认：{DEFAULT_RANDOM_SEED}）")

    args = parser.parse_args()
    seed_everything(args.seed)

    print(f"\n{'=' * 60}")
    print("AttackMI - MI_CommonWeakness 迁移攻击")
    print("=" * 60)
    print(f"  原图目录：       {args.source_dir}")
    print(f"  目标目录：       {args.target_dir}")
    print(f"  输出目录：       {args.output_dir}")
    print(f"  配对模式：       {args.match_mode}")
    print(f"  Epsilon：        {args.epsilon}")
    print(f"  步长：           {args.step_size}")
    print(f"  总步数：         {args.steps}")
    print(f"  内层步长：       {args.inner_step_size}")
    print(f"  反向步长：       {args.reverse_step_size}")
    print(f"  输入分辨率：     {args.input_res}")
    print(f"  随机种子：       {args.seed}")
    print("=" * 60)

    pairs = match_source_target_pairs(
        args.source_dir, args.target_dir,
        match_mode=args.match_mode,
        input_res=args.input_res,
    )

    run_attack(
        pairs,
        args.output_dir,
        epsilon=args.epsilon,
        step_size=args.step_size,
        total_step=args.steps,
        inner_step_size=args.inner_step_size,
        reverse_step_size=args.reverse_step_size,
        input_res=args.input_res,
    )


if __name__ == "__main__":
    main()
