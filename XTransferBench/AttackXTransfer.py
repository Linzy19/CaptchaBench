"""
AttackXTransfer.py - 基于 CLIP 的对抗攻击（XTransferBench 框架）

攻击策略：
    - 以原图（source）作为待扰动的输入
    - 以目标 JSON 注释中的文本（prompt + label）作为攻击目标
    - 两种模式：
        * per_image（默认）：每张图独立 PGD，各有自己的 delta
          - targeted（定向）：将 CLIP 图像特征拉向目标文本嵌入
          - untargeted（非定向）：将 CLIP 图像特征推离原始图像嵌入
        * uap：在所有原图上训练一个通用对抗扰动（UAP），
          然后将同一个 delta 应用到每张图上

关于 PGDLInfinity 的复用说明：
    XTransferBench/xtransfer/attacks/attack_generation.py 中的 PGDLInfinity 类
    在 __init__ 中调用了 misc.broadcast_tensor()（分布式训练原语），
    且维护一个共享 delta 用于 UAP 训练，不适合独立的逐图 PGD。
    因此本脚本实现了独立的 pgd_clip_per_image() 和 train_uap() 函数，
    在算法上与 XTransferBench 保持一致，同时避免了分布式依赖。

参考：
    XTransferBench/xtransfer/attacks/attack_generation.py  (PGDLInfinity 设计)
    XTransferBench/xtransfer/generate_universal_perturbation.py  (UAP 训练循环)
    Attack-Bard/AttackVLM.py  (输出格式参考)

逐图定向攻击（默认）：
    python AttackXTransfer.py \\
        --source_dir ../Data_source/sample_source \\
        --target_dir ../Data_source/sample_target \\
        --output_dir ./output_xtransfer \\
        --mode per_image --attack_type targeted \\
        --epsilon 16 --step_size 1 --steps 300

UAP 模式：
    python AttackXTransfer.py \\
        --source_dir ../Data_source/sample_source \\
        --target_dir ../Data_source/sample_target \\
        --output_dir ./output_xtransfer_uap \\
        --mode uap --attack_type targeted \\
        --epsilon 16 --step_size 1 --steps 300 --uap_epochs 5

非定向攻击：
    python AttackXTransfer.py \\
        --source_dir ./source --target_dir ./target \\
        --output_dir ./output \\
        --attack_type untargeted --epsilon 16 --steps 300
"""

import torch
import torch.nn.functional as F
import os
import sys
import json
import glob
import time
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# ---------------------------------------------------------------------------
# 路径配置：将 XTransferBench 根目录加入 sys.path，使 xtransfer 包可导入
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import open_clip

# 设备选择（与 generate_universal_perturbation.py 保持一致）
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

DEFAULT_SEED = 7


# ===========================================================================
# 工具函数
# ===========================================================================

def seed_everything(seed: int = DEFAULT_SEED):
    """固定所有随机种子，确保实验可复现。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_image_files(directory: str):
    """递归扫描目录，返回所有支持格式的图像路径（已排序）。"""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
                  "*.JPEG", "*.JPG", "*.PNG"]
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def read_target_text_from_json(json_path: str):
    """从 JSON 注释文件提取 'prompt Label: label' 字符串，失败时返回 None。"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  [警告] 无法读取 JSON {json_path}：{exc}")
        return None
    parts = []
    prompt = data.get("prompt", "").strip()
    if prompt:
        parts.append(prompt)
    labels = [a.get("label", "").strip()
              for a in data.get("annotations", [])
              if a.get("label", "").strip()]
    if labels:
        parts.append("Label: " + ", ".join(labels))
    return " ".join(parts) if parts else None


# ===========================================================================
# 数据加载与配对
# ===========================================================================

def build_pairs(source_dir: str, target_dir: str,
                image_size: int, match_mode: str = "auto"):
    """
    构建 (source_path, target_text, source_tensor) 三元组列表。

    配对模式（--match_mode）：
        auto  : 先按文件名匹配，失败的回退到按顺序位置匹配
        name  : 严格文件名匹配（无法匹配时跳过）
        order : 按排序后位置对齐（zip 顺序）
    """
    source_images = get_image_files(source_dir)
    target_jsons  = sorted(glob.glob(os.path.join(target_dir, "*.json")))

    if not source_images:
        raise ValueError(f"原图目录中未找到图像文件：{source_dir}")
    if not target_jsons:
        raise ValueError(f"目标目录中未找到 JSON 文件：{target_dir}")

    # 建立文件名 -> JSON 路径的映射，用于名称匹配
    target_map = {os.path.splitext(os.path.basename(p))[0]: p
                  for p in target_jsons}

    resizer   = transforms.Resize((image_size, image_size))
    to_tensor = transforms.ToTensor()

    pairs = []

    if match_mode == "order":
        # 按位置顺序逐一配对
        for src_path, tgt_json in zip(source_images, target_jsons):
            text = read_target_text_from_json(tgt_json)
            if not text:
                continue
            try:
                tensor = resizer(to_tensor(Image.open(src_path).convert("RGB")))
                pairs.append((src_path, text, tensor))
            except Exception as e:
                print(f"  [警告] {src_path}：{e}")
    else:
        # auto / name：先尝试文件名匹配，auto 模式下失败的回退到顺序匹配
        unmatched = []
        for src_path in source_images:
            stem = os.path.splitext(os.path.basename(src_path))[0]
            tgt_json = target_map.get(stem)
            if tgt_json is None:
                if match_mode == "name":
                    print(f"  [警告] 未找到 '{stem}' 的名称匹配 JSON，已跳过")
                    continue
                unmatched.append(src_path)
                continue
            text = read_target_text_from_json(tgt_json)
            if not text:
                unmatched.append(src_path)
                continue
            try:
                tensor = resizer(to_tensor(Image.open(src_path).convert("RGB")))
                pairs.append((src_path, text, tensor))
            except Exception as e:
                print(f"  [警告] {src_path}：{e}")

        # 对未匹配的原图按顺序回退（仅 auto 模式）
        if unmatched and match_mode == "auto":
            used = {target_map.get(
                        os.path.splitext(os.path.basename(p))[0])
                    for p, _, _ in pairs}
            remaining = [j for j in target_jsons if j not in used]
            for src_path, tgt_json in zip(unmatched, remaining):
                text = read_target_text_from_json(tgt_json)
                if not text:
                    continue
                try:
                    tensor = resizer(to_tensor(Image.open(src_path).convert("RGB")))
                    pairs.append((src_path, text, tensor))
                except Exception as e:
                    print(f"  [警告] {src_path}：{e}")

    if not pairs:
        raise ValueError("未构建出有效的原图-目标对。")
    print(f"[数据] 共构建 {len(pairs)} 对原图-目标")
    return pairs


# ===========================================================================
# CLIP 模型加载
# ===========================================================================

def load_clip_model(model_name: str = "ViT-B-32",
                    pretrained: str = "laion2b_s34b_b79k"):
    """加载 OpenCLIP 模型，返回 (model, tokenizer, normalize_transform)。"""
    print(f"[模型] 加载 OpenCLIP {model_name} ({pretrained}) ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    # CLIP 归一化变换是预处理管线的最后一步
    normalize  = preprocess.transforms[-1]

    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"  ✓ 模型已加载到 {DEVICE}")
    return model, tokenizer, normalize


# ===========================================================================
# 逐图 PGD 攻击
# ===========================================================================

def pgd_clip_per_image(
    model, image, normalize, tokenizer,
    target_text, attack_type="targeted",
    epsilon=16/255, step_size=1/255, steps=300, log_interval=50,
):
    """
    独立逐图 PGD 攻击，使用 CLIP 作为代理模型。

    targeted（定向）  ：最大化 cos_sim(encode_image(adv), encode_text(target_text))
    untargeted（非定向）：最小化 cos_sim(encode_image(adv), encode_image(clean))

    注意：encode_text/encode_image 输出通过 F.normalize 归一化，
    兼容所有版本的 open_clip（不使用 normalize=True 参数）。
    """
    image = image.to(DEVICE)

    with torch.no_grad():
        if attack_type == "targeted":
            if not target_text:
                raise ValueError("定向攻击需要提供 target_text")
            tokens    = tokenizer([target_text]).to(DEVICE)
            target_emb = F.normalize(model.encode_text(tokens).float(), dim=-1)
        else:
            # 非定向：以原图特征作为参考，推离
            target_emb = F.normalize(model.encode_image(normalize(image)).float(), dim=-1)

    # 随机初始化扰动 delta
    delta = torch.empty_like(image).uniform_(-epsilon, epsilon).to(DEVICE)
    delta.requires_grad_(True)

    for step in range(steps):
        x_adv   = torch.clamp(image + delta, 0.0, 1.0)
        adv_emb = F.normalize(model.encode_image(normalize(x_adv)).float(), dim=-1)
        cos_sim = (adv_emb * target_emb).sum(dim=1)

        # 定向：最大化余弦相似度（负号使其成为最小化目标）；非定向：最小化
        loss = -cos_sim.mean() if attack_type == "targeted" else cos_sim.mean()
        loss.backward()

        with torch.no_grad():
            delta_new = delta - step_size * delta.grad.sign()
            delta_new = torch.clamp(delta_new, -epsilon, epsilon)
            delta.copy_(delta_new)
            delta.grad.zero_()

        if log_interval > 0 and (step + 1) % log_interval == 0:
            print(f"      步骤 [{step+1:>4}/{steps}] "
                  f"loss={loss.item():.5f}  cos_sim={cos_sim.mean().item():.5f}")

    with torch.no_grad():
        adv_image = torch.clamp(image + delta, 0.0, 1.0)
    return adv_image.detach()


# ===========================================================================
# UAP 训练
# ===========================================================================

def train_uap(
    model, all_pairs, normalize, tokenizer,
    attack_type="targeted",
    epsilon=16/255, step_size=1/255,
    steps_per_epoch=300, epochs=5,
    image_size=224, log_interval=50,
):
    """
    在所有原图上训练通用对抗扰动（UAP）。
    外层循环与 generate_universal_perturbation.py 保持一致。
    """
    print(f"\n[UAP] 开始训练  图像数={len(all_pairs)}, epochs={epochs}, "
          f"steps/epoch={steps_per_epoch}, epsilon={epsilon*255:.1f}/255")

    # 初始化通用 delta
    delta = torch.empty(1, 3, image_size, image_size,
                        device=DEVICE).uniform_(-epsilon, epsilon)
    delta.requires_grad_(True)

    # 预计算所有图像的目标嵌入
    target_embs = []
    with torch.no_grad():
        for src_path, target_text, tensor in all_pairs:
            img = tensor.unsqueeze(0).to(DEVICE) if tensor.dim() == 3 else tensor.to(DEVICE)
            if attack_type == "targeted":
                tokens = tokenizer([target_text]).to(DEVICE)
                emb = F.normalize(model.encode_text(tokens).float(), dim=-1)
            else:
                emb = F.normalize(model.encode_image(normalize(img)).float(), dim=-1)
            target_embs.append(emb)

    for epoch in range(epochs):
        epoch_losses = []
        for step in range(steps_per_epoch):
            total_loss = torch.tensor(0.0, device=DEVICE)

            # 累加所有图像的损失
            for i, (src_path, _, tensor) in enumerate(all_pairs):
                img = tensor.unsqueeze(0).to(DEVICE) if tensor.dim() == 3 else tensor.to(DEVICE)
                d = delta
                # 若 delta 尺寸与图像不匹配则双线性插值
                if d.shape[2] != img.shape[2] or d.shape[3] != img.shape[3]:
                    d = F.interpolate(delta, (img.shape[2], img.shape[3]),
                                      mode="bilinear", align_corners=False)
                x_adv   = torch.clamp(img + d, 0.0, 1.0)
                adv_emb = F.normalize(model.encode_image(normalize(x_adv)).float(), dim=-1)
                cos_sim = (adv_emb * target_embs[i]).sum(dim=1)
                loss_i  = -cos_sim.mean() if attack_type == "targeted" else cos_sim.mean()
                total_loss = total_loss + loss_i

            avg_loss = total_loss / len(all_pairs)
            avg_loss.backward()

            with torch.no_grad():
                delta_new = delta - step_size * delta.grad.sign()
                delta_new = torch.clamp(delta_new, -epsilon, epsilon)
                delta.copy_(delta_new)
                delta.grad.zero_()

            epoch_losses.append(avg_loss.item())
            if log_interval > 0 and (step + 1) % log_interval == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] 步骤 [{step+1}/{steps_per_epoch}] "
                      f"avg_loss={avg_loss.item():.5f}")

        print(f"  [Epoch {epoch+1}/{epochs}] mean_loss={np.mean(epoch_losses):.5f}")

    print("  ✓ UAP 训练完成")
    return delta.detach()


# ===========================================================================
# 攻击执行器
# ===========================================================================

def run_per_image_attack(pairs, output_dir, model, tokenizer, normalize,
                         attack_type, epsilon, step_size, steps, log_interval):
    """逐图 PGD 攻击：对每张原图独立计算 delta 并保存对抗图像。"""
    os.makedirs(output_dir, exist_ok=True)

    match_info = {
        "attack_type": "xtransfer_clip_attack",
        "method": f"PGD_CLIP_{'定向' if attack_type=='targeted' else '非定向'}",
        "parameters": {
            "mode": "per_image", "attack_type": attack_type,
            "epsilon_255": epsilon * 255, "step_size_255": step_size * 255,
            "steps": steps,
        },
        "pairs": [],
        "timing": {},
    }

    total_start = time.time()
    per_image_times = []
    n_ok = n_fail = 0

    for idx, (src_path, target_text, src_tensor) in enumerate(pairs):
        src_name = os.path.splitext(os.path.basename(src_path))[0]
        out_fname = f"{src_name}_adv.png"
        out_path  = os.path.join(output_dir, out_fname)

        print(f"\n  [{idx+1}/{len(pairs)}] {os.path.basename(src_path)}")
        print(f"    目标：{target_text[:80]}{'...' if len(target_text)>80 else ''}")

        img_start = time.time()
        try:
            img = src_tensor.unsqueeze(0) if src_tensor.dim() == 3 else src_tensor
            adv = pgd_clip_per_image(
                model=model, image=img, normalize=normalize, tokenizer=tokenizer,
                target_text=target_text, attack_type=attack_type,
                epsilon=epsilon, step_size=step_size, steps=steps,
                log_interval=log_interval,
            )
            transforms.ToPILImage()(adv.squeeze(0).cpu().clamp(0, 1)).save(out_path)
            print(f"    ✓ 已保存：{out_path}  ({time.time()-img_start:.1f}s)")
            n_ok += 1
            per_image_times.append(time.time() - img_start)
            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "target_text": target_text[:300], "output": out_fname,
                "time_sec": time.time() - img_start,
            })
        except Exception as exc:
            print(f"    ✗ 失败：{exc}")
            n_fail += 1
            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "target_text": target_text[:300], "output": None, "error": str(exc),
            })

    elapsed = time.time() - total_start
    match_info["timing"] = {
        "total_time_sec": elapsed, "total_images": len(pairs),
        "success": n_ok, "failed": n_fail,
        "avg_time_per_image_sec": elapsed / len(pairs) if pairs else 0,
        "per_image_times_sec": per_image_times,
        "min_time_sec": min(per_image_times) if per_image_times else 0,
        "max_time_sec": max(per_image_times) if per_image_times else 0,
    }
    return match_info


def run_uap_attack(pairs, output_dir, model, tokenizer, normalize,
                   attack_type, epsilon, step_size, steps, uap_epochs,
                   image_size, log_interval):
    """UAP 模式：训练通用扰动后批量应用到所有原图。"""
    os.makedirs(output_dir, exist_ok=True)

    match_info = {
        "attack_type": "xtransfer_clip_attack",
        "method": f"UAP_CLIP_{'定向' if attack_type=='targeted' else '非定向'}",
        "parameters": {
            "mode": "uap", "attack_type": attack_type,
            "epsilon_255": epsilon * 255, "step_size_255": step_size * 255,
            "steps_per_epoch": steps, "uap_epochs": uap_epochs, "image_size": image_size,
        },
        "pairs": [], "timing": {},
    }

    total_start = time.time()
    uap_apply_times = []
    train_start = time.time()
    uap_delta = train_uap(
        model=model, all_pairs=pairs, normalize=normalize, tokenizer=tokenizer,
        attack_type=attack_type, epsilon=epsilon, step_size=step_size,
        steps_per_epoch=steps, epochs=uap_epochs, image_size=image_size,
        log_interval=log_interval,
    )
    train_time = time.time() - train_start

    # 保存通用扰动张量
    torch.save(uap_delta.cpu(), os.path.join(output_dir, "uap_delta.pth"))
    print(f"  ✓ UAP delta 已保存至 {output_dir}/uap_delta.pth")

    n_ok = n_fail = 0
    for idx, (src_path, target_text, src_tensor) in enumerate(pairs):
        apply_start = time.time()
        src_name = os.path.splitext(os.path.basename(src_path))[0]
        out_fname = f"{src_name}_uap_adv.png"
        out_path  = os.path.join(output_dir, out_fname)
        print(f"  [{idx+1}/{len(pairs)}] {os.path.basename(src_path)}")
        try:
            img = src_tensor.unsqueeze(0).to(DEVICE) if src_tensor.dim() == 3 else src_tensor.to(DEVICE)
            d = uap_delta.to(DEVICE)
            if d.shape[2] != img.shape[2] or d.shape[3] != img.shape[3]:
                d = F.interpolate(d, (img.shape[2], img.shape[3]),
                                  mode="bilinear", align_corners=False)
            d = torch.clamp(d, -epsilon, epsilon)
            adv = torch.clamp(img + d, 0.0, 1.0)
            transforms.ToPILImage()(adv.squeeze(0).cpu()).save(out_path)
            print(f"    ✓ 已保存：{out_path}")
            n_ok += 1
            uap_apply_times.append(time.time() - apply_start)
            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "target_text": target_text[:300], "output": out_fname,
            })
        except Exception as exc:
            print(f"    ✗ 失败：{exc}")
            n_fail += 1
            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "target_text": target_text[:300], "output": None, "error": str(exc),
            })

    elapsed = time.time() - total_start
    match_info["timing"] = {
        "total_time_sec": elapsed, "uap_train_time_sec": train_time,
        "apply_time_sec": elapsed - train_time,
        "total_images": len(pairs), "success": n_ok, "failed": n_fail,
        "per_image_apply_times_sec": uap_apply_times,
        "min_apply_time_sec": min(uap_apply_times) if uap_apply_times else 0,
        "max_apply_time_sec": max(uap_apply_times) if uap_apply_times else 0,
    }
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
        description="AttackXTransfer - 基于 CLIP 的对抗攻击（XTransferBench）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--source_dir",       type=str, required=True,
                        help="原图目录（待扰动的图像）")
    parser.add_argument("--target_dir",       type=str, required=True,
                        help="目标目录（包含目标文本 JSON 文件）")
    parser.add_argument("--output_dir",       type=str, default="./output_xtransfer",
                        help="输出目录（默认：./output_xtransfer）")
    parser.add_argument("--mode",             type=str, default="per_image",
                        choices=["per_image", "uap"],
                        help="攻击模式：per_image（逐图）/ uap（通用扰动，默认：per_image）")
    parser.add_argument("--attack_type",      type=str, default="targeted",
                        choices=["targeted", "untargeted"],
                        help="攻击类型：targeted（定向）/ untargeted（非定向，默认：targeted）")
    parser.add_argument("--clip_model",       type=str, default="ViT-B-32",
                        help="OpenCLIP 模型名称（默认：ViT-B-32）")
    parser.add_argument("--clip_pretrained",  type=str, default="laion2b_s34b_b79k",
                        help="OpenCLIP 预训练权重（默认：laion2b_s34b_b79k）")
    parser.add_argument("--epsilon",          type=float, default=16.0,
                        help="L-inf 扰动上限（0-255 尺度，默认：16）")
    parser.add_argument("--step_size",        type=float, default=1.0,
                        help="PGD 步长（0-255 尺度，默认：1）")
    parser.add_argument("--steps",            type=int,   default=300,
                        help="PGD 迭代步数 / UAP 每轮步数（默认：300）")
    parser.add_argument("--uap_epochs",       type=int,   default=5,
                        help="UAP 训练轮数（默认：5，仅 uap 模式）")
    parser.add_argument("--image_size",       type=int,   default=224,
                        help="输入图像尺寸（默认：224）")
    parser.add_argument("--match_mode",       type=str,   default="auto",
                        choices=["auto", "name", "order"],
                        help="配对模式：auto/name/order（默认：auto）")
    parser.add_argument("--seed",             type=int,   default=DEFAULT_SEED,
                        help=f"随机种子（默认：{DEFAULT_SEED}）")
    parser.add_argument("--log_interval",     type=int,   default=50,
                        help="每隔 N 步打印一次损失（默认：50）")

    args = parser.parse_args()
    seed_everything(args.seed)

    epsilon_01   = args.epsilon   / 255.0
    step_size_01 = args.step_size / 255.0

    print(f"\n{'='*65}")
    print("AttackXTransfer - 基于 CLIP 的对抗攻击")
    print(f"{'='*65}")
    print(f"  原图目录：     {args.source_dir}")
    print(f"  目标目录：     {args.target_dir}")
    print(f"  输出目录：     {args.output_dir}")
    print(f"  攻击模式：     {args.mode}")
    print(f"  攻击类型：     {args.attack_type}")
    print(f"  CLIP 模型：    {args.clip_model} ({args.clip_pretrained})")
    print(f"  Epsilon：      {args.epsilon}/255")
    print(f"  步长：         {args.step_size}/255")
    print(f"  步数：         {args.steps}")
    if args.mode == "uap":
        print(f"  UAP 轮数：     {args.uap_epochs}")
    print(f"  图像尺寸：     {args.image_size}")
    print(f"  配对模式：     {args.match_mode}")
    print(f"  设备：         {DEVICE}")
    print(f"  随机种子：     {args.seed}")
    print(f"{'='*65}\n")

    print("[第1步/共3步] 构建原图-目标对 ...")
    pairs = build_pairs(
        source_dir=args.source_dir, target_dir=args.target_dir,
        image_size=args.image_size, match_mode=args.match_mode,
    )

    print("\n[第2步/共3步] 加载 CLIP 代理模型 ...")
    model, tokenizer, normalize = load_clip_model(
        model_name=args.clip_model, pretrained=args.clip_pretrained)

    print(f"\n[第3步/共3步] 执行 {args.mode} 攻击（{args.attack_type}）...")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "per_image":
        match_info = run_per_image_attack(
            pairs=pairs, output_dir=args.output_dir,
            model=model, tokenizer=tokenizer, normalize=normalize,
            attack_type=args.attack_type,
            epsilon=epsilon_01, step_size=step_size_01, steps=args.steps,
            log_interval=args.log_interval,
        )
    else:
        match_info = run_uap_attack(
            pairs=pairs, output_dir=args.output_dir,
            model=model, tokenizer=tokenizer, normalize=normalize,
            attack_type=args.attack_type,
            epsilon=epsilon_01, step_size=step_size_01, steps=args.steps,
            uap_epochs=args.uap_epochs, image_size=args.image_size,
            log_interval=args.log_interval,
        )

    match_info_path = os.path.join(args.output_dir, "match_info.json")
    with open(match_info_path, "w", encoding="utf-8") as f:
        json.dump(match_info, f, indent=2, ensure_ascii=False)

    timing  = match_info.get("timing", {})
    n_ok    = timing.get("success", 0)
    total_t = timing.get("total_time_sec", 0.0)

    print(f"\n{'='*65}")
    print("XTransfer CLIP 攻击完成！")
    print(f"{'='*65}")
    print(f"  攻击模式：     {args.mode}")
    print(f"  攻击类型：     {args.attack_type}")
    print(f"  生成数量：     {n_ok} 张对抗图像")
    print(f"  输出目录：     {args.output_dir}")
    print(f"  match_info：   {match_info_path}")
    print(f"  总耗时：       {total_t:.2f}s")
    print(f"{'='*65}")

    per_times = timing.get("per_image_times_sec",
                    timing.get("per_image_apply_times_sec", []))
    write_attack_log(
        output_dir=args.output_dir,
        script_name="AttackXTransfer.py",
        total=timing.get("total_images", 0),
        success=n_ok,
        failed=timing.get("failed", 0),
        total_time=total_t,
        per_image_times=per_times,
    )


if __name__ == "__main__":
    main()
