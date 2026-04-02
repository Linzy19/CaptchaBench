"""
AttackVLM.py - VLM Text-level Adversarial Attack

Attack strategy:
    - Use source images (original images) as the tool images to perturb
    - Use target text (annotation prompt + label) as the attack objective
    - Forces VLM models to generate target text when given the adversarial image
    - Uses SSA_CommonWeakness with ensemble of VLM surrogates (BLIP2, InstructBLIP, MiniGPT4)

Based on: attack_vlm_misclassify.py + quick_test_attack_cuda_v4.py

Usage:
    # Basic usage - attack with target text from JSON annotation files
    python AttackVLM.py --source_dir ./dataset/NIPS17 --target_text "This is a cat" --output_dir ./output_attack_vlm

    # Use target text from JSON files (prompt + label from annotations)
    python AttackVLM.py --source_dir ./source --output_dir ./output --use_json_text

    # Custom attack parameters
    python AttackVLM.py --source_dir ./source --target_text "target description" --output_dir ./output \
        --epsilon 16 --steps 500

    # Use specific prompt + label combination
    python AttackVLM.py --source_dir ./source --target_prompt "Rocky mountain terrain" \
        --target_label "贞" --output_dir ./output
"""

import torch
import torch.nn as nn
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

# Add the Attack-Bard root to path for importing modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from surrogates import get_gpt4_image_model, Blip2VisionModel, InstructBlipVisionModel
from attacks import SSA_CommonWeakness
from utils.ImageHandling import save_image, get_image

DEFAULT_RANDOM_SEED = 2023


def seed_everything(seed=DEFAULT_RANDOM_SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_image_files(directory):
    """Get all image files in a directory"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPEG', '*.JPG', '*.PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(image_paths)


class VLMAttackCriterion:
    """
    Custom criterion for VLM attack.
    The VLM surrogates return loss (lower = more similar to target text).
    We negate the loss to maximize the similarity (targeted attack).
    """
    def __init__(self, log_interval=120):
        self.count = 0
        self.log_interval = log_interval

    def __call__(self, loss, *args):
        self.count += 1
        if self.count % self.log_interval == 0:
            print(f"    [Step {self.count}] VLM loss: {loss.item():.4f}")
        return -loss


def build_target_text_from_json(json_path):
    """
    Build target text from a JSON annotation file.
    Combines the prompt and label fields.

    JSON format example:
    {
        "prompt": "Rocky mountain terrain with ...",
        "annotations": [{"label": "贞", ...}]
    }

    Returns: "Rocky mountain terrain with ... Label: 贞"
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    parts = []

    # Add prompt
    prompt = data.get("prompt", "")
    if prompt:
        parts.append(prompt)

    # Add labels from annotations
    annotations = data.get("annotations", [])
    labels = [ann.get("label", "") for ann in annotations if ann.get("label")]
    if labels:
        label_str = ", ".join(labels)
        parts.append(f"Label: {label_str}")

    return " ".join(parts) if parts else None


def load_source_images_with_text(source_dir, target_text=None, target_prompt=None,
                                  target_label=None, use_json_text=False, input_res=224):
    """
    Load source images and determine target text for each.

    Priority:
        1. use_json_text=True: read prompt+label from each image's JSON file
        2. target_prompt + target_label: combine them
        3. target_text: use directly

    Returns:
        list of (source_path, target_text_str, source_tensor)
    """
    source_files = get_image_files(source_dir)
    if not source_files:
        raise ValueError(f"No images in source directory: {source_dir}")

    resizer = transforms.Resize((input_res, input_res))
    to_tensor = transforms.ToTensor()

    items = []

    for src_path in tqdm(source_files, desc="Loading source images"):
        try:
            src_img = to_tensor(Image.open(src_path).convert("RGB"))
            src_img = resizer(src_img)

            # Determine target text
            text = None

            if use_json_text:
                # Try to read from JSON file with same name
                json_path = os.path.splitext(src_path)[0] + ".json"
                if os.path.exists(json_path):
                    text = build_target_text_from_json(json_path)
                if not text:
                    print(f"  [Warning] No JSON or no valid text for {os.path.basename(src_path)}, using fallback text")

            if text is None and target_prompt and target_label:
                text = f"{target_prompt} Label: {target_label}"

            if text is None and target_text:
                text = target_text

            if text is None:
                print(f"  ✗ No target text for {os.path.basename(src_path)}, skipping")
                continue

            items.append((src_path, text, src_img))
        except Exception as e:
            print(f"  ✗ Failed to load {os.path.basename(src_path)}: {e}")

    if not items:
        raise ValueError("No valid source images with target text found!")

    print(f"[Done] Loaded {len(items)} source images with target text")
    return items


def run_vlm_attack(items, output_dir, epsilon=16, step_size=1, total_step=500,
                    use_gpt4=True, log_interval=120):
    """
    Run VLM-level adversarial attack using BLIP2/InstructBLIP/(MiniGPT4) surrogates.

    Attack flow:
        1. Group items by target_text (models need to be reloaded for different texts)
        2. For each text group:
            a. Load VLM surrogates with this target text
            b. Build SSA_CommonWeakness attacker
            c. For each source image, run attack
        3. Save adversarial images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group by target text to reuse models
    text_groups = {}
    for src_path, text, src_tensor in items:
        if text not in text_groups:
            text_groups[text] = []
        text_groups[text].append((src_path, src_tensor))

    print(f"\n[Info] {len(text_groups)} unique target text(s), {len(items)} total images")

    match_info = {
        "attack_type": "vlm_text_attack",
        "method": "SSA_CommonWeakness",
        "parameters": {
            "epsilon": epsilon,
            "step_size": step_size,
            "total_step": total_step,
            "use_gpt4": use_gpt4,
        },
        "text_groups": [],
        "pairs": [],
    }

    total_start = time.time()
    total_processed = 0
    per_image_times = []
    n_success = 0
    n_failed = 0

    for group_idx, (target_text, group_items) in enumerate(text_groups.items()):
        print(f"\n{'=' * 60}")
        print(f"Text Group [{group_idx+1}/{len(text_groups)}]")
        print(f"  Target Text: {target_text[:100]}{'...' if len(target_text) > 100 else ''}")
        print(f"  Images: {len(group_items)}")
        print(f"{'=' * 60}")

        # Load VLM models for this target text
        model_load_start = time.time()
        print("\nLoading VLM surrogate models...")

        vlm_models = []

        try:
            blip2 = Blip2VisionModel(target_text=target_text)
            vlm_models.append(blip2)
            print("  ✓ BLIP2 loaded")
        except Exception as e:
            print(f"  ✗ BLIP2 failed: {e}")

        try:
            instruct_blip = InstructBlipVisionModel(target_text=target_text)
            vlm_models.append(instruct_blip)
            print("  ✓ InstructBLIP loaded")
        except Exception as e:
            print(f"  ✗ InstructBLIP failed: {e}")

        if use_gpt4:
            try:
                gpt4 = get_gpt4_image_model(target_text=target_text)
                vlm_models.append(gpt4)
                print("  ✓ MiniGPT4 loaded")
            except Exception as e:
                print(f"  ✗ MiniGPT4 failed: {e}")

        if not vlm_models:
            print("  [Error] No VLM models loaded, skipping this group")
            continue

        model_load_time = time.time() - model_load_start
        print(f"  Model load time: {model_load_time:.2f}s")

        # Build criterion and attacker
        criterion = VLMAttackCriterion(log_interval=log_interval)
        attacker = SSA_CommonWeakness(
            vlm_models,
            epsilon=epsilon / 255.0,
            step_size=step_size / 255.0,
            total_step=total_step,
            criterion=criterion,
        )

        # Attack each image in this group
        group_start = time.time()
        for i, (src_path, src_tensor) in enumerate(group_items):
            print(f"\n  [{i+1}/{len(group_items)}] Attacking: {os.path.basename(src_path)}")
            img_start = time.time()

            src_img = src_tensor.unsqueeze(0).cuda()
            adv_img = attacker(src_img, None)

            # Save adversarial image
            src_name = os.path.splitext(os.path.basename(src_path))[0]
            output_path = os.path.join(output_dir, f"{src_name}_adv.png")
            save_image(adv_img.squeeze(0), output_path)
            print(f"    ✓ Saved: {output_path}")

            # Copy corresponding source JSON file if exists
            src_json_path = os.path.splitext(src_path)[0] + ".json"
            if os.path.exists(src_json_path):
                dst_json_path = os.path.join(output_dir, f"{src_name}_adv.json")
                shutil.copy2(src_json_path, dst_json_path)

            match_info["pairs"].append({
                "source": os.path.basename(src_path),
                "target_text": target_text[:200],
                "output": f"{src_name}_adv.png",
            })
            total_processed += 1
            img_elapsed = time.time() - img_start
            per_image_times.append(img_elapsed)
            n_success += 1

        group_time = time.time() - group_start
        match_info["text_groups"].append({
            "target_text": target_text[:200],
            "num_images": len(group_items),
            "attack_time_sec": group_time,
        })

        # Clean up GPU memory
        del vlm_models, attacker, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # Save match info
    match_info["timing"] = {
        "total_time_sec": total_time,
        "total_images": total_processed,
        "avg_time_per_image_sec": total_time / total_processed if total_processed else 0,
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
        script_name="AttackVLM.py",
        total=total_processed,
        success=n_success,
        failed=n_failed,
        total_time=total_time,
        per_image_times=per_image_times,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"VLM Attack Complete!")
    print(f"{'=' * 60}")
    print(f"  Generated: {total_processed} adversarial images")
    print(f"  Text Groups: {len(text_groups)}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Total Time: {total_time:.2f}s")
    if total_processed:
        print(f"  Avg per Image: {total_time/total_processed:.2f}s")
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


def main():
    parser = argparse.ArgumentParser(description="AttackVLM - VLM Text-level Adversarial Attack")

    # Directory parameters
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Source image directory (original images to perturb)")
    parser.add_argument("--output_dir", type=str, default="./output_attack_vlm",
                        help="Output directory for adversarial images")

    # Target text parameters (mutually exclusive modes)
    parser.add_argument("--target_text", type=str, default=None,
                        help="Target text for all images (e.g., 'This is a bomb')")
    parser.add_argument("--target_prompt", type=str, default=None,
                        help="Target prompt (combined with --target_label)")
    parser.add_argument("--target_label", type=str, default=None,
                        help="Target label (combined with --target_prompt)")
    parser.add_argument("--use_json_text", action="store_true",
                        help="Read target text from each image's JSON file (prompt + label)")

    # Attack parameters
    parser.add_argument("-e", "--epsilon", type=float, default=16,
                        help="Max perturbation (0-255 scale, default: 16)")
    parser.add_argument("--step_size", type=float, default=1,
                        help="PGD step size (0-255 scale, default: 1)")
    parser.add_argument("-s", "--steps", type=int, default=500,
                        help="Total attack steps (default: 500)")
    parser.add_argument("-r", "--input_res", type=int, default=224,
                        help="Input resolution (default: 224)")
    parser.add_argument("--no_gpt4", action="store_true",
                        help="Disable MiniGPT4 model (use only BLIP2 + InstructBLIP)")
    parser.add_argument("--log_interval", type=int, default=120,
                        help="Log loss every N steps (default: 120)")

    # Other
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to process (default: all)")

    args = parser.parse_args()

    # Validate target text specification
    if not args.target_text and not args.use_json_text and not (args.target_prompt and args.target_label):
        parser.error("Must specify one of: --target_text, --use_json_text, or (--target_prompt + --target_label)")

    seed_everything(args.seed)

    print(f"\n{'=' * 60}")
    print(f"AttackVLM - VLM Text-level Adversarial Attack")
    print(f"{'=' * 60}")
    print(f"Source Dir:    {args.source_dir}")
    print(f"Output Dir:    {args.output_dir}")
    if args.target_text:
        print(f"Target Text:   {args.target_text[:80]}{'...' if args.target_text and len(args.target_text) > 80 else ''}")
    elif args.use_json_text:
        print(f"Target Text:   [From JSON annotation files]")
    elif args.target_prompt and args.target_label:
        print(f"Target Prompt: {args.target_prompt[:60]}...")
        print(f"Target Label:  {args.target_label}")
    print(f"Epsilon:       {args.epsilon}")
    print(f"Step Size:     {args.step_size}")
    print(f"Steps:         {args.steps}")
    print(f"Resolution:    {args.input_res}")
    print(f"Use MiniGPT4:  {not args.no_gpt4}")
    print(f"Seed:          {args.seed}")
    print(f"{'=' * 60}")

    # Load source images with target text
    items = load_source_images_with_text(
        args.source_dir,
        target_text=args.target_text,
        target_prompt=args.target_prompt,
        target_label=args.target_label,
        use_json_text=args.use_json_text,
        input_res=args.input_res,
    )

    # Limit number of images if specified
    if args.max_images and len(items) > args.max_images:
        print(f"[Limit] Processing first {args.max_images} of {len(items)} images")
        items = items[:args.max_images]

    # Run VLM attack
    run_vlm_attack(
        items,
        args.output_dir,
        epsilon=args.epsilon,
        step_size=args.step_size,
        total_step=args.steps,
        use_gpt4=not args.no_gpt4,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
