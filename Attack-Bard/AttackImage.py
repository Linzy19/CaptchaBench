"""
AttackImage.py - Image Encoder Feature Attack (Image-level)

Attack strategy:
    - Use source images (original images) as the tool images to perturb
    - Use target images as the attack targets
    - Align source image features toward target image features
    - Uses SSA_CommonWeakness with ensemble of BLIP/CLIP/ViT feature extractors

Based on: attack_img_encoder_misdescription.py + quick_test_attack_cuda_v4.py

Usage:
    # Basic usage - all source images attack toward all target images (random matching)
    python AttackImage.py --source_dir ./dataset/NIPS17 --target_dir ./target_images --output_dir ./output_attack_img

    # Custom attack parameters
    python AttackImage.py --source_dir ./source --target_dir ./target --output_dir ./output \
        --epsilon 16 --steps 500 --step_size 1 --match_mode random

    # Single target mode - all source images attack toward one target
    python AttackImage.py --source_dir ./source --target_dir ./target --output_dir ./output --match_mode single

    # Use target_paths.json
    python AttackImage.py --source_dir ./source --target_json /path/to/target_paths.json --output_dir ./output
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

from surrogates import (
    BlipFeatureExtractor,
    ClipFeatureExtractor,
    EnsembleFeatureLoss,
    VisionTransformerFeatureExtractor,
)
from attacks import SSA_CommonWeakness
from utils import get_list_image, save_list_images
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


def load_target_paths(target_dir, target_json=None):
    """
    Load target image paths with fallback chain:
        1. If target_json is specified and exists, load from it
        2. If target_dir/target_paths.json exists, load from it
        3. Otherwise, auto-scan target_dir for image files
    """
    json_candidates = []
    if target_json:
        json_candidates.append(target_json if os.path.isabs(target_json) else os.path.join(target_dir, target_json))
    json_candidates.append(os.path.join(target_dir, "target_paths.json"))

    for json_path in json_candidates:
        if os.path.exists(json_path):
            try:
                print(f"[Load] Reading target paths from JSON: {json_path}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                target_files = data if isinstance(data, list) else data.get("target_files", [])
                if target_files:
                    print(f"[Load] Successfully loaded {len(target_files)} target paths")
                    return target_files
            except Exception as e:
                print(f"[Warning] Failed to load {json_path}: {e}")

    # Fallback: auto-scan directory
    print(f"[Auto-scan] Scanning directory: {target_dir}")
    target_files = get_image_files(target_dir)
    if not target_files:
        raise ValueError(f"No image files found in target directory: {target_dir}")
    print(f"[Auto-scan] Found {len(target_files)} image files")
    return target_files


def match_source_target_pairs(source_dir, target_dir, target_json=None,
                               match_mode="auto", input_res=224):
    """
    Load source and target images with flexible matching strategies.

    Match modes:
        - "auto": auto-detect based on target count
        - "single": force single-target mode
        - "random": random matching
    """
    source_files = get_image_files(source_dir)
    target_files = load_target_paths(target_dir, target_json)

    if not source_files:
        raise ValueError(f"No images in source directory: {source_dir}")
    if not target_files:
        raise ValueError(f"No target images found for: {target_dir}")

    num_targets = len(target_files)
    num_sources = len(source_files)

    if match_mode == "auto":
        match_mode = "single" if num_targets == 1 else "random"
        print(f"\n[Auto] Detected {num_targets} target(s) -> using '{match_mode}' mode")

    print(f"[Info] Match mode: {match_mode.upper()} | Source: {num_sources} | Target: {num_targets}")

    resizer = transforms.Resize((input_res, input_res))
    to_tensor = transforms.ToTensor()

    pairs = []  # list of (source_path, target_path, source_tensor, target_tensor)

    if match_mode == "single":
        # All sources use the same target
        tgt_path = target_files[0]
        tgt_img = to_tensor(Image.open(tgt_path).convert("RGB"))
        tgt_img = resizer(tgt_img)
        print(f"\n[Single] All sources will use: {os.path.basename(tgt_path)}")
        for src_path in tqdm(source_files, desc="Loading source images"):
            try:
                src_img = to_tensor(Image.open(src_path).convert("RGB"))
                src_img = resizer(src_img)
                pairs.append((src_path, tgt_path, src_img, tgt_img.clone()))
            except Exception as e:
                print(f"  ✗ Failed to load {os.path.basename(src_path)}: {e}")

    elif match_mode == "random":
        # Random matching
        print(f"\n[Random] Matching {num_sources} sources from {num_targets} targets...")
        for src_path in tqdm(source_files, desc="Loading image pairs"):
            tgt_path = random.choice(target_files)
            try:
                src_img = to_tensor(Image.open(src_path).convert("RGB"))
                src_img = resizer(src_img)
                tgt_img = to_tensor(Image.open(tgt_path).convert("RGB"))
                tgt_img = resizer(tgt_img)
                pairs.append((src_path, tgt_path, src_img, tgt_img))
            except Exception as e:
                print(f"  ✗ Failed to load pair: {e}")

    if not pairs:
        raise ValueError("No valid source-target pairs found!")

    print(f"[Done] Matched successfully: {len(pairs)} pairs")
    return pairs


def ssa_cw_count_to_index(count, num_models=3, ssa_N=20):
    """
    SSA_CommonWeakness model index selector.
    Rotates through models every ssa_N SSA iterations.
    """
    max_count = ssa_N * num_models
    count = count % max_count
    count = count // ssa_N
    return count


def run_attack(pairs, output_dir, epsilon=16, step_size=1, total_step=500, input_res=224):
    """
    Run image-level adversarial attack using BLIP/CLIP/ViT feature extractors + SSA_CommonWeakness.

    Attack flow:
        1. Load ensemble feature extractors (BLIP, CLIP, ViT)
        2. For each source-target pair:
            a. Set target image features as ground truth
            b. Apply SSA_CommonWeakness to perturb source image toward target features
        3. Save adversarial images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize feature extractors
    print("\n" + "=" * 60)
    print("Initializing Feature Extractors...")
    print("=" * 60)

    model_load_start = time.time()
    blip = BlipFeatureExtractor().eval().cuda().requires_grad_(False)
    print("  ✓ BLIP Feature Extractor loaded")
    clip = ClipFeatureExtractor().eval().cuda().requires_grad_(False)
    print("  ✓ CLIP Feature Extractor loaded")
    vit = VisionTransformerFeatureExtractor().eval().cuda().requires_grad_(False)
    print("  ✓ ViT Feature Extractor loaded")
    models = [vit, blip, clip]
    model_load_time = time.time() - model_load_start
    print(f"  Model load time: {model_load_time:.2f}s")

    # Build EnsembleFeatureLoss - uses target features as ground truth
    ssa_cw_loss = EnsembleFeatureLoss(
        models,
        lambda count: ssa_cw_count_to_index(count, num_models=len(models)),
        feature_loss=torch.nn.MSELoss()
    )

    # Build SSA_CommonWeakness attacker
    attacker = SSA_CommonWeakness(
        models,
        epsilon=epsilon / 255.0,
        step_size=step_size / 255.0,
        total_step=total_step,
        criterion=ssa_cw_loss,
    )

    print(f"\n{'=' * 60}")
    print(f"Attack Parameters:")
    print(f"  Epsilon: {epsilon}/255 = {epsilon/255:.4f}")
    print(f"  Step Size: {step_size}/255 = {step_size/255:.4f}")
    print(f"  Total Steps: {total_step}")
    print(f"  Input Resolution: {input_res}")
    print(f"  Number of Pairs: {len(pairs)}")
    print(f"{'=' * 60}\n")

    # Match info for logging
    match_info = {
        "attack_type": "image_encoder_feature_attack",
        "method": "SSA_CommonWeakness",
        "models": ["ViT", "BLIP", "CLIP"],
        "parameters": {
            "epsilon": epsilon,
            "step_size": step_size,
            "total_step": total_step,
            "input_res": input_res,
        },
        "pairs": [],
    }

    attack_start = time.time()
    json_copied = 0

    for i, (src_path, tgt_path, src_tensor, tgt_tensor) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] Attacking: {os.path.basename(src_path)} -> {os.path.basename(tgt_path)}")

        src_img = src_tensor.unsqueeze(0).cuda()
        tgt_img = tgt_tensor.unsqueeze(0).cuda()

        # Set target image features as ground truth for EnsembleFeatureLoss
        ssa_cw_loss.set_ground_truth(tgt_img)

        # Run attack: perturb source image to align with target features
        adv_img = attacker(src_img, None)

        # Save adversarial image
        src_name = os.path.splitext(os.path.basename(src_path))[0]
        output_path = os.path.join(output_dir, f"{src_name}_adv.png")
        save_image(adv_img.squeeze(0), output_path)
        print(f"  ✓ Saved: {output_path}")

        # Copy corresponding source JSON file if exists
        src_json_path = os.path.splitext(src_path)[0] + ".json"
        if os.path.exists(src_json_path):
            dst_json_path = os.path.join(output_dir, f"{src_name}_adv.json")
            shutil.copy2(src_json_path, dst_json_path)
            json_copied += 1

        match_info["pairs"].append({
            "source": os.path.basename(src_path),
            "target": os.path.basename(tgt_path),
            "output": f"{src_name}_adv.png",
        })

    attack_time = time.time() - attack_start

    # Save match info
    match_info["timing"] = {
        "model_load_time_sec": model_load_time,
        "attack_time_sec": attack_time,
        "total_time_sec": model_load_time + attack_time,
        "avg_time_per_image_sec": attack_time / len(pairs) if pairs else 0,
    }
    with open(os.path.join(output_dir, "match_info.json"), 'w', encoding='utf-8') as f:
        json.dump(match_info, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Attack Complete!")
    print(f"{'=' * 60}")
    print(f"  Generated: {len(pairs)} adversarial images")
    print(f"  JSON Copied: {json_copied}/{len(pairs)}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Model Load: {model_load_time:.2f}s")
    print(f"  Attack Time: {attack_time:.2f}s")
    print(f"  Total: {model_load_time + attack_time:.2f}s")
    print(f"  Avg per Image: {attack_time/len(pairs):.2f}s")
    print(f"{'=' * 60}")

    return match_info


def main():
    parser = argparse.ArgumentParser(description="AttackImage - Image Encoder Feature Attack")

    # Directory parameters
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Source image directory (original images to perturb)")
    parser.add_argument("--target_dir", type=str, default=None,
                        help="Target image directory (attack target features)")
    parser.add_argument("--target_json", type=str, default=None,
                        help="Target image paths JSON file")
    parser.add_argument("--output_dir", type=str, default="./output_attack_image",
                        help="Output directory for adversarial images")

    # Attack parameters
    parser.add_argument("-e", "--epsilon", type=float, default=16,
                        help="Max perturbation (0-255 scale, default: 16)")
    parser.add_argument("--step_size", type=float, default=1,
                        help="PGD step size (0-255 scale, default: 1)")
    parser.add_argument("-s", "--steps", type=int, default=500,
                        help="Total attack steps (default: 500)")
    parser.add_argument("-r", "--input_res", type=int, default=224,
                        help="Input resolution (default: 224)")
    parser.add_argument("-m", "--match_mode", type=str, default="auto",
                        choices=["auto", "single", "random"],
                        help="Source-target matching strategy (default: auto)")

    # Other
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")

    args = parser.parse_args()

    # Validate directories
    if args.target_dir is None and args.target_json is None:
        parser.error("Must specify either --target_dir or --target_json")
    if args.target_dir is None:
        # Derive target_dir from target_json path
        args.target_dir = os.path.dirname(args.target_json)

    seed_everything(args.seed)

    print(f"\n{'=' * 60}")
    print(f"AttackImage - Image Encoder Feature Attack")
    print(f"{'=' * 60}")
    print(f"Source Dir:  {args.source_dir}")
    print(f"Target Dir:  {args.target_dir}")
    print(f"Output Dir:  {args.output_dir}")
    print(f"Match Mode:  {args.match_mode}")
    print(f"Epsilon:     {args.epsilon}")
    print(f"Step Size:   {args.step_size}")
    print(f"Steps:       {args.steps}")
    print(f"Resolution:  {args.input_res}")
    print(f"Seed:        {args.seed}")
    print(f"{'=' * 60}")

    # Load and match image pairs
    pairs = match_source_target_pairs(
        args.source_dir,
        args.target_dir,
        target_json=args.target_json,
        match_mode=args.match_mode,
        input_res=args.input_res,
    )

    # Run attack
    run_attack(
        pairs,
        args.output_dir,
        epsilon=args.epsilon,
        step_size=args.step_size,
        total_step=args.steps,
        input_res=args.input_res,
    )


if __name__ == "__main__":
    main()
