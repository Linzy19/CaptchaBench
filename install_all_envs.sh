#!/usr/bin/env bash
# ============================================================
# install_all_envs.sh — 批量创建并安装所有 conda 环境
# 用法：
#   bash install_all_envs.sh [选项]
#
# 选项：
#   --force            强制重新创建所有环境（删除已有的重建）
#   --skip_mi          跳过 adv_attack 环境
#   --skip_aspl        跳过 anti_dreambooth 环境
#   --skip_mmcoa       跳过 mmcoa 环境
#   --skip_nightshade  跳过 nightshade 环境
#   --skip_xtransfer   跳过 xtransfer 环境
#   --skip_vlm         跳过 attack_bard 环境
#   --skip_attackvlm   跳过 attackvlm 环境
#   --cuda_version     PyTorch CUDA 版本（默认：11.8）
#   --python_version   Python 版本（默认：3.10）
# ============================================================
set -eo pipefail

# 错误处理：脚本失败时打印行号和退出码
trap 'echo "[错误] 脚本在第 ${LINENO} 行失败，退出码：$?" >&2' ERR

# ============================================================
# A区：全局配置
# ============================================================

# 强制重建标志（0=不强制, 1=强制删除后重建）
FORCE_RECREATE=0

# PyTorch CUDA 版本
PYTORCH_CUDA_VERSION="11.8"

# Python 版本
PYTHON_VERSION="3.10"

# 项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# conda 环境安装目录（持久化存储路径）
CONDA_ENV_DIR="/apdcephfs_qy3/share_470749/lzy_private/conda_env_lzy"

# 跳过标志（0=不跳过, 1=跳过）
SKIP_MI=0
SKIP_ASPL=0
SKIP_MMCOA=0
SKIP_NIGHTSHADE=0
SKIP_XTRANSFER=0
SKIP_VLM=0
SKIP_ATTACKVLM=0

# ============================================================
# B区：参数解析
# ============================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE_RECREATE=1
            shift
            ;;
        --skip_mi)
            SKIP_MI=1
            shift
            ;;
        --skip_aspl)
            SKIP_ASPL=1
            shift
            ;;
        --skip_mmcoa)
            SKIP_MMCOA=1
            shift
            ;;
        --skip_nightshade)
            SKIP_NIGHTSHADE=1
            shift
            ;;
        --skip_xtransfer)
            SKIP_XTRANSFER=1
            shift
            ;;
        --skip_vlm)
            SKIP_VLM=1
            shift
            ;;
        --skip_attackvlm)
            SKIP_ATTACKVLM=1
            shift
            ;;
        --cuda_version)
            PYTORCH_CUDA_VERSION="$2"
            shift 2
            ;;
        --python_version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        *)
            echo "[注意] 未识别的参数：$1，已忽略" >&2
            shift
            ;;
    esac
done

# ============================================================
# C区：conda 初始化函数
# ============================================================

init_conda() {
    local conda_base
    conda_base="$(conda info --base 2>/dev/null)" || {
        echo "[错误] 未找到 conda，请先安装 Anaconda/Miniconda" >&2
        exit 1
    }
    # shellcheck source=/dev/null
    source "${conda_base}/etc/profile.d/conda.sh"
}

# ============================================================
# D区：环境创建函数（幂等）
# 如果环境已存在且未指定 --force，则跳过创建
# 如果指定了 --force，则删除后重建
# ============================================================

create_or_skip_env() {
    local env_name="$1"
    local python_ver="$2"
    local env_path="${CONDA_ENV_DIR}/${env_name}"

    # 检查目标路径是否已存在环境
    if [[ -d "${env_path}" && -f "${env_path}/bin/python" ]]; then
        if [[ ${FORCE_RECREATE} -eq 1 ]]; then
            echo "[重建] 删除并重新创建环境：${env_path}"
            rm -rf "${env_path}"
        else
            echo "[跳过] 环境已存在：${env_path}（使用 --force 可强制重建）"
            return 0
        fi
    fi

    echo "[创建] 创建环境：${env_path} (Python ${python_ver})"
    conda create --prefix "${env_path}" python="${python_ver}" -y
}

# ============================================================
# E区：PyTorch 安装函数
# ============================================================

install_pytorch() {
    local cuda_ver="$1"
    echo "[安装] PyTorch (CUDA ${cuda_ver})..."
    # 使用 pip 从 PyTorch 官方源安装，避免 conda nvidia 频道 404 问题
    local cuda_suffix
    cuda_suffix="cu$(echo "${cuda_ver}" | tr -d '.')"
    pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${cuda_suffix}"
}

# ============================================================
# F区：6个环境安装函数
# ============================================================

# ----------------------------
# install_adv_attack：安装 AdversarialAttacks 环境（adv_attack）
# 对应攻击脚本：AttackMI.py
# ----------------------------
install_adv_attack() {
    echo ""
    echo "==============================="
    echo "[安装] 安装环境：adv_attack (AttackMI)"
    echo "==============================="

    create_or_skip_env "adv_attack" "${PYTHON_VERSION}"
    set +e
    conda activate "${CONDA_ENV_DIR}/adv_attack"
    set -e

    # 安装 PyTorch
    install_pytorch "${PYTORCH_CUDA_VERSION}"

    # 安装项目依赖
    cd "${SCRIPT_DIR}/AdversarialAttacks"
    echo "[安装] 安装 AdversarialAttacks 依赖..."
    pip install -r requirements.txt

    # 验证安装
    echo "[验证] 验证 adv_attack 环境..."
    python -c "import torch; import timm; import scipy; print('adv_attack OK, CUDA:', torch.cuda.is_available())"

    echo "[信息] adv_attack 环境安装完成"
}

# ----------------------------
# install_anti_dreambooth：安装 Anti-DreamBooth 环境（anti_dreambooth）
# 对应攻击脚本：AttackASPL.py
# ----------------------------
install_anti_dreambooth() {
    echo ""
    echo "==============================="
    echo "[安装] 安装环境：anti_dreambooth (AttackASPL)"
    echo "==============================="

    create_or_skip_env "anti_dreambooth" "${PYTHON_VERSION}"
    set +e
    conda activate "${CONDA_ENV_DIR}/anti_dreambooth"
    set -e

    # 安装 PyTorch
    install_pytorch "${PYTORCH_CUDA_VERSION}"

    # 安装项目依赖（Anti-DreamBooth 使用 requirements_attack.txt）
    cd "${SCRIPT_DIR}/Anti-DreamBooth"
    echo "[安装] 安装 Anti-DreamBooth 攻击依赖..."
    pip install -r requirements_attack.txt

    # 验证安装
    echo "[验证] 验证 anti_dreambooth 环境..."
    python -c "import torch; import diffusers; print('anti_dreambooth OK, CUDA:', torch.cuda.is_available())"

    echo "[信息] anti_dreambooth 环境安装完成"
}

# ----------------------------
# install_mmcoa：安装 MMCoA 环境（mmcoa）
# 对应攻击脚本：AttackMMCoA.py
# ----------------------------
install_mmcoa() {
    echo ""
    echo "==============================="
    echo "[安装] 安装环境：mmcoa (AttackMMCoA)"
    echo "==============================="

    create_or_skip_env "mmcoa" "${PYTHON_VERSION}"
    set +e
    conda activate "${CONDA_ENV_DIR}/mmcoa"
    set -e

    # 安装 PyTorch
    install_pytorch "${PYTORCH_CUDA_VERSION}"

    # 安装项目依赖
    cd "${SCRIPT_DIR}/MMCoA"
    echo "[注意] MMCoA 需要从 GitHub 安装 OpenAI CLIP，需要网络连接..."
    pip install -r requirements.txt

    # 验证安装
    echo "[验证] 验证 mmcoa 环境..."
    python -c "import clip; import scipy; print('mmcoa OK')"

    echo "[信息] mmcoa 环境安装完成"
}

# ----------------------------
# install_nightshade：安装 nightshade-release 环境（nightshade）
# 对应攻击脚本：AttackNightshade.py
# ----------------------------
install_nightshade() {
    echo ""
    echo "==============================="
    echo "[安装] 安装环境：nightshade (AttackNightshade)"
    echo "==============================="

    create_or_skip_env "nightshade" "${PYTHON_VERSION}"
    set +e
    conda activate "${CONDA_ENV_DIR}/nightshade"
    set -e

    # 安装 PyTorch
    install_pytorch "${PYTORCH_CUDA_VERSION}"

    # 安装项目依赖
    cd "${SCRIPT_DIR}/nightshade-release"
    echo "[安装] 安装 nightshade 依赖..."
    pip install -r requirements.txt

    # 验证安装
    echo "[验证] 验证 nightshade 环境..."
    python -c "from diffusers import AutoencoderKL; from einops import rearrange; print('nightshade OK')"

    echo "[信息] nightshade 环境安装完成"
}

# ----------------------------
# install_xtransfer：安装 XTransferBench 环境（xtransfer）
# 对应攻击脚本：AttackXTransfer.py
# ----------------------------
install_xtransfer() {
    echo ""
    echo "==============================="
    echo "[安装] 安装环境：xtransfer (AttackXTransfer)"
    echo "==============================="

    create_or_skip_env "xtransfer" "${PYTHON_VERSION}"
    set +e
    conda activate "${CONDA_ENV_DIR}/xtransfer"
    set -e

    # 安装 PyTorch
    install_pytorch "${PYTORCH_CUDA_VERSION}"

    # 安装项目依赖
    cd "${SCRIPT_DIR}/XTransferBench"
    echo "[安装] 安装 XTransferBench 依赖..."
    pip install -r requirements.txt

    # 验证安装（open_clip 是 XTransferBench 的核心依赖）
    echo "[验证] 验证 xtransfer 环境..."
    python -c "import open_clip; print('xtransfer OK, version:', open_clip.__version__)"

    echo "[信息] xtransfer 环境安装完成"
}

# ----------------------------
# install_attack_bard：安装 Attack-Bard 环境（attack_bard）
# 对应攻击脚本：AttackVLM.py
# ----------------------------
install_attack_bard() {
    echo ""
    echo "==============================="
    echo "[安装] 安装环境：attack_bard (AttackVLM)"
    echo "==============================="

    create_or_skip_env "attack_bard" "${PYTHON_VERSION}"
    set +e
    conda activate "${CONDA_ENV_DIR}/attack_bard"
    set -e

    # 安装 PyTorch
    install_pytorch "${PYTORCH_CUDA_VERSION}"

    # 安装项目依赖
    cd "${SCRIPT_DIR}/Attack-Bard"
    echo "[安装] 安装 Attack-Bard 依赖..."
    pip install -r requirements.txt

    # 验证安装
    echo "[验证] 验证 attack_bard 环境..."
    python -c "import torch; import transformers; import timm; print('attack_bard OK, CUDA:', torch.cuda.is_available())"

    echo "[信息] attack_bard 环境安装完成"
}

# ----------------------------
# install_attackvlm：安装 AttackVLM 环境（attackvlm）
# 对应攻击脚本：AttackVLMNew.py
# 使用 LAVIS 库（BLIP/BLIP2 视觉编码器迁移攻击）
# ----------------------------
install_attackvlm() {
    echo ""
    echo "==============================="
    echo "[安装] 安装环境：attackvlm (AttackVLMNew)"
    echo "==============================="

    create_or_skip_env "attackvlm" "${PYTHON_VERSION}"
    set +e
    conda activate "${CONDA_ENV_DIR}/attackvlm"
    set -e

    # 安装 PyTorch
    install_pytorch "${PYTORCH_CUDA_VERSION}"

    # 安装 LAVIS（salesforce-lavis）
    echo "[安装] 安装 salesforce-lavis..."
    pip install salesforce-lavis

    # 安装 CLIP（AttackVLM LAVIS_tool 依赖）
    echo "[安装] 安装 openai-clip..."
    pip install git+https://github.com/openai/CLIP.git

    # 安装其他常用依赖
    pip install einops lmdb tqdm wandb

    # 验证安装
    echo "[验证] 验证 attackvlm 环境..."
    python -c "import torch; from lavis.models import load_model_and_preprocess; import clip; print('attackvlm OK, CUDA:', torch.cuda.is_available())"

    echo "[信息] attackvlm 环境安装完成"
}

main() {
    echo "[信息] install_all_envs.sh 启动"
    echo "[信息] 项目根目录：${SCRIPT_DIR}"

    # 初始化 conda
    init_conda

    echo "============================================================"
    echo "[信息] 批量安装所有对抗攻击 conda 环境"
    echo "[信息] Python 版本：${PYTHON_VERSION}"
    echo "[信息] PyTorch CUDA 版本：${PYTORCH_CUDA_VERSION}"
    echo "[信息] 强制重建：$([[ ${FORCE_RECREATE} -eq 1 ]] && echo '是' || echo '否')"
    echo "============================================================"

    # 根据跳过标志决定安装哪些环境
    # 注意：不使用 "[[...]] && func || true" 模式，因为 || true 会掩盖 func 内部的真实错误
    if [[ ${SKIP_MI} -eq 0 ]];         then install_adv_attack;      fi
    if [[ ${SKIP_ASPL} -eq 0 ]];       then install_anti_dreambooth;  fi
    if [[ ${SKIP_MMCOA} -eq 0 ]];      then install_mmcoa;            fi
    if [[ ${SKIP_NIGHTSHADE} -eq 0 ]]; then install_nightshade;       fi
    if [[ ${SKIP_XTRANSFER} -eq 0 ]];  then install_xtransfer;        fi
    if [[ ${SKIP_VLM} -eq 0 ]];        then install_attack_bard;      fi
    if [[ ${SKIP_ATTACKVLM} -eq 0 ]];  then install_attackvlm;        fi

    echo ""
    echo "============================================================"
    echo "[信息] 所有环境安装完成！"
    echo ""
    echo "[信息] 使用方式："
    echo "  bash run_all_attacks.sh \\"
    echo "    --source_dir /path/to/source \\"
    echo "    --target_dir /path/to/target"
    echo "============================================================"
}

main "$@"
