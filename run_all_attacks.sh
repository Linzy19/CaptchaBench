#!/usr/bin/env bash
# ============================================================
# run_all_attacks.sh — 批量运行所有对抗攻击脚本
# 用法：
#   bash run_all_attacks.sh \
#     --source_dir /path/to/source \
#     --target_dir /path/to/target \
#     [--mini]                              # mini模式：只跑少量图片用于快速校验
#     [--epsilon 16] \        # 全局覆盖 epsilon（0-255尺度），不传则各方法使用论文默认值
#     [--steps 300] \         # 全局覆盖 steps，不传则各方法使用论文默认值
#     [--comp_epsilon "8 16 32"] \           # 比较模式：epsilon列表
#     [--comp_steps "100 300 500"] \         # 比较模式：steps列表
#     [--comp_strategy grid|zip] \           # grid=笛卡尔积，zip=逐对
#     [--match_json /path/to/match.json] \   # 从 match.json 加载预定义配对
#     [--sd_model /path/to/local/stable-diffusion-2-1] \  # SD 模型本地路径
#     [--skip_mi] [--skip_aspl] [--skip_mmcoa] \
#     [--skip_nightshade] [--skip_xtransfer] [--skip_bard] \
#     [--skip_attackvlm]
#
# 输出目录结构：
#   outputs/
#     run_20260310_163400/           # 每次运行自动创建带时间戳的目录
#       log/                          # 所有日志
#         AttackMI_eps16_steps300.log
#         AttackMI_eps16_steps300_resource_log.txt
#         ...
#       images/                       # 所有图片输出
#         mi_eps16_steps300/
#         aspl_eps16_steps300/
#         ...
#
# 示例（mini模式 - 快速校验）：
#   bash run_all_attacks.sh \
#     --source_dir /path/to/source \
#     --target_dir /path/to/target \
#     --match_json /path/to/match.json \
#     --mini
#
# 示例（全量模式）：
#   bash run_all_attacks.sh \
#     --source_dir /path/to/source \
#     --target_dir /path/to/target \
#     --match_json /path/to/match.json
# ============================================================
set -euo pipefail

# 错误处理：脚本失败时打印行号和退出码
trap 'echo "[错误] 脚本在第 ${LINENO} 行失败，退出码：$?" >&2' ERR

# ============================================================
# A区：全局变量默认值
# ============================================================

# 必需参数
SOURCE_DIR=""
TARGET_DIR=""

# 可选参数默认值
# 留空 = 各方法使用各自论文默认值（见 ATTACK_PARAMS.md）
# 传入后作为全局覆盖，统一覆盖所有方法（用于横向比较实验）
EPSILON=""
STEPS=""

# 各方法论文默认参数
# MI        : epsilon=16, steps=300  (CWA/ICLR2024)
# ASPL      : pgd_eps=0.05([-1,1]), pgd_steps=200, pgd_alpha=0.005  (Anti-DreamBooth/ICCV2023)
# MMCoA     : epsilon=1,  num_iters=100  (MMCoA/arXiv2404)
# Nightshade: eps=0.05([0,1]), steps=500  (Nightshade/Oakland2024)
# XTransfer : epsilon=12, steps=300  (XTransferBench/ICML2025)
# Bard      : epsilon=8,  steps=300  (AttackVLM/NeurIPS2023)
# AttackVLMNew: epsilon=8, steps=300  (参考AttackVLM/NeurIPS2023)

# 运行模式：0=全量模式, 1=mini模式（只跑少量图片用于快速校验）
MINI_MODE=0
MINI_COUNT=3  # mini模式下每个攻击跑多少张图片

# 输出根目录（运行时会自动在此目录下创建 run_<时间戳> 子目录）
OUTPUT_ROOT="/apdcephfs_qy3/share_470749/lzy_private/posion_attack/outputs"

# 运行时自动生成的目录（run_<时间戳>）
RUN_DIR=""
# 日志目录和图片输出目录（自动生成）
LOG_DIR=""
OUTPUT_BASE=""

# 比较模式参数
COMP_EPSILON=""
COMP_STEPS=""
COMP_STRATEGY="grid"  # grid=笛卡尔积 或 zip=逐对匹配

# SD 模型本地路径（ASPL/Nightshade 需要）
SD_MODEL="/apdcephfs_qy3/share_470749/lzy_private/posion_attack/models/stable-diffusion-2-1"

# match.json 配对文件（可选）
MATCH_JSON=""

# 跳过标志（0=不跳过, 1=跳过）
SKIP_MI=0
SKIP_ASPL=0
SKIP_MMCOA=0
SKIP_NIGHTSHADE=0
SKIP_XTRANSFER=0
SKIP_BARD=0
SKIP_ATTACKVLM=0

# 项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# bc 是否可用标志
BC_AVAILABLE=0

# ============================================================
# B区：参数解析（while循环）
# ============================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source_dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --target_dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        --mini)
            MINI_MODE=1
            shift
            ;;
        --mini_count)
            MINI_COUNT="$2"
            shift 2
            ;;
        --epsilon)
            EPSILON="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --comp_epsilon)
            COMP_EPSILON="$2"
            shift 2
            ;;
        --comp_steps)
            COMP_STEPS="$2"
            shift 2
            ;;
        --comp_strategy)
            COMP_STRATEGY="$2"
            shift 2
            ;;
        --match_json)
            MATCH_JSON="$2"
            shift 2
            ;;
        --sd_model)
            SD_MODEL="$2"
            shift 2
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
        --skip_bard)
            SKIP_BARD=1
            shift
            ;;
        --skip_attackvlm)
            SKIP_ATTACKVLM=1
            shift
            ;;
        *)
            echo "[注意] 未识别的参数：$1，已忽略" >&2
            shift
            ;;
    esac
done

# 参数校验：SOURCE_DIR 必须指定
if [[ -z "${SOURCE_DIR}" ]]; then
    echo "[错误] 必须指定 --source_dir 参数" >&2
    exit 1
fi

# 如果不是仅VLM模式，TARGET_DIR 也必须指定
ALL_NON_VLM_SKIPPED=$(( SKIP_MI && SKIP_ASPL && SKIP_MMCOA && SKIP_NIGHTSHADE && SKIP_XTRANSFER && SKIP_ATTACKVLM ))
if [[ ${ALL_NON_VLM_SKIPPED} -eq 0 ]] && [[ -z "${TARGET_DIR}" ]]; then
    echo "[错误] 必须指定 --target_dir 参数（如果只运行VLM攻击，请同时指定 --skip_mi --skip_aspl --skip_mmcoa --skip_nightshade --skip_xtransfer --skip_attackvlm）" >&2
    exit 1
fi

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
# D区：conda 环境切换函数
# ============================================================

activate_env() {
    local env_name="$1"
    echo "[环境] 切换到 conda 环境：${env_name}"
    conda deactivate 2>/dev/null || true
    conda activate "${env_name}" || {
        echo "[错误] 无法激活环境 ${env_name}，请先运行 install_all_envs.sh" >&2
        return 1
    }
    echo "[环境] Python 路径：$(which python)"
}

# ============================================================
# E0区：资源监控辅助函数
# ============================================================

GPU_MONITOR_PID=""
GPU_BASELINE_MEM=0

# 获取当前 GPU 显存使用量（MiB），取所有 GPU 中最大的
get_current_gpu_mem() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1
}

start_gpu_monitor() {
    local monitor_file="$1"
    # 记录启动时的基准显存（baseline），用于计算净增量
    GPU_BASELINE_MEM=$(get_current_gpu_mem)
    if [[ -z "${GPU_BASELINE_MEM}" ]]; then
        GPU_BASELINE_MEM=0
    fi
    echo "${GPU_BASELINE_MEM}" > "${monitor_file}.baseline"
    echo "${GPU_BASELINE_MEM}" > "${monitor_file}"
    echo "[显存] 基准显存(baseline): ${GPU_BASELINE_MEM} MiB"
    (
        local peak=${GPU_BASELINE_MEM}
        while true; do
            local current
            current=$(get_current_gpu_mem)
            if [[ -n "${current}" ]] && [[ "${current}" -gt "${peak}" ]]; then
                peak=${current}
                echo "${peak}" > "${monitor_file}"
            fi
            sleep 1
        done
    ) &
    GPU_MONITOR_PID=$!
}

# stop_gpu_monitor 返回格式: "peak_mem baseline_mem"
stop_gpu_monitor() {
    local monitor_file="$1"
    if [[ -n "${GPU_MONITOR_PID}" ]] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
    GPU_MONITOR_PID=""
    local peak=0
    local baseline=0
    if [[ -f "${monitor_file}" ]]; then
        peak=$(cat "${monitor_file}")
        rm -f "${monitor_file}"
    fi
    if [[ -f "${monitor_file}.baseline" ]]; then
        baseline=$(cat "${monitor_file}.baseline")
        rm -f "${monitor_file}.baseline"
    fi
    echo "${peak} ${baseline}"
}

format_duration() {
    local total_seconds="$1"
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

write_resource_log() {
    local log_dir="$1"
    local attack_name="$2"
    local epsilon="$3"
    local steps="$4"
    local start_time="$5"
    local end_time="$6"
    local peak_gpu_mem="$7"
    local baseline_gpu_mem="${8:-0}"

    local duration=$(( end_time - start_time ))
    local duration_fmt
    duration_fmt="$(format_duration ${duration})"

    # 计算净增显存
    local net_gpu_mem=$(( peak_gpu_mem - baseline_gpu_mem ))
    if [[ ${net_gpu_mem} -lt 0 ]]; then
        net_gpu_mem=0
    fi

    local start_str
    start_str="$(date -d @${start_time} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r ${start_time} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'N/A')"
    local end_str
    end_str="$(date -d @${end_time} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r ${end_time} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'N/A')"

    local log_file="${log_dir}/${attack_name}_eps${epsilon}_steps${steps}_resource_log.txt"

    {
        echo "═══════════════════════════════════════════════════════"
        echo "  资源使用日志 — ${attack_name}"
        echo "═══════════════════════════════════════════════════════"
        echo ""
        echo "  攻击方法       : ${attack_name}"
        echo "  Epsilon        : ${epsilon}"
        echo "  Steps          : ${steps}"
        echo ""
        echo "  ─── 时间信息 ───"
        echo "  开始时间       : ${start_str}"
        echo "  结束时间       : ${end_str}"
        echo "  总运行时间     : ${duration_fmt} (${duration} 秒)"
        echo ""
        echo "  ─── GPU 显存信息 ───"
        echo "  基准显存(启动前) : ${baseline_gpu_mem} MiB"
        echo "  峰值 GPU 显存    : ${peak_gpu_mem} MiB"
        echo "  净增显存(峰-基准): ${net_gpu_mem} MiB"
        echo ""
        echo "  ─── GPU 设备信息 ───"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
            echo "  GPU ${line}"
        done
        echo ""
        echo "═══════════════════════════════════════════════════════"
    } > "${log_file}"

    echo "[日志] 资源使用日志已保存至：${log_file}"
    echo "[日志]   运行时间：${duration_fmt} (${duration}秒)"
    echo "[日志]   基准显存：${baseline_gpu_mem} MiB → 峰值：${peak_gpu_mem} MiB → 净增：${net_gpu_mem} MiB"
}

# ============================================================
# E区：幂等性检查 & mini模式 match.json 生成
# ============================================================

should_skip() {
    local output_dir="$1"
    if [[ -f "${output_dir}/match_info.json" ]]; then
        echo "[跳过] 检测到已有结果：${output_dir}/match_info.json"
        return 0
    fi
    return 1
}

# mini模式：生成只包含前 N 对的临时 match.json
# 用法：get_effective_match_json
# 返回：如果是mini模式且有match.json，返回临时文件路径；否则返回原始MATCH_JSON
EFFECTIVE_MATCH_JSON=""
generate_mini_match_json() {
    if [[ ${MINI_MODE} -eq 0 ]] || [[ -z "${MATCH_JSON}" ]] || [[ ! -f "${MATCH_JSON}" ]]; then
        EFFECTIVE_MATCH_JSON="${MATCH_JSON}"
        return 0
    fi

    local tmp_json="${RUN_DIR}/_mini_match.json"
    if [[ -f "${tmp_json}" ]]; then
        # Already generated
        EFFECTIVE_MATCH_JSON="${tmp_json}"
        return 0
    fi

    python3 -c "
import json, sys
with open('${MATCH_JSON}', 'r') as f:
    data = json.load(f)
pairs = data.get('pairs', [])[:${MINI_COUNT}]
data['pairs'] = pairs
data['metadata']['pair_count'] = len(pairs)
data['metadata']['mini_mode'] = True
with open('${tmp_json}', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f'[mini] 生成mini模式 match.json：{len(pairs)} 对 -> ${tmp_json}')
"
    EFFECTIVE_MATCH_JSON="${tmp_json}"
}

# ============================================================
# 浮点计算辅助函数
# ============================================================

calc_eps_0_1() {
    local epsilon="$1"
    local raw
    if [[ ${BC_AVAILABLE} -eq 1 ]]; then
        raw="$(echo "scale=6; ${epsilon}/255" | bc)"
        case "${raw}" in
            .*) raw="0${raw}" ;;
        esac
        echo "${raw}"
    else
        awk "BEGIN { printf \"%.6f\", ${epsilon}/255 }"
    fi
}

calc_eps_aspl() {
    local epsilon="$1"
    local raw
    if [[ ${BC_AVAILABLE} -eq 1 ]]; then
        raw="$(echo "scale=6; ${epsilon}/255" | bc)"
        case "${raw}" in
            .*) raw="0${raw}" ;;
        esac
        echo "${raw}"
    else
        awk "BEGIN { printf \"%.6f\", ${epsilon}/255 }"
    fi
}

# ============================================================
# F区：各攻击函数
# 所有攻击函数使用全局变量 OUTPUT_BASE（图片输出）和 LOG_DIR（日志）
# ============================================================

# ----------------------------
# run_mi：运行 AttackMI（AdversarialAttacks）
# 论文默认：epsilon=16, steps=300  (CWA/ICLR2024)
# ----------------------------
run_mi() {
    local epsilon="${1:-16}"
    local steps="${2:-300}"
    local output_dir="${OUTPUT_BASE}/mi_eps${epsilon}_steps${steps}"

    echo ""
    echo "[信息] === AttackMI：epsilon=${epsilon}, steps=${steps} ==="
    echo "[信息] 输出目录：${output_dir}"

    if should_skip "${output_dir}"; then
        return 0
    fi

    mkdir -p "${output_dir}"
    activate_env "adv_attack"

    local gpu_monitor_file="${output_dir}/.gpu_peak_mem"
    start_gpu_monitor "${gpu_monitor_file}"
    local t_start
    t_start=$(date +%s)

    mkdir -p "${LOG_DIR}"
    local log_file="${LOG_DIR}/AttackMI_eps${epsilon}_steps${steps}.log"
    cd "${SCRIPT_DIR}/AdversarialAttacks"
    python AttackMI.py \
        --source_dir "${SOURCE_DIR}" \
        --target_dir "${TARGET_DIR}" \
        --output_dir "${output_dir}" \
        --epsilon "${epsilon}" \
        --steps "${steps}" \
        ${EFFECTIVE_MATCH_JSON:+--match_json "${EFFECTIVE_MATCH_JSON}"} \
        2>&1 | tee "${log_file}"

    local t_end
    t_end=$(date +%s)
    local gpu_result
    gpu_result=$(stop_gpu_monitor "${gpu_monitor_file}")
    local peak_mem
    peak_mem=$(echo "${gpu_result}" | awk '{print $1}')
    local baseline_mem
    baseline_mem=$(echo "${gpu_result}" | awk '{print $2}')
    write_resource_log "${LOG_DIR}" "AttackMI" "${epsilon}" "${steps}" "${t_start}" "${t_end}" "${peak_mem}" "${baseline_mem}"

    echo "[信息] AttackMI 完成，结果保存至：${output_dir}"
    echo "[信息] 运行日志：${log_file}"
}

# ----------------------------
# run_aspl：运行 AttackASPL（Anti-DreamBooth）
# 论文默认：pgd_eps=0.05([-1,1]空间), pgd_steps=200, pgd_alpha=0.005  (Anti-DreamBooth/ICCV2023)
# 传入全局 epsilon(0-255) 时换算：pgd_eps = epsilon/255
# 不传时直接使用论文值 pgd_eps=0.05，不做换算
# ----------------------------
run_aspl() {
    local epsilon="$1"
    local steps="${2:-200}"
    local pgd_eps
    local output_dir

    if [[ -z "${epsilon}" ]]; then
        # 论文默认：直接使用 0.05（[-1,1]空间），对应约 6.4/255 像素
        pgd_eps="0.050000"
        output_dir="${OUTPUT_BASE}/aspl_eps0.05_steps${steps}"
    else
        pgd_eps="$(calc_eps_aspl "${epsilon}")"
        output_dir="${OUTPUT_BASE}/aspl_eps${epsilon}_steps${steps}"
    fi

    # pgd_alpha 论文固定值：5e-3 = 0.005（Anti-DreamBooth ICCV2023）
    local pgd_alpha="0.005000"

    echo ""
    echo "[信息] === AttackASPL：pgd_eps=${pgd_eps}, pgd_steps=${steps} ==="
    echo "[信息] 输出目录：${output_dir}"
    echo "[信息] pgd_alpha=${pgd_alpha}（论文固定值 5e-3）"

    if should_skip "${output_dir}"; then
        return 0
    fi

    mkdir -p "${output_dir}"
    activate_env "anti_dreambooth"

    local gpu_monitor_file="${output_dir}/.gpu_peak_mem"
    start_gpu_monitor "${gpu_monitor_file}"
    local t_start
    t_start=$(date +%s)

    mkdir -p "${LOG_DIR}"
    local log_file="${LOG_DIR}/AttackASPL_eps${epsilon}_steps${steps}.log"
    cd "${SCRIPT_DIR}/Anti-DreamBooth"
    python AttackASPL.py \
        --source_dir "${SOURCE_DIR}" \
        --target_dir "${TARGET_DIR}" \
        --output_dir "${output_dir}" \
        --sd_model "${SD_MODEL}" \
        --pgd_eps "${pgd_eps}" \
        --pgd_steps "${steps}" \
        --pgd_alpha "${pgd_alpha}" \
        ${EFFECTIVE_MATCH_JSON:+--match_json "${EFFECTIVE_MATCH_JSON}"} \
        2>&1 | tee "${log_file}"

    local t_end
    t_end=$(date +%s)
    local gpu_result
    gpu_result=$(stop_gpu_monitor "${gpu_monitor_file}")
    local peak_mem
    peak_mem=$(echo "${gpu_result}" | awk '{print $1}')
    local baseline_mem
    baseline_mem=$(echo "${gpu_result}" | awk '{print $2}')
    write_resource_log "${LOG_DIR}" "AttackASPL" "${epsilon}" "${steps}" "${t_start}" "${t_end}" "${peak_mem}" "${baseline_mem}"

    echo "[信息] AttackASPL 完成，结果保存至：${output_dir}"
    echo "[信息] 运行日志：${log_file}"
}

# ----------------------------
# run_mmcoa：运行 AttackMMCoA（MMCoA）
# 论文默认：epsilon=1, num_iters=100  (MMCoA/arXiv2404.19287，测试阶段)
# 注意：论文 epsilon=1/255 针对 CLIP 多模态空间，远小于传统图像分类设置
# ----------------------------
run_mmcoa() {
    local epsilon="${1:-1}"
    local steps="${2:-100}"
    local output_dir="${OUTPUT_BASE}/mmcoa_eps${epsilon}_steps${steps}"

    echo ""
    echo "[信息] === AttackMMCoA：epsilon=${epsilon}, num_iters=${steps} ==="
    echo "[信息] 输出目录：${output_dir}"

    if should_skip "${output_dir}"; then
        return 0
    fi

    mkdir -p "${output_dir}"
    activate_env "mmcoa"

    local gpu_monitor_file="${output_dir}/.gpu_peak_mem"
    start_gpu_monitor "${gpu_monitor_file}"
    local t_start
    t_start=$(date +%s)

    mkdir -p "${LOG_DIR}"
    local log_file="${LOG_DIR}/AttackMMCoA_eps${epsilon}_steps${steps}.log"
    cd "${SCRIPT_DIR}/MMCoA"
    python AttackMMCoA.py \
        --source_dir "${SOURCE_DIR}" \
        --target_dir "${TARGET_DIR}" \
        --output_dir "${output_dir}" \
        --epsilon "${epsilon}" \
        --num_iters "${steps}" \
        ${EFFECTIVE_MATCH_JSON:+--match_json "${EFFECTIVE_MATCH_JSON}"} \
        2>&1 | tee "${log_file}"

    local t_end
    t_end=$(date +%s)
    local gpu_result
    gpu_result=$(stop_gpu_monitor "${gpu_monitor_file}")
    local peak_mem
    peak_mem=$(echo "${gpu_result}" | awk '{print $1}')
    local baseline_mem
    baseline_mem=$(echo "${gpu_result}" | awk '{print $2}')
    write_resource_log "${LOG_DIR}" "AttackMMCoA" "${epsilon}" "${steps}" "${t_start}" "${t_end}" "${peak_mem}" "${baseline_mem}"

    echo "[信息] AttackMMCoA 完成，结果保存至：${output_dir}"
    echo "[信息] 运行日志：${log_file}"
}

# ----------------------------
# run_nightshade：运行 AttackNightshade（nightshade-release）
# 论文默认：eps=0.05([0,1]空间，Linf实现), steps=500  (Nightshade/Oakland2024)
# 传入全局 epsilon(0-255) 时换算：ns_eps = epsilon/255
# 不传时直接使用论文值 ns_eps=0.05，不做换算
# ----------------------------
run_nightshade() {
    local epsilon="$1"
    local steps="${2:-500}"
    local ns_eps
    local output_dir

    if [[ -z "${epsilon}" ]]; then
        # 论文默认：直接使用 0.05（[0,1]空间），对应约 12.75/255 像素
        ns_eps="0.050000"
        output_dir="${OUTPUT_BASE}/nightshade_eps0.05_steps${steps}"
    else
        ns_eps="$(calc_eps_0_1 "${epsilon}")"
        output_dir="${OUTPUT_BASE}/nightshade_eps${epsilon}_steps${steps}"
    fi

    echo ""
    echo "[信息] === AttackNightshade：ns_eps=${ns_eps}, steps=${steps} ==="
    echo "[信息] 输出目录：${output_dir}"

    if should_skip "${output_dir}"; then
        return 0
    fi

    mkdir -p "${output_dir}"
    activate_env "nightshade"

    local gpu_monitor_file="${output_dir}/.gpu_peak_mem"
    start_gpu_monitor "${gpu_monitor_file}"
    local t_start
    t_start=$(date +%s)

    mkdir -p "${LOG_DIR}"
    local log_file="${LOG_DIR}/AttackNightshade_eps${epsilon}_steps${steps}.log"
    cd "${SCRIPT_DIR}/nightshade-release"
    python AttackNightshade.py \
        --source_dir "${SOURCE_DIR}" \
        --target_dir "${TARGET_DIR}" \
        --output_dir "${output_dir}" \
        --sd_model "${SD_MODEL}" \
        --eps "${ns_eps}" \
        --steps "${steps}" \
        ${EFFECTIVE_MATCH_JSON:+--match_json "${EFFECTIVE_MATCH_JSON}"} \
        2>&1 | tee "${log_file}"

    local t_end
    t_end=$(date +%s)
    local gpu_result
    gpu_result=$(stop_gpu_monitor "${gpu_monitor_file}")
    local peak_mem
    peak_mem=$(echo "${gpu_result}" | awk '{print $1}')
    local baseline_mem
    baseline_mem=$(echo "${gpu_result}" | awk '{print $2}')
    write_resource_log "${LOG_DIR}" "AttackNightshade" "${epsilon}" "${steps}" "${t_start}" "${t_end}" "${peak_mem}" "${baseline_mem}"

    echo "[信息] AttackNightshade 完成，结果保存至：${output_dir}"
    echo "[信息] 运行日志：${log_file}"
}

# ----------------------------
# run_xtransfer：运行 AttackXTransfer（XTransferBench）
# 论文默认：epsilon=12, steps=300  (XTransferBench/ICML2025, arXiv:2505.05528)
# ----------------------------
run_xtransfer() {
    local epsilon="${1:-12}"
    local steps="${2:-300}"
    local output_dir="${OUTPUT_BASE}/xtransfer_eps${epsilon}_steps${steps}"

    echo ""
    echo "[信息] === AttackXTransfer：epsilon=${epsilon}, steps=${steps} ==="
    echo "[信息] 输出目录：${output_dir}"

    if should_skip "${output_dir}"; then
        return 0
    fi

    mkdir -p "${output_dir}"
    activate_env "xtransfer"

    local gpu_monitor_file="${output_dir}/.gpu_peak_mem"
    start_gpu_monitor "${gpu_monitor_file}"
    local t_start
    t_start=$(date +%s)

    mkdir -p "${LOG_DIR}"
    local log_file="${LOG_DIR}/AttackXTransfer_eps${epsilon}_steps${steps}.log"
    cd "${SCRIPT_DIR}/XTransferBench"
    python AttackXTransfer.py \
        --source_dir "${SOURCE_DIR}" \
        --target_dir "${TARGET_DIR}" \
        --output_dir "${output_dir}" \
        --epsilon "${epsilon}" \
        --steps "${steps}" \
        ${EFFECTIVE_MATCH_JSON:+--match_json "${EFFECTIVE_MATCH_JSON}"} \
        2>&1 | tee "${log_file}"

    local t_end
    t_end=$(date +%s)
    local gpu_result
    gpu_result=$(stop_gpu_monitor "${gpu_monitor_file}")
    local peak_mem
    peak_mem=$(echo "${gpu_result}" | awk '{print $1}')
    local baseline_mem
    baseline_mem=$(echo "${gpu_result}" | awk '{print $2}')
    write_resource_log "${LOG_DIR}" "AttackXTransfer" "${epsilon}" "${steps}" "${t_start}" "${t_end}" "${peak_mem}" "${baseline_mem}"

    echo "[信息] AttackXTransfer 完成，结果保存至：${output_dir}"
    echo "[信息] 运行日志：${log_file}"
}

# ----------------------------
# run_bard：运行 AttackBard（Attack-Bard）
# 论文默认：epsilon=8, steps=300  (AttackVLM/NeurIPS2023, arXiv:2305.16934)
# 特殊：无 --target_dir，使用 --use_json_text 从同名 JSON 读取文本目标
# ----------------------------
run_bard() {
    local epsilon="${1:-8}"
    local steps="${2:-300}"
    local output_dir="${OUTPUT_BASE}/bard_eps${epsilon}_steps${steps}"

    echo ""
    echo "[信息] === AttackBard：epsilon=${epsilon}, steps=${steps} ==="
    echo "[信息] 输出目录：${output_dir}"
    echo "[注意] AttackBard 无 --target_dir 参数，使用 --use_json_text 从 source_dir 同名 JSON 读取文本目标"

    if should_skip "${output_dir}"; then
        return 0
    fi

    mkdir -p "${output_dir}"
    activate_env "attack_bard"

    local gpu_monitor_file="${output_dir}/.gpu_peak_mem"
    start_gpu_monitor "${gpu_monitor_file}"
    local t_start
    t_start=$(date +%s)

    mkdir -p "${LOG_DIR}"
    local log_file="${LOG_DIR}/AttackBard_eps${epsilon}_steps${steps}.log"
    cd "${SCRIPT_DIR}/Attack-Bard"
    python AttackBard.py \
        --source_dir "${SOURCE_DIR}" \
        --output_dir "${output_dir}" \
        --epsilon "${epsilon}" \
        --steps "${steps}" \
        --use_json_text \
        ${EFFECTIVE_MATCH_JSON:+--match_json "${EFFECTIVE_MATCH_JSON}"} \
        2>&1 | tee "${log_file}"

    local t_end
    t_end=$(date +%s)
    local gpu_result
    gpu_result=$(stop_gpu_monitor "${gpu_monitor_file}")
    local peak_mem
    peak_mem=$(echo "${gpu_result}" | awk '{print $1}')
    local baseline_mem
    baseline_mem=$(echo "${gpu_result}" | awk '{print $2}')
    write_resource_log "${LOG_DIR}" "AttackBard" "${epsilon}" "${steps}" "${t_start}" "${t_end}" "${peak_mem}" "${baseline_mem}"

    echo "[信息] AttackBard 完成，结果保存至：${output_dir}"
    echo "[信息] 运行日志：${log_file}"
}

# ----------------------------
# run_attackvlm_new：运行 AttackVLMNew（AttackVLM/LAVIS）
# 论文默认：epsilon=8, steps=300  (参考 AttackVLM/NeurIPS2023)
# 使用 LAVIS 的 BLIP/BLIP2 视觉编码器进行图像-图像特征对齐迁移攻击
# ----------------------------
run_attackvlm_new() {
    local epsilon="${1:-8}"
    local steps="${2:-300}"
    local output_dir="${OUTPUT_BASE}/attackvlm_eps${epsilon}_steps${steps}"

    echo ""
    echo "[信息] === AttackVLMNew：epsilon=${epsilon}, steps=${steps} ==="
    echo "[信息] 输出目录：${output_dir}"

    if should_skip "${output_dir}"; then
        return 0
    fi

    mkdir -p "${output_dir}"
    activate_env "attackvlm"

    local gpu_monitor_file="${output_dir}/.gpu_peak_mem"
    start_gpu_monitor "${gpu_monitor_file}"
    local t_start
    t_start=$(date +%s)

    mkdir -p "${LOG_DIR}"
    local log_file="${LOG_DIR}/AttackVLMNew_eps${epsilon}_steps${steps}.log"
    cd "${SCRIPT_DIR}/AttackVLM"
    python AttackVLMNew.py \
        --source_dir "${SOURCE_DIR}" \
        --target_dir "${TARGET_DIR}" \
        --output_dir "${output_dir}" \
        --epsilon "${epsilon}" \
        --steps "${steps}" \
        ${EFFECTIVE_MATCH_JSON:+--match_json "${EFFECTIVE_MATCH_JSON}"} \
        2>&1 | tee "${log_file}"

    local t_end
    t_end=$(date +%s)
    local gpu_result
    gpu_result=$(stop_gpu_monitor "${gpu_monitor_file}")
    local peak_mem
    peak_mem=$(echo "${gpu_result}" | awk '{print $1}')
    local baseline_mem
    baseline_mem=$(echo "${gpu_result}" | awk '{print $2}')
    write_resource_log "${LOG_DIR}" "AttackVLMNew" "${epsilon}" "${steps}" "${t_start}" "${t_end}" "${peak_mem}" "${baseline_mem}"

    echo "[信息] AttackVLMNew 完成，结果保存至：${output_dir}"
    echo "[信息] 运行日志：${log_file}"
}

# ============================================================
# G区：执行所有攻击的辅助函数
# ============================================================

run_all_attacks_for() {
    local eps="$1"
    local stps="$2"

    if [[ ${SKIP_MI} -eq 0 ]];         then run_mi          "${eps}" "${stps}"; fi
    if [[ ${SKIP_ASPL} -eq 0 ]];       then run_aspl         "${eps}" "${stps}"; fi
    if [[ ${SKIP_MMCOA} -eq 0 ]];      then run_mmcoa        "${eps}" "${stps}"; fi
    if [[ ${SKIP_NIGHTSHADE} -eq 0 ]]; then run_nightshade   "${eps}" "${stps}"; fi
    if [[ ${SKIP_XTRANSFER} -eq 0 ]];  then run_xtransfer    "${eps}" "${stps}"; fi
    if [[ ${SKIP_BARD} -eq 0 ]];       then run_bard         "${eps}" "${stps}"; fi
    if [[ ${SKIP_ATTACKVLM} -eq 0 ]];  then run_attackvlm_new "${eps}" "${stps}"; fi
}

# ============================================================
# H区：正常模式（单次运行）
# ============================================================

run_normal_mode() {
    local eps="${EPSILON}"
    local stps="${STEPS}"

    echo "============================================================"
    if [[ -z "${eps}" ]] && [[ -z "${stps}" ]]; then
        echo "[信息] 正常模式：各方法使用论文默认参数（见 ATTACK_PARAMS.md）"
    else
        echo "[信息] 正常模式：全局覆盖 epsilon=${eps:-各方法默认}, steps=${stps:-各方法默认}"
    fi
    if [[ ${MINI_MODE} -eq 1 ]]; then
        echo "[信息] MINI模式：每个攻击只跑 ${MINI_COUNT} 张图片"
    else
        echo "[信息] 全量模式：运行所有图片"
    fi
    echo "============================================================"

    run_all_attacks_for "${eps}" "${stps}"
}

# ============================================================
# I区：比较模式（grid 或 zip）
# ============================================================

run_comparison_mode() {
    read -ra EPS_LIST  <<< "${COMP_EPSILON}"
    read -ra STEP_LIST <<< "${COMP_STEPS}"

    echo "============================================================"
    echo "[信息] 比较模式：strategy=${COMP_STRATEGY}"
    echo "[信息]   epsilon 列表：${COMP_EPSILON}"
    echo "[信息]   steps 列表：${COMP_STEPS}"
    if [[ ${MINI_MODE} -eq 1 ]]; then
        echo "[信息] MINI模式：每个攻击只跑 ${MINI_COUNT} 张图片"
    else
        echo "[信息] 全量模式：运行所有图片"
    fi
    echo "============================================================"

    # Collect parameter combinations
    local -a combo_eps_arr=()
    local -a combo_steps_arr=()

    if [[ "${COMP_STRATEGY}" == "grid" ]]; then
        for eps in "${EPS_LIST[@]}"; do
            for stps in "${STEP_LIST[@]}"; do
                combo_eps_arr+=("${eps}")
                combo_steps_arr+=("${stps}")
            done
        done
    else
        local count="${#EPS_LIST[@]}"
        if [[ "${#STEP_LIST[@]}" -lt "${count}" ]]; then count="${#STEP_LIST[@]}"; fi
        for (( i=0; i<count; i++ )); do
            combo_eps_arr+=("${EPS_LIST[$i]}")
            combo_steps_arr+=("${STEP_LIST[$i]}")
        done
    fi

    for (( idx=0; idx<${#combo_eps_arr[@]}; idx++ )); do
        local eps="${combo_eps_arr[$idx]}"
        local stps="${combo_steps_arr[$idx]}"
        echo ""
        echo "--- 组合[${idx}]：epsilon=${eps}, steps=${stps} ---"
        run_all_attacks_for "${eps}" "${stps}"
    done
}

# ============================================================
# J区：汇总和主函数
# ============================================================

print_summary() {
    echo ""
    echo "============================================================"
    echo "[信息] 所有攻击任务已完成。"
    echo "[信息] 本次运行目录：${RUN_DIR}"
    echo "[信息]   日志目录：${LOG_DIR}"
    echo "[信息]   图片输出：${OUTPUT_BASE}"
    echo "============================================================"

    # Summarize all resource logs
    local summary_file="${LOG_DIR}/all_resource_summary.txt"
    {
        echo "═══════════════════════════════════════════════════════════════"
        echo "  全部攻击资源使用汇总"
        echo "  生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        if [[ ${MINI_MODE} -eq 1 ]]; then
            echo "  运行模式: MINI（每个攻击 ${MINI_COUNT} 张图片）"
        else
            echo "  运行模式: 全量"
        fi
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        printf "  %-20s  %-8s  %-8s  %-18s  %-14s  %-14s  %-14s\n" "攻击方法" "Epsilon" "Steps" "运行时间" "基准显存(MiB)" "峰值显存(MiB)" "净增显存(MiB)"
        echo "  ──────────────────────────────────────────────────────────────────────────────────────────────────"
        find "${LOG_DIR}" -name "*_resource_log.txt" -type f | sort | while IFS= read -r rlog; do
            local a_name a_eps a_steps a_duration a_baseline a_peak a_net
            a_name=$(grep '攻击方法' "${rlog}" | head -1 | awk -F': ' '{print $2}' | xargs)
            a_eps=$(grep 'Epsilon' "${rlog}" | head -1 | awk -F': ' '{print $2}' | xargs)
            a_steps=$(grep 'Steps' "${rlog}" | head -1 | awk -F': ' '{print $2}' | xargs)
            a_duration=$(grep '总运行时间' "${rlog}" | head -1 | awk -F': ' '{print $2}' | xargs)
            a_baseline=$(grep '基准显存' "${rlog}" | head -1 | awk -F': ' '{print $2}' | xargs)
            a_peak=$(grep '峰值 GPU 显存' "${rlog}" | head -1 | awk -F': ' '{print $2}' | xargs)
            a_net=$(grep '净增显存' "${rlog}" | head -1 | awk -F': ' '{print $2}' | xargs)
            printf "  %-20s  %-8s  %-8s  %-18s  %-14s  %-14s  %-14s\n" "${a_name}" "${a_eps}" "${a_steps}" "${a_duration}" "${a_baseline}" "${a_peak}" "${a_net}"
        done
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
    } > "${summary_file}"
    echo "[信息] 资源使用汇总报告已保存至：${summary_file}"
    cat "${summary_file}"
}

main() {
    echo "[信息] run_all_attacks.sh 启动"
    echo "[信息] 项目根目录：${SCRIPT_DIR}"
    echo "[信息] 源目录：${SOURCE_DIR}"
    echo "[信息] 目标目录：${TARGET_DIR:-（VLM-only 模式，无目标目录）}"
    if [[ ${MINI_MODE} -eq 1 ]]; then
        echo "[信息] 运行模式：⚡ MINI（每个攻击只跑 ${MINI_COUNT} 张图片，用于快速校验）"
    else
        echo "[信息] 运行模式：📦 全量（运行所有图片）"
    fi
    if [[ -n "${MATCH_JSON}" ]]; then
        echo "[信息] 配对文件：${MATCH_JSON}"
    fi
    echo "[信息] SD 模型路径：${SD_MODEL}"

    # Check bc availability
    if command -v bc &>/dev/null; then
        BC_AVAILABLE=1
        echo "[信息] 检测到 bc，将使用 bc 进行浮点计算"
    else
        BC_AVAILABLE=0
        echo "[注意] 未检测到 bc，将使用 awk 进行浮点计算"
    fi

    # Init conda
    init_conda

    # Create run directory with timestamp: outputs/run_<timestamp>/
    local mode_tag=""
    if [[ ${MINI_MODE} -eq 1 ]]; then
        mode_tag="mini"
    else
        mode_tag="full"
    fi
    RUN_DIR="${OUTPUT_ROOT}/run_${mode_tag}_$(date '+%Y%m%d_%H%M%S')"
    LOG_DIR="${RUN_DIR}/log"
    OUTPUT_BASE="${RUN_DIR}/images"

    mkdir -p "${RUN_DIR}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${OUTPUT_BASE}"

    echo "[信息] 本次运行目录：${RUN_DIR}"
    echo "[信息]   日志目录：${LOG_DIR}"
    echo "[信息]   图片输出：${OUTPUT_BASE}"

    # Generate effective match.json for mini mode
    generate_mini_match_json

    # Save run config for reference
    {
        echo "运行配置"
        echo "=========="
        echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "模式: ${mode_tag}"
        if [[ ${MINI_MODE} -eq 1 ]]; then
            echo "mini数量: ${MINI_COUNT}"
        fi
        echo "source_dir: ${SOURCE_DIR}"
        echo "target_dir: ${TARGET_DIR:-N/A}"
        echo "match_json: ${MATCH_JSON:-N/A}"
        echo "effective_match_json: ${EFFECTIVE_MATCH_JSON:-N/A}"
        echo "epsilon: ${EPSILON:-（各方法使用论文默认值）}"
        echo "steps: ${STEPS:-（各方法使用论文默认值）}"
        echo "sd_model: ${SD_MODEL}"
        echo "skip_mi: ${SKIP_MI}"
        echo "skip_aspl: ${SKIP_ASPL}"
        echo "skip_mmcoa: ${SKIP_MMCOA}"
        echo "skip_nightshade: ${SKIP_NIGHTSHADE}"
        echo "skip_xtransfer: ${SKIP_XTRANSFER}"
    echo "skip_bard: ${SKIP_BARD}"
        echo "skip_attackvlm: ${SKIP_ATTACKVLM}"
        if [[ -n "${COMP_EPSILON}" ]]; then
            echo "comp_epsilon: ${COMP_EPSILON}"
            echo "comp_steps: ${COMP_STEPS}"
            echo "comp_strategy: ${COMP_STRATEGY}"
        fi
    } > "${RUN_DIR}/run_config.txt"

    # Choose run mode
    if [[ -n "${COMP_EPSILON}" ]] || [[ -n "${COMP_STEPS}" ]]; then
        if [[ -z "${COMP_EPSILON}" ]] || [[ -z "${COMP_STEPS}" ]]; then
            echo "[错误] 比较模式需要同时指定 --comp_epsilon 和 --comp_steps" >&2
            exit 1
        fi
        run_comparison_mode
    else
        run_normal_mode
    fi

    print_summary
}

main "$@"
