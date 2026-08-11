#!/usr/bin/env bash
set -Eeuo pipefail

# MiniChat Linux 一键训练脚本。
# 正常用法：bash start.sh
# 可通过环境变量指定 Python：PYTHON_BIN=/path/to/python bash start.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${LOG_DIR}/train.pid"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${LOG_DIR}"

is_running() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

safe_remove_generated_dir() {
    local relative_path="$1"
    local target="${SCRIPT_DIR}/${relative_path}"

    # 只允许删除项目目录内明确列出的生成目录。
    case "${relative_path}" in
        out/bbpe_tokenizer|out/tokenized_data|model_weights) ;;
        *)
            echo "拒绝清理未授权路径：${relative_path}" >&2
            return 1
            ;;
    esac

    rm -rf -- "${target}"
}

run_stage() {
    local stage_name="$1"
    local log_file="$2"
    shift 2

    echo "[$(date '+%F %T')] 开始阶段：${stage_name}"
    if "$@" >"${log_file}" 2>&1; then
        echo "[$(date '+%F %T')] 完成阶段：${stage_name}（日志：${log_file}）"
    else
        local exit_code=$?
        echo "[$(date '+%F %T')] 阶段失败：${stage_name}，退出码 ${exit_code}（日志：${log_file}）" >&2
        return "${exit_code}"
    fi
}

worker_main() {
    local run_id="$1"
    local cleanup_log="${LOG_DIR}/cleanup_${run_id}.log"
    local token_train_log="${LOG_DIR}/token_train_${run_id}.log"
    local tokenizer_log="${LOG_DIR}/tokenizer_${run_id}.log"
    local train_log="${LOG_DIR}/train_${run_id}.log"

    cd "${SCRIPT_DIR}"
    trap 'rm -f -- "${PID_FILE}"' EXIT

    {
        echo "[$(date '+%F %T')] 清理训练生成物"
        safe_remove_generated_dir "out/bbpe_tokenizer"
        safe_remove_generated_dir "out/tokenized_data"
        safe_remove_generated_dir "model_weights"
        mkdir -p "${SCRIPT_DIR}/out"
        echo "[$(date '+%F %T')] 清理完成"
    } >"${cleanup_log}" 2>&1

    run_stage "token train" "${token_train_log}" "${PYTHON_BIN}" -u "${SCRIPT_DIR}/token_train.py"
    run_stage "tokenizer" "${tokenizer_log}" "${PYTHON_BIN}" -u "${SCRIPT_DIR}/tokenizer.py"
    run_stage "model train" "${train_log}" "${PYTHON_BIN}" -u "${SCRIPT_DIR}/train.py"

    echo "[$(date '+%F %T')] 全部训练阶段执行完成"
}

if [[ "${1:-}" == "--worker" ]]; then
    worker_main "${2:?缺少 run_id}"
    exit 0
fi

if is_running; then
    echo "训练任务已在运行，PID：$(cat "${PID_FILE}")"
    echo "可查看日志目录：${LOG_DIR}"
    exit 1
fi

# 清理上一次异常退出遗留的无效 PID 文件。
rm -f -- "${PID_FILE}"

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
PIPELINE_LOG="${LOG_DIR}/pipeline_${RUN_ID}.log"

nohup bash "${SCRIPT_DIR}/start.sh" --worker "${RUN_ID}" \
    >"${PIPELINE_LOG}" 2>&1 </dev/null &
BACKGROUND_PID=$!
echo "${BACKGROUND_PID}" >"${PID_FILE}"

# 在交互式 shell 中解除作业关联；非交互式 shell 下失败也不影响 nohup。
disown "${BACKGROUND_PID}" 2>/dev/null || true

echo "训练任务已在后台启动，PID：${BACKGROUND_PID}"
echo "关闭 SSH 窗口不会终止任务。"
echo "流水线日志：${PIPELINE_LOG}"
echo "阶段日志目录：${LOG_DIR}"
echo "查看进度：tail -f '${PIPELINE_LOG}'"
