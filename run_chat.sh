#!/bin/bash
#
# Emind AI 交互式对话
# 用法:
#   ./run_chat.sh                                         # 云端推理
#   ./run_chat.sh --backend vllm --model ./models/emind-4b  # vLLM 本地
#   ./run_chat.sh --backend ollama                          # Ollama 本地
#   ./run_chat.sh --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BACKEND="cloud_api"
MODEL=""
API_KEY="${EMIND_API_KEY:-}"
VLLM_PREFIX_CACHING=true
VLLM_TP=1
VLLM_DTYPE=auto

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "  --backend BACKEND   推理后端 (cloud_api/vllm/ollama/llama_cpp/huggingface)"
    echo "  --model PATH        模型路径 (vLLM 模式必需)"
    echo "  --api-key KEY       API 密钥"
    echo "  --vllm-tp N         Tensor Parallel 大小 (默认: 1)"
    echo "  --vllm-dtype TYPE   数据类型 (默认: auto)"
    echo "  --no-prefix-caching 禁用 Prefix Caching"
    echo "  --help              显示此帮助"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --vllm-tp) VLLM_TP="$2"; shift 2 ;;
        --vllm-dtype) VLLM_DTYPE="$2"; shift 2 ;;
        --no-prefix-caching) VLLM_PREFIX_CACHING=false; shift ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "错误: 未找到 Python 解释器"
    exit 1
fi

export EMIND_API_KEY="$API_KEY"

echo "========================================"
echo "  亦梓·智脑 Emind AI — 交互式对话"
echo "  后端: $BACKEND"
[ -n "$MODEL" ] && echo "  模型: $MODEL"
echo "========================================"
echo ""

ARGS=(infer --backend "$BACKEND" --interactive)
[ -n "$MODEL" ] && ARGS+=(--model "$MODEL")
[ "$VLLM_PREFIX_CACHING" = true ] && ARGS+=(--vllm-prefix-caching)
ARGS+=(--vllm-tp "$VLLM_TP" --vllm-dtype "$VLLM_DTYPE")

exec $PYTHON cli.py "${ARGS[@]}"
