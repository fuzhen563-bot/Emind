#!/bin/bash
#
# Emind AI 一键启动脚本
# 用法:
#   ./start.sh                     # Web 服务 (默认)
#   ./start.sh --vllm --model ./models/emind-4b  # vLLM Server
#   ./start.sh --port 8080 --backend cloud_api
#   ./start.sh --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=3333
BACKEND=""
VLLM=false
MODEL=""
VLLM_PORT=8000
VLLM_TP=1
VLLM_DTYPE=auto
VLLM_PREFIX_CACHING=true
API_KEY="${EMIND_API_KEY:-}"

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "Web 服务选项:"
    echo "  --port PORT         服务端口 (默认: 3333)"
    echo "  --backend BACKEND   推理后端 (cloud_api/ollama/local)"
    echo "  --api-key KEY       API 密钥"
    echo ""
    echo "vLLM Server 选项:"
    echo "  --vllm              启动 vLLM Server (替代 Web 服务)"
    echo "  --model PATH        模型路径"
    echo "  --vllm-port PORT    vLLM 端口 (默认: 8000)"
    echo "  --vllm-tp N         Tensor Parallel 大小 (默认: 1)"
    echo "  --vllm-dtype TYPE   数据类型 (auto/float16/bfloat16/fp8)"
    echo "  --no-prefix-caching 禁用 Prefix Caching"
    echo ""
    echo "环境变量:"
    echo "  EMIND_API_KEY      亦API 密钥"
    echo "  EMIND_LOG_LEVEL    日志级别 (debug/info/warning/error)"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage ;;
        --port) PORT="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --vllm) VLLM=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --vllm-port) VLLM_PORT="$2"; shift 2 ;;
        --vllm-tp) VLLM_TP="$2"; shift 2 ;;
        --vllm-dtype) VLLM_DTYPE="$2"; shift 2 ;;
        --no-prefix-caching) VLLM_PREFIX_CACHING=false; shift ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "错误: 未找到 Python 解释器 (尝试安装 python3)"
    exit 1
fi

export EMIND_API_KEY="$API_KEY"

echo "========================================"
echo "  亦梓·智脑 Emind AI v2.0"
echo "  亦梓科技 © 2026"
echo "========================================"
echo ""

if [ "$VLLM" = true ]; then
    if [ -z "$MODEL" ]; then
        echo "错误: --vllm 模式需要 --model 参数"
        exit 1
    fi
    echo "模式: vLLM Server"
    echo "模型: $MODEL"
    echo "端口: $VLLM_PORT"
    echo "TP: $VLLM_TP | dtype: $VLLM_DTYPE"
    echo "Prefix Caching: $VLLM_PREFIX_CACHING"
    echo ""

    ARGS=(serve --vllm --model "$MODEL" --vllm-port "$VLLM_PORT" --vllm-tp "$VLLM_TP" --vllm-dtype "$VLLM_DTYPE")
    [ "$VLLM_PREFIX_CACHING" = true ] && ARGS+=(--vllm-prefix-caching)
    [ -n "$API_KEY" ] && ARGS+=(--api-key "$API_KEY")

    exec $PYTHON cli.py "${ARGS[@]}"
else
    echo "模式: Web 服务"
    echo "端口: $PORT"
    [ -n "$BACKEND" ] && echo "后端: $BACKEND"
    echo ""

    ARGS=(serve --port "$PORT")
    [ -n "$BACKEND" ] && ARGS+=(--backend "$BACKEND")

    exec $PYTHON cli.py "${ARGS[@]}"
fi
