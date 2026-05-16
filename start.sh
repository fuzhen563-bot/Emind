#!/bin/bash
#
# Emind AI 一键启动脚本
# 用法: ./start.sh [--port 8080] [--backend auto]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${2:-3333}
BACKEND=${4:-auto}

echo "========================================"
echo "  亦梓·智脑 Emind AI v2.0"
echo "  亦梓科技 © 2026"
echo "========================================"
echo ""
echo "后端模式: $BACKEND"
echo "服务端口: $PORT"
echo ""

# 自动检测 Python
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "错误: 未找到 Python"
    exit 1
fi

# 启动服务
EMIND_LOG_LEVEL=info $PYTHON cli.py serve --port "$PORT"
