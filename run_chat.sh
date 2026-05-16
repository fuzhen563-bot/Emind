#!/bin/bash
#
# Emind AI 交互式对话
# 用法: ./run_chat.sh [--backend cloud_api]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BACKEND=${2:-auto}

echo "========================================"
echo "  亦梓·智脑 Emind AI — 交互式对话"
echo "========================================"
echo ""

PYTHON=$(command -v python3 || command -v python)
$PYTHON cli.py infer --backend "$BACKEND" --interactive
