#!/bin/bash
# 一键运行：从任意目录执行蒸馏
cd "$(dirname "$0")/.." || exit 1
exec python3 -m zhengliu.distill "$@"
