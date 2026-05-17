#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEACHER_BACKEND="cloud_api"
TEACHER_MODEL="deepseek-v4-flash"
TEACHER_BASE_URL="https://api.deepseek.com"
API_KEY="${EMIND_API_KEY:-}"

CODE=3000
REASONING=500
DEEP_REASONING=1000
ANTI_HALLUCINATION=1500
IDENTITY=100
OUTPUT_DIR="data/distilled/emind_code_4b"
WORKERS=10
MAX_TOKENS=2048
TEMPERATURE=0.7

while [ $# -gt 0 ]; do
    case "$1" in
        --code) CODE="$2"; shift 2 ;;
        --reasoning) REASONING="$2"; shift 2 ;;
        --deep-reasoning) DEEP_REASONING="$2"; shift 2 ;;
        --anti-hallucination) ANTI_HALLUCINATION="$2"; shift 2 ;;
        --identity) IDENTITY="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --model) TEACHER_MODEL="$2"; shift 2 ;;
        --base-url) TEACHER_BASE_URL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "  --code NUM              Code samples (default: 3000)"
            echo "  --reasoning NUM         Reasoning samples (default: 500)"
            echo "  --deep-reasoning NUM    Deep reasoning samples (default: 1000)"
            echo "  --anti-hallucination NUM Anti-hallucination samples (default: 1500)"
            echo "  --identity NUM          Identity samples (default: 100)"
            echo "  --api-key KEY           DeepSeek API Key"
            echo "  --model NAME            Teacher model (default: deepseek-v4-flash)"
            echo "  --base-url URL          API URL (default: https://api.deepseek.com)"
            echo "  --max-tokens NUM        Max new tokens (default: 2048)"
            echo "  --temperature TEMP      Temperature (default: 0.7)"
            echo "  --workers NUM           Concurrent workers (default: 10)"
            echo "  --output DIR            Output directory"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$API_KEY" ]; then
    echo "Error: EMIND_API_KEY not set. Use --api-key or export EMIND_API_KEY."
    exit 1
fi

echo "=========================================="
echo " Emind Data Distillation"
echo " Teacher: $TEACHER_MODEL"
echo " Code: $CODE | Reasoning: $REASONING | Deep: $DEEP_REASONING"
echo " Anti-Hallucination: $ANTI_HALLUCINATION | Identity: $IDENTITY"
echo " Workers: $WORKERS | Output: $OUTPUT_DIR"
echo "=========================================="

python -u cli.py pipeline \
    --distill-code "$CODE" \
    --distill-reasoning "$REASONING" \
    --distill-deep-reasoning "$DEEP_REASONING" \
    --distill-anti-hallucination "$ANTI_HALLUCINATION" \
    --distill-identity "$IDENTITY" \
    --teacher-backend "$TEACHER_BACKEND" \
    --teacher-api-key "$API_KEY" \
    --teacher-model "$TEACHER_MODEL" \
    --teacher-base-url "$TEACHER_BASE_URL" \
    --distill-output "$OUTPUT_DIR" \
    --max-new-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE"

echo "Done! Output: $OUTPUT_DIR/distilled_sft.jsonl"
