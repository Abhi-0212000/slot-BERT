#!/usr/bin/env bash
# ============================================================================
# train.sh — Slot-BERT Training Script
# ============================================================================
# Starts Visdom server in background, then runs training.
# Designed to survive SSH disconnects via detached mode.
#
# Usage:
#   # Interactive (inside container):
#   docker compose run --rm train bash
#   bash scripts/train.sh --mode train_cholec
#
#   # Detached (survives SSH disconnect):
#   docker compose run -d --name qte9489-slotbert-train train bash scripts/train.sh --mode train_cholec
#
#   # Check logs (live):
#   docker logs -f qte9489-slotbert-train
#
#   # Check logs (last 50 lines):
#   docker logs --tail 50 qte9489-slotbert-train
#
#   # Check if still running:
#   docker ps | grep qte9489-slotbert
#
#   # Stop:
#   docker stop qte9489-slotbert-train
#   docker rm qte9489-slotbert-train
#
# Outputs (all persisted on host via bind mounts):
#   - Checkpoints:  ./checkpoints/
#   - Visdom data:  ./visdom_data/
#   - Outputs:      ./outputs/
#   - Visdom UI:    http://<hostname>:8097
# ============================================================================
set -euo pipefail

# ── Parse args ──────────────────────────────────────────────────────────────
MODE="${2:-train_cholec}"  # default: train_cholec

# Parse --mode from any position
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ── Config ──────────────────────────────────────────────────────────────────
VISDOM_PORT=8097

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Slot-BERT Training                                        ║"
echo "║  Mode:   ${MODE}                                           "
echo "║  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "║  Visdom: http://0.0.0.0:${VISDOM_PORT}                     "
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Start Visdom server in background ───────────────────────────────────────
echo "[1/2] Starting Visdom server on port ${VISDOM_PORT}..."
python -m visdom.server -port ${VISDOM_PORT} --hostname 0.0.0.0 &
VISDOM_PID=$!

# Wait for Visdom to be ready
for i in $(seq 1 10); do
    if curl -s "http://localhost:${VISDOM_PORT}" > /dev/null 2>&1; then
        echo "      Visdom ready (PID: ${VISDOM_PID})"
        break
    fi
    sleep 1
done

# ── Run training ────────────────────────────────────────────────────────────
echo "[2/2] Starting training (mode: ${MODE})..."
echo ""

python main.py --mode "${MODE}"

echo ""
echo "✓ Training complete (mode: ${MODE})"
echo "  Checkpoints saved to: /app/src/Model_checkpoint/"
