#!/usr/bin/env bash
# Run from 方案H初版. No instance creation, payment, credential handling, or auto-renewal.
set -euo pipefail
command_name="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
case "$command_name" in
  setup)
    uv sync --frozen --extra neural
    .venv/bin/python -c 'import torch; print({"torch":torch.__version__,"cuda":torch.version.cuda,"available":torch.cuda.is_available()}); assert torch.cuda.is_available(), "CUDA unavailable: inspect driver/wheel compatibility before training"'
    ;;
  download)
    : "${DATA_ROOT:?Set a NEW DATA_ROOT destination}"
    .venv/bin/python -m obsplan.training_data.download "$DATA_ROOT" \
      --index datasets/synthetic-v0.2/release-assets.json \
      --cache "${DOWNLOAD_CACHE:-outputs/v02-download-cache}"
    ;;
  verify)
    : "${DATA_ROOT:?Set DATA_ROOT}" "${RUN_ROOT:?Set a NEW RUN_ROOT}"
    .venv/bin/python scripts/neural_verify_data.py --root "$DATA_ROOT" \
      --index datasets/synthetic-v0.2/release-assets.json --output "$RUN_ROOT/data-hashes.json"
    .venv/bin/python -m obsplan.neural_lns.protocol verify experiments/neural-lns-v1
    ;;
  train|resume|collect|calibrate|evaluate)
    : "${WALL_SECONDS:?Set the explicitly authorized finite wall-time cap}"
    if ! [[ "$WALL_SECONDS" =~ ^[0-9]+$ ]] || [ "$WALL_SECONDS" -lt 1 ]; then
      echo 'WALL_SECONDS must be a positive integer' >&2; exit 2
    fi
    command -v timeout >/dev/null || { echo 'Linux GNU timeout is required for paid runs' >&2; exit 2; }
    case "$command_name" in
      train|resume) module=obsplan.neural_lns.training ;;
      collect) module=obsplan.neural_lns.collect ;;
      calibrate|evaluate) module=obsplan.neural_lns.campaign; set -- "$command_name" "$@" ;;
    esac
    timeout --signal=INT --kill-after=30s "${WALL_SECONDS}s" .venv/bin/python -m "$module" "$@"
    ;;
  *)
    echo 'Commands: setup | download | verify | train | resume | collect | calibrate | evaluate'
    echo 'train/resume take training CLI arguments, including stage static/dynamic and --config/--output.'
    echo 'Every paid job requires WALL_SECONDS; credentials belong in SSH config, never here.'
    ;;
esac
