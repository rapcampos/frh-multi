#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p outputs .scratch/long-generation-queue

printf '%s\n' "$$" > .scratch/long-generation-queue/queue.pid
printf '%s\n' "running" > .scratch/long-generation-queue/status
finish_queue() {
    exit_code=$?
    printf '%s\n' "$exit_code" > .scratch/long-generation-queue/exit_code
    printf '%s\n' "finished" > .scratch/long-generation-queue/status
}
trap finish_queue EXIT

exec 9> .scratch/long-generation-queue/queue.lock
if ! flock -n 9; then
    echo "A long-generation queue already holds the repository lock." >&2
    exit 1
fi

physical_gpu="${FRH_GPU_INDEX:-2}"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED="1"
export TOKENIZERS_PARALLELISM="false"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/frh-uv-cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${FRH_MODEL_MAX_MEMORY:-}" ]]; then
    export FRH_MODEL_MAX_MEMORY
fi
export FRH_MODEL_DEVICE_MAP="${FRH_MODEL_DEVICE_MAP:-auto}"

completed_53=0
if [[ -f resources/53_long_composition.jsonl ]]; then
    completed_53="$(wc -l < resources/53_long_composition.jsonl)"
fi
if [[ "$completed_53" -eq 450 ]]; then
    echo "$(date --iso-8601=seconds) experiment 53 already has 450 records; skipping"
else
    echo "$(date --iso-8601=seconds) starting experiment 53 on physical GPU $physical_gpu"
    uv run papermill \
        53_long_generation_composition.ipynb \
        outputs/53_long_generation_composition_full.ipynb \
        -p RUN_GENERATION True \
        -p SMOKE False \
        -p OVERWRITE False \
        -p CONFIRM_FULL_RUN RUN_53_450_CELLS \
        -p RUN_BENCHMARK False \
        -p BENCHMARK_ONLY False \
        -p RUN_TRAJECTORIES True \
        --log-output \
        --progress-bar
fi

echo "$(date --iso-8601=seconds) experiment 53 complete; starting experiment 54"
uv run papermill \
    54_long_generation_negative.ipynb \
    outputs/54_long_generation_negative_full.ipynb \
    -p RUN_GENERATION True \
    -p SMOKE False \
    -p OVERWRITE False \
    -p CONFIRM_FULL_RUN RUN_54_60_CELLS \
    -p RUN_TRAJECTORIES True \
    --log-output \
    --progress-bar

echo "$(date --iso-8601=seconds) experiment 54 complete; queue finished"
