#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=outputs/logs/%j.out
#SBATCH --error=outputs/logs/%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=0-10:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRAFTER="peerrh/treeflash-qwen3-4b"         # DFlash or TreeFlash model id / local checkpoint path
TARGET="Qwen/Qwen3-4B"
TREE_SIZE="64"
VERIFIER_TEMPERATURE="0.0"
QUALITY_DATASETS=(humaneval mbpp) # gsm8k, math500, humaneval, mbpp, mt-bench
TOP_M="16"
MAX_NEW_TOKENS="2048"
N_SAMPLES="64"

python "$SCRIPT_DIR/benchmark.py" \
    --drafter "$DRAFTER" \
    --target "$TARGET" \
    --tree-size "$TREE_SIZE" \
    --verifier-temperature "$VERIFIER_TEMPERATURE" \
    --quality-datasets "${QUALITY_DATASETS[@]}" \
    --top-m "$TOP_M" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --n-samples "$N_SAMPLES" \
    --output-dir "./results/benchmark" \
    --compute-speedup

# Add this flag above for a DFlash drafter:
#     --is_chain \
