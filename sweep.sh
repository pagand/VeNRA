#!/bin/bash
#SBATCH -J VeNRA_sweep
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%N-%j-%x.out
#SBATCH --nodelist=cs-venus-09
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8

# =============================================================================
# VeNRA Hyperparameter Sweep — Sequential on 1 GPU
#
# Grid being tested (justified below):
#
#  learning_rate : 5e-5, 1e-4
#    → Previous run at 2e-4 peaked at step 100 then degraded (too aggressive).
#      5e-5 is the conservative fix. 1e-4 is a middle ground worth testing.
#
#  lora_rank     : 32, 64
#    → r=64 is spec default. r=32 halves adapter params, faster convergence,
#      often sufficient for classification tasks. Worth confirming.
#
#  warmup_ratio  : 0.05, 0.10
#    → Model showed instability in early steps. More warmup stabilises the
#      label-token decision boundary before full LR kicks in.
#
# Total combinations : 2 × 2 × 2 = 8 runs
# Estimated time     : ~3 hrs/run × 8 = ~24 hrs  (fits in 3-day limit)
#
# Each run saves to its own subdirectory under ./data/output/
# WandB run names encode the config for easy comparison.
# =============================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate .env

hostname
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

cd /localscratch/pagand/VeNRA
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYOPENGL_PLATFORM=egl

nvidia-smi
ulimit -u 1029439

mkdir -p logs

# ---------------------------------------------------------------------------
# Helper: pretty-print the current config before each run
# ---------------------------------------------------------------------------
print_config() {
    echo ""
    echo "============================================================"
    echo "  RUN $RUN_ID / $TOTAL_RUNS"
    echo "  learning_rate : $LR"
    echo "  lora_rank     : $RANK"
    echo "  warmup_ratio  : $WARMUP"
    echo "  output_dir    : $OUT_DIR"
    echo "  run_name      : $RUN_NAME"
    echo "  start time    : $(date)"
    echo "============================================================"
}

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
LEARNING_RATES=(5e-5 1e-4)
LORA_RANKS=(32 64)
WARMUP_RATIOS=(0.05 0.10)

TOTAL_RUNS=$(( ${#LEARNING_RATES[@]} * ${#LORA_RANKS[@]} * ${#WARMUP_RATIOS[@]} ))
RUN_ID=0
FAILED_RUNS=()

for LR in "${LEARNING_RATES[@]}"; do
    for RANK in "${LORA_RANKS[@]}"; do
        for WARMUP in "${WARMUP_RATIOS[@]}"; do

            RUN_ID=$(( RUN_ID + 1 ))

            # Unique name and output dir for this config
            RUN_NAME="venra-lr${LR}-r${RANK}-w${WARMUP}"
            OUT_DIR="./data/output/sweep/${RUN_NAME}"

            print_config

            # Skip if this run already completed (useful if job is requeued)
            if [ -f "${OUT_DIR}/training_complete.flag" ]; then
                echo "  ⏩  Already completed — skipping."
                continue
            fi

            mkdir -p "$OUT_DIR"

            # ------------------------------------------------------------------
            # Launch training
            # Each hyperparameter is passed as a CLI flag so train.py overrides
            # its module-level constants. The --run_name flag is a standard
            # TrainingArguments field and drives the WandB run name.
            # ------------------------------------------------------------------
            srun python -u src/hal_det/training/train.py \
                --output_dir          "$OUT_DIR"          \
                --learning_rate       "$LR"               \
                --lora_rank           "$RANK"             \
                --warmup_ratio        "$WARMUP"           \
                --run_name            "$RUN_NAME"

            EXIT_CODE=$?

            if [ $EXIT_CODE -eq 0 ]; then
                # Mark as complete so reruns skip it
                touch "${OUT_DIR}/training_complete.flag"
                echo "  ✅  Run $RUN_ID completed successfully at $(date)"
            else
                echo "  ❌  Run $RUN_ID FAILED with exit code $EXIT_CODE"
                FAILED_RUNS+=("$RUN_NAME (exit $EXIT_CODE)")
            fi

            # Brief pause between runs to let GPU memory fully clear
            sleep 10

        done
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  SWEEP COMPLETE  ($(date))"
echo "  Total runs    : $TOTAL_RUNS"
echo "  Failed runs   : ${#FAILED_RUNS[@]}"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "  Failed configs:"
    for f in "${FAILED_RUNS[@]}"; do
        echo "    - $f"
    done
fi
echo "============================================================"

# Print best flip_rate_global across all runs by reading wandb summary files
echo ""
echo "--- Per-run best flip_rate_global (from local wandb summary) ---"
for LR in "${LEARNING_RATES[@]}"; do
    for RANK in "${LORA_RANKS[@]}"; do
        for WARMUP in "${WARMUP_RATIOS[@]}"; do
            RUN_NAME="venra-lr${LR}-r${RANK}-w${WARMUP}"
            OUT_DIR="./data/output/sweep/${RUN_NAME}"
            SUMMARY=$(find "$OUT_DIR" -name "wandb-summary.json" 2>/dev/null | head -1)
            if [ -n "$SUMMARY" ]; then
                FLIP=$(python3 -c "
import json, sys
try:
    d = json.load(open('$SUMMARY'))
    print(f\"{d.get('eval_audit/flip_rate_global', 'N/A'):.4f}\")
except Exception as e:
    print('N/A')
")
                echo "  $RUN_NAME  →  flip_rate_global = $FLIP"
            else
                echo "  $RUN_NAME  →  no summary found"
            fi
        done
    done
done
echo "================================================================"