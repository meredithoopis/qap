#!/bin/bash
# Shortcut filter (Week 1, step 1.3): one Slurm array task per retained task, 1 GPU each.
# Each array task trains IG's diffusion policy on L0-L2 of its task, then evaluates it on L0 and L3.
#
# From the repository root:
#     python -m qap.week1.shortcut_filter prepare        # once: writes configs + per-task scripts
#     sbatch qap/scripts/shortcut.sbatch                 # all 13 tasks, at most 8 at a time (edit %8 to match free GPUs)
#     sbatch --array=3,7 qap/scripts/shortcut.sbatch     # re-run selected tasks (line numbers of tasks.txt)
# Order: test and calibration tasks are lines 1-8 of tasks.txt, so they run first.
# Logs: shortcut_<jobid>_<arrayindex>.out in the repository root.
# Resumable: a task whose final_model.pt exists skips training; finished evaluations are skipped.
# When every task is done:  python -m qap.week1.shortcut_filter summarize
#
#SBATCH --job-name=qap_shortcut
#SBATCH --array=1-13%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=shortcut_%A_%a.out
#SBATCH --error=shortcut_%A_%a.out
# Uncomment / edit if your cluster needs them:
##SBATCH --partition=<gpu-partition>
##SBATCH --account=<account>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"
if command -v conda >/dev/null 2>&1; then conda deactivate 2>/dev/null || true; fi
source .venv/bin/activate
unset MS_ASSET_DIR
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export QAP_DATA_ROOT=${QAP_DATA_ROOT:-$PWD/qap_runs}
export PYTHONUNBUFFERED=1

LIST="$QAP_DATA_ROOT/shortcut/tasks.txt"
[[ -f "$LIST" ]] || { echo "missing $LIST: run 'python -m qap.week1.shortcut_filter prepare' first"; exit 1; }
TASK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[[ -n "$TASK" ]] || { echo "no task on line ${SLURM_ARRAY_TASK_ID} of $LIST"; exit 1; }

echo "array ${SLURM_ARRAY_TASK_ID}: $TASK  host $(hostname)  gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
start=$(date +%s)
bash "$QAP_DATA_ROOT/shortcut/tasks/$TASK.sh"
echo "$TASK wall time: $(( ($(date +%s) - start) / 60 )) min"
