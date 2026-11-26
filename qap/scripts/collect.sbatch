#!/bin/bash
# Week 1 CPU data collection as a Slurm job (no nohup needed). From the repository root:
#     sbatch -J qap_mp   --export=ALL,WHAT=mp   qap/scripts/collect.sbatch   # 1.4 robot episodes as object-centric states
#     sbatch -J qap_play --export=ALL,WHAT=play qap/scripts/collect.sbatch   # 1.5 robot play (~1M transitions)
# Both are resumable: re-submitting continues where the previous job stopped.
# Needs qap/data/splits.json (held-out L3 scenes are skipped). Log: <job-name>_<jobid>.out, e.g. qap_mp_67500.out
# Progress:  grep '^{' qap_mp_<jobid>.out | tail      (one JSON line per finished scene/job)
#
#SBATCH --job-name=qap_collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
##SBATCH --nodelist=worker-0

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"
if command -v conda >/dev/null 2>&1; then conda deactivate 2>/dev/null || true; fi
source .venv/bin/activate
unset MS_ASSET_DIR
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export QAP_DATA_ROOT=${QAP_DATA_ROOT:-$PWD/qap_runs}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1      # many simulator processes: no thread oversubscription
WHAT=${WHAT:?pass --export=ALL,WHAT=mp or WHAT=play}
PROCS=${PROCS:-$(( ${SLURM_CPUS_PER_TASK:-32} - 1 ))}

[[ -f qap/data/splits.json ]] || { echo "qap/data/splits.json missing: run the freeze (screen_tasks) first"; exit 1; }
echo "$WHAT on $(hostname), job ${SLURM_JOB_ID:-}, $PROCS processes, data root $QAP_DATA_ROOT  $(date)"
start=$(date +%s)
case "$WHAT" in
  mp)   python -m qap.collect.collect_mp_states --corpus 45 --levels L0 L1 L2 L3 --n-per-scene 50 --procs "$PROCS" ;;
  play) python -m qap.collect.play --target-transitions 1000000 --procs "$PROCS" ;;
  *)    echo "unknown WHAT=$WHAT (use mp or play)"; exit 1 ;;
esac
echo "$WHAT finished in $(( ($(date +%s) - start) / 60 )) min  $(date)"
