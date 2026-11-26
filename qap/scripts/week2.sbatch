#!/bin/bash
# Week 2 as Slurm jobs, one stage per job (QOS: one running job per user, <=16 CPUs per job).
# Submit from the repository root, in this order (see RUN_GUIDE.md, Week 2):
#
#   validate / graded / goal use CPU simulation but are run with a GPU for the renderer
#   (CPU-only via --gres=none is possible but untested on IG scenes):
#     sbatch -J w2_validate --export=ALL,STAGE=validate,QAP_RENDER_BACKEND= qap/scripts/week2.sbatch
#     sbatch -J w2_graded   --export=ALL,STAGE=graded,QAP_RENDER_BACKEND= qap/scripts/week2.sbatch
#   1 GPU:
#     sbatch -J w2_labels   --export=ALL,STAGE=labels                 qap/scripts/week2.sbatch
#     sbatch -J w2_qrl      --export=ALL,STAGE=train,RUN=qrl_iqe      qap/scripts/week2.sbatch
#     sbatch -J w2_tmd      --export=ALL,STAGE=train,RUN=tmd_iqe      qap/scripts/week2.sbatch
#     sbatch -J w2_prauc    --export=ALL,STAGE=prauc                  qap/scripts/week2.sbatch
#     sbatch -J w2_mrn      --export=ALL,STAGE=train,RUN=qrl_mrn      qap/scripts/week2.sbatch   # optional
#     sbatch -J w2_hilp     --export=ALL,STAGE=train,RUN=hilp_l2      qap/scripts/week2.sbatch   # optional
#     sbatch -J w2_awr      --export=ALL,STAGE=awr                    qap/scripts/week2.sbatch   # DIST=tmd_iqe if TMD won
#   (GPU as above):
#     sbatch -J w2_goal     --export=ALL,STAGE=goal,QAP_RENDER_BACKEND= qap/scripts/week2.sbatch
#   Fallback if the PR-AUC go/no-go fails:
#     sbatch -J w2_distill  --export=ALL,STAGE=train,RUN=distill_iqe  qap/scripts/week2.sbatch
#
# Log: <job-name>_<jobid>.out. Chain jobs without watching: --dependency=afterok:<previous jobid>
#
#SBATCH --job-name=qap_week2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
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
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
# Per planner trial wall-clock limit. G4 trials never succeed and run into it; normal trials take
# <50 s (validate, job 67666), so 120 s instead of the 300 s default saves hours on G4.
export QAP_TRIAL_TIMEOUT=${QAP_TRIAL_TIMEOUT:-120}
STAGE=${STAGE:?pass --export=ALL,STAGE=validate|graded|labels|train|prauc|awr|goal}
PROCS=${PROCS:-$(( ${SLURM_CPUS_PER_TASK:-16} - 1 ))}
R=$QAP_DATA_ROOT/runs
E=$QAP_DATA_ROOT/episodes

have_gpu() { [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; }
# Without a GPU, SAPIEN's Vulkan renderer fails to start; state-only stages do not need it.
have_gpu || export QAP_RENDER_BACKEND=${QAP_RENDER_BACKEND:-none}
need_gpu() { have_gpu || { echo "stage $STAGE needs a GPU: submit without --gres=none"; exit 1; }; }

[[ -f qap/data/frozen_tasks.json ]] || { echo "qap/data/frozen_tasks.json missing"; exit 1; }
grep -q retained_after_shortcut qap/data/frozen_tasks.json || \
  echo "WARNING: shortcut results not applied (run screen_tasks --apply-shortcut-results first)"
echo "$STAGE on $(hostname), job ${SLURM_JOB_ID:-}, $PROCS processes, GPU=${CUDA_VISIBLE_DEVICES:-none}, data root $QAP_DATA_ROOT  $(date)"
start=$(date +%s)

# name -> training flags
train_flags() {
  case "$1" in
    qrl_iqe)     echo "--objective qrl     --head iqe" ;;
    tmd_iqe)     echo "--objective tmd     --head iqe" ;;
    qrl_mrn)     echo "--objective qrl     --head mrn" ;;
    hilp_l2)     echo "--objective hilp    --head l2"  ;;
    distill_iqe) echo "--objective distill --head iqe" ;;
    *) echo "unknown RUN=$1 (qrl_iqe tmd_iqe qrl_mrn hilp_l2 distill_iqe)" >&2; exit 1 ;;
  esac
}

case "$STAGE" in
  validate)   # 2.1a: does every graded scene build and behave? (read the table before going on)
    python -m qap.scenes.graded_runner validate --trials 5 --procs "$PROCS" ;;
  graded)     # 2.1b: graded robot episodes + planner feasibility labels
    python -m qap.scenes.graded_runner episodes --n-per-config 50 --procs "$PROCS"
    python -m qap.scenes.graded_runner feasibility --trials 10 --procs "$PROCS" ;;
  labels)     # 2.2: planner-certified (s, a, g) triples with 224x224 frames for DINOv2
    need_gpu
    python -m qap.feasibility.label_triples --seeds 20 --images --procs "$PROCS" ;;
  train)      # 2.3: one distance model (5-member ensemble) per job
    need_gpu
    RUN=${RUN:?pass RUN=qrl_iqe|tmd_iqe|qrl_mrn|hilp_l2|distill_iqe}
    if [[ "$RUN" == distill_iqe ]]; then DATA=("$E/mp" "$E/graded"); else DATA=("$E/mp" "$E/play"); fi
    if [[ -d "$E/graded" && "$RUN" != distill_iqe ]]; then DATA+=("$E/graded"); fi   # held-out G2-G4 are filtered out
    # shellcheck disable=SC2046
    python -m qap.quasimetric.train $(train_flags "$RUN") --ensemble 5 --run-name "${RUN}_k5" --data "${DATA[@]}" ;;
  prauc)      # 2.4: PR-AUC of every trained distance vs DINOv2; selects QRL vs TMD on calibration
    need_gpu
    CK=()
    for n in qrl_iqe tmd_iqe qrl_mrn hilp_l2 distill_iqe; do
      if [[ -f "$R/${n}_k5/ckpt.pt" ]]; then CK+=("$n=$R/${n}_k5/ckpt.pt"); fi
    done
    [[ ${#CK[@]} -gt 0 ]] || { echo "no checkpoints under $R"; exit 1; }
    SEL=()
    if [[ -f "$R/qrl_iqe_k5/ckpt.pt" && -f "$R/tmd_iqe_k5/ckpt.pt" ]]; then SEL=(--select qrl_iqe tmd_iqe); fi
    python -m qap.feasibility.prauc --visual ${SEL[@]+"${SEL[@]}"} --ckpt "${CK[@]}" ;;
  awr)        # 2.5a: policy extraction with the selected distance
    need_gpu
    DIST=${DIST:-qrl_iqe}
    python -m qap.policy.awr --dist-ckpt "$R/${DIST}_k5/ckpt.pt" --run-name awr_main ;;
  goal)       # 2.5b: oracle-goal reaching (policy runs on CPU inside the workers)
    python -m qap.policy.eval_goal_reaching --policy "$R/awr_main/policy.pt" --grades G0 G1 G3 --seeds 10 --procs "$PROCS" ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
echo "$STAGE finished in $(( ($(date +%s) - start) / 60 )) min  $(date)"
