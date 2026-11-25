#!/bin/bash
# Week 2 pipeline (run from the repository root after Week-1 data collection finished).
#   STAGE=graded|train|labels|prauc|policy  bash qap/scripts/week2.sh
set -euo pipefail
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export QAP_DATA_ROOT=${QAP_DATA_ROOT:-$PWD/qap_runs}
PROCS=${PROCS:-32}
# GPU for the main ensemble / AWR and for the variant runs. Default: both on GPU 0 (one H100
# holds the main run and the variants side by side). With two GPUs: GPU_AUX=1.
GPU_MAIN=${GPU_MAIN:-0}
GPU_AUX=${GPU_AUX:-0}
STAGE=${STAGE:-all}
R=$QAP_DATA_ROOT/runs
L=$QAP_DATA_ROOT/logs
mkdir -p "$L"
run() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

if run graded; then        # CPU: can run in parallel with 'train'
  python -m qap.scenes.graded_runner validate --trials 5 --procs "$PROCS" | tee "$L/graded_validate.log"
  python -m qap.scenes.graded_runner episodes --n-per-config 50 --procs "$PROCS" > "$L/graded_episodes.log" 2>&1
  python -m qap.scenes.graded_runner feasibility --trials 10 --procs "$PROCS" > "$L/graded_feasibility.log" 2>&1
fi

if run train; then         # main ensemble on GPU_MAIN, variants on GPU_AUX (sequential), in parallel
  CUDA_VISIBLE_DEVICES=$GPU_MAIN nohup python -m qap.quasimetric.train --objective qrl --head iqe --ensemble 5 \
      --run-name qrl_iqe_k5 > "$L/qrl_iqe_k5.log" 2>&1 &
  ( CUDA_VISIBLE_DEVICES=$GPU_AUX python -m qap.quasimetric.train --objective tmd  --head iqe --ensemble 5 --run-name tmd_iqe_k5  > "$L/tmd_iqe_k5.log" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU_AUX python -m qap.quasimetric.train --objective qrl  --head mrn --ensemble 5 --run-name qrl_mrn_k5  > "$L/qrl_mrn_k5.log" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU_AUX python -m qap.quasimetric.train --objective hilp --head l2  --ensemble 5 --run-name hilp_l2_k5  > "$L/hilp_l2_k5.log" 2>&1
  ) &
  wait
fi

if run labels; then        # CPU (+ renderer for --images)
  python -m qap.feasibility.label_triples --seeds 20 --images --procs "$PROCS" > "$L/label_triples.log" 2>&1
fi

if run prauc; then
  python -m qap.feasibility.prauc --visual --select qrl_iqe tmd_iqe \
    --ckpt qrl_iqe=$R/qrl_iqe_k5/ckpt.pt tmd_iqe=$R/tmd_iqe_k5/ckpt.pt \
           qrl_mrn=$R/qrl_mrn_k5/ckpt.pt hilp_l2=$R/hilp_l2_k5/ckpt.pt | tee "$L/prauc.log"
fi

if run policy; then        # uses the selected distance (edit DIST if TMD won)
  DIST=${DIST:-$R/qrl_iqe_k5/ckpt.pt}
  CUDA_VISIBLE_DEVICES=$GPU_MAIN python -m qap.policy.awr --dist-ckpt "$DIST" --run-name awr_main > "$L/awr_main.log" 2>&1
  python -m qap.policy.eval_goal_reaching --policy "$R/awr_main/policy.pt" --grades G0 G1 G3 --seeds 10 \
      --procs "$PROCS" | tee "$L/goal_reaching.log"
fi
