#!/bin/bash
# Tiny end-to-end check of the QAP pipeline on the server (~15-30 min, CPU + 1 GPU).
# Writes into $QAP_DATA_ROOT (default: qap_runs_smoke) so it never mixes with real runs.
# Run from the repository root after the IG install:  bash qap/scripts/smoke_test.sh
set -euo pipefail
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export QAP_DATA_ROOT=${QAP_DATA_ROOT:-$PWD/qap_runs_smoke}
TASK=${TASK:-PlaceFoodScale}

echo "== unit tests (heads)";            python -m pytest qap/tests -q
echo "== static inventory";              python -m qap.week1.inventory --out "$QAP_DATA_ROOT/inventory_static"
echo "== screening (dry run)";           python -m qap.week1.screen_tasks --dry-run
echo "== runtime inventory, one task";   python -m qap.week1.runtime_inventory --tasks "$TASK" --levels L0 L3 --procs 2
echo "== MP states, 2 episodes";         python -m qap.collect.collect_mp_states --tasks "$TASK" --levels L0 L3 --n-per-scene 2 --procs 2 --include-heldout
echo "== play, ~2k transitions";         python -m qap.collect.play --tasks "$TASK" --levels L0 --target-transitions 2000 --episodes-per-job 4 --procs 2 --include-heldout
echo "== train d_theta 300 steps";       python -m qap.quasimetric.train --objective qrl --head iqe --ensemble 2 --steps 300 --batch 64 --log-every 50 --run-name smoke_qrl --include-heldout
echo "== train TMD 100 steps";           python -m qap.quasimetric.train --objective tmd --head iqe --ensemble 1 --steps 100 --batch 64 --nce-batch 32 --log-every 50 --run-name smoke_tmd --include-heldout
echo "== graded validate";               python -m qap.scenes.graded_runner validate --tasks "$TASK" --trials 1 --procs 3
echo "== triples (2 seeds)";             python -m qap.feasibility.label_triples --tasks "$TASK" --grades G0 G3 G4 --seeds 2 --procs 3
echo "== PR-AUC";                        python -m qap.feasibility.prauc --ckpt qrl="$QAP_DATA_ROOT/runs/smoke_qrl/ckpt.pt" tmd="$QAP_DATA_ROOT/runs/smoke_tmd/ckpt.pt" --select qrl tmd
echo "== AWR 200 steps";                 python -m qap.policy.awr --dist-ckpt "$QAP_DATA_ROOT/runs/smoke_qrl/ckpt.pt" --steps 200 --batch 64 --run-name smoke_awr --include-heldout
echo "== goal reaching (1 seed)";        python -m qap.policy.eval_goal_reaching --policy "$QAP_DATA_ROOT/runs/smoke_awr/policy.pt" --tasks "$TASK" --grades G0 --seeds 1 --budget-w 50 --budget-g 50 --procs 1
echo "SMOKE TEST OK (numbers are meaningless at this scale)"
