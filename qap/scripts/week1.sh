#!/bin/bash
# Week 1 pipeline (run from the repository root). Each block can be run on its own.
#   STAGE=inventory|freeze|shortcut|mp|play  bash qap/scripts/week1.sh
set -euo pipefail
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export QAP_DATA_ROOT=${QAP_DATA_ROOT:-$PWD/qap_runs}
PROCS=${PROCS:-32}
GPUS=${GPUS:-0}             # space-separated GPU ids for the shortcut-filter DP runs, e.g. GPUS="0 1"
STAGE=${STAGE:-all}
mkdir -p "$QAP_DATA_ROOT/logs"

run() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

if run inventory; then
  python -m qap.week1.runtime_inventory --procs "$PROCS"
  # review qap/data/category_overrides.draft.json -> save as qap/data/category_overrides.json,
  # then re-run runtime_inventory once so category_vocab.json uses the reviewed names.
  python -m qap.week1.inventory --runtime "$QAP_DATA_ROOT/inventory/runtime_inventory.json"
fi

if run freeze; then
  python -m qap.week1.screen_tasks            # writes qap/data/frozen_tasks.json + splits.json (refuses to overwrite)
fi

if run shortcut; then
  python -m qap.week1.shortcut_filter prepare --gpus $GPUS
  nohup bash "$QAP_DATA_ROOT/shortcut/run_shortcut.sh" > "$QAP_DATA_ROOT/logs/shortcut.log" 2>&1 &
  echo "shortcut filter running in background (pid $!) -> $QAP_DATA_ROOT/logs/shortcut.log"
fi

if run mp; then
  nohup python -m qap.collect.collect_mp_states --corpus 45 --levels L0 L1 L2 L3 --n-per-scene 50 \
      --procs "$PROCS" > "$QAP_DATA_ROOT/logs/collect_mp.log" 2>&1 &
  echo "MP state collection in background (pid $!)"
fi

if run play; then
  nohup python -m qap.collect.play --target-transitions 1000000 --procs "$PROCS" \
      > "$QAP_DATA_ROOT/logs/play.log" 2>&1 &
  echo "play generation in background (pid $!)"
fi
