# QAP — Week 1 & Week 2 run guide

This folder adds the Week 1 and Week 2 work from the execution plan, built on top of the
Imitator Game (IG) repo. It does not change any IG file. Everything below is meant to run on
the Linux GPU server, from the repository root.

```
qap/
  common/       task table (built from IG configs), env factory (L0-L3 flags), episode I/O
  state/        object-centric state extractor + gym recorder (contacts -> entry states W_a)
  week1/        inventory, IG Table-11 reference, screening/freeze, shortcut filter, repro check
  collect/      IG planner episodes as states (45-task corpus), robot play generator
  quasimetric/  object-token transformer, IQE / MRN / L2 heads, QRL / TMD / HILP / distill, DINOv2 cosine
  scenes/       IG-Sub graded scenes G0-G4: specs, builder, validate / episodes / planner feasibility
  feasibility/  (state, affordance, goal) triple labelling with the planner, PR-AUC (SoRB protocol)
  policy/       AWR goal-conditioned policy from -d_theta, oracle-goal reaching evaluation
  scripts/      Slurm jobs: shortcut.sbatch, collect.sbatch (Week 1), week2.sbatch (Week 2);
                smoke_test.sh, check_dataset.py; week1.sh / week2.sh (non-Slurm launchers, unused here)
  data/         frozen deliverables (hypotheses.md now; inventory, frozen task list, splits after Week 1)
  tests/        CPU property tests for the heads
```

Large outputs go under `$QAP_DATA_ROOT` (default `./qap_runs`, already in `.gitignore`).

---

## 0. Server setup (Week 1, item 1)

```bash
cd The-Imitator-Game
uv sync                      # creates .venv (Python 3.12, from pyproject.toml) and installs everything
source .venv/bin/activate    # repeat in every new shell / tmux pane
which python                 # should print .../The-Imitator-Game/.venv/bin/python
# lerobot patch from the IG README (path computed, the README's python3.11 is wrong for 3.12)
LEROBOT_DIR=$(python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))")
curl -sSL https://raw.githubusercontent.com/huggingface/lerobot/0e81a275fcdbf03d74f78aa69eaa28c172a9f256/src/lerobot/datasets/lerobot_dataset.py \
  -o "$LEROBOT_DIR/datasets/lerobot_dataset.py"
# Vulkan: see the ManiSkill install docs (needed for any rendering, e.g. --images)

# assets -> ~/.maniskill/data
mkdir -p ~/.maniskill/data
hf download imitator-game/IG-10K-Assets --repo-type dataset --local-dir ~/.maniskill/data
for f in ~/.maniskill/data/*.tar.zst; do tar --use-compress-program=unzstd -xf "$f" -C ~/.maniskill/data; done
rm ~/.maniskill/data/*.tar.zst

# simulated IG-10K (human clips + sim robot episodes) -> demos/
hf download imitator-game/IG-10K-Dataset --repo-type dataset --local-dir demos
# The download is NOT laid out as demos/demo_data + demos/imitator_data (IG README). It gives:
#   demos/imitator_human_v1/       human_H<N>/              <- --human-root (shortcut filter default)
#   demos/imitator_sim_v1_zed2i/   L<k>_TwoRobot<Task>-v1/  <- --sim-root   (shortcut filter default)
#   demos/imitator_robot_v1/, demos/imitator_human_v1_levels/   (not used in Weeks 1-2)
ls demos/imitator_human_v1 | head -3; ls demos/imitator_sim_v1_zed2i | head -3

export PYTHONPATH=$PWD:$PYTHONPATH
export QAP_DATA_ROOT=$PWD/qap_runs

# completeness / layout check of the 50 human + 200 sim datasets (read-only; add --deep to
# compare parquet rows with meta/info.json, --robot for the real-robot folders)
python -m qap.scripts.check_dataset
```

Check the simulator and planner work:

```bash
# first: numpy/scipy must import together (pyproject caps scipy<1.16 because numpy<2)
python -c "import numpy, scipy, scipy.spatial, trimesh; print(numpy.__version__, scipy.__version__)"

# needs a level flag (--L0..--L3) or --traj-name, otherwise two_robot_run hits an unbound
# variable; -o none skips camera rendering. Output goes to a throwaway temp folder only:
#   $TMPCHK/TwoRobotPlaceFoodScale-v1/motionplanning/l0_TwoRobotPlaceFoodScale-v1.{h5,json}
TMPCHK=$(mktemp -d /tmp/ig_check.XXXXXX)
python -m mani_skill.examples.motionplanning.dual.two_robot_run -e TwoRobotPlaceFoodScale-v1 -n 1 --only-count-success --L0 -o none --record-dir "$TMPCHK"
ls -lh "$TMPCHK"/TwoRobotPlaceFoodScale-v1/motionplanning/
rm -rf "$TMPCHK"
```

### Smoke test (strongly recommended before the big jobs)

Runs every QAP stage on one task with tiny sizes, in a separate output folder
(`qap_runs_smoke`). It takes about 15–30 minutes. Any API mismatch with your installed
ManiSkill/SAPIEN shows up here, not after hours of collection.

```bash
bash qap/scripts/smoke_test.sh              # TASK=PlaceFoodScale by default
```

---

## Week 1

### 1.1 Task inventory

There are two parts. The static part needs no simulator. The runtime part instantiates every
scene once.

```bash
python -m qap.week1.inventory                                  # -> qap/data/task_inventory.{csv,md}
python -m qap.week1.runtime_inventory --procs 32               # all 50 tasks x L0-L3, CPU
python -m qap.week1.inventory --runtime $QAP_DATA_ROOT/inventory/runtime_inventory.json
```

For each task the inventory gives: demonstrated object, L3 substitute, role, motion primitive
at L0 vs L3, motion/grasp change, intent preserved, reversibility, substitute assets, substitute
coverage in other tasks (by asset id statically and by category at runtime), predicate
availability (evaluate() keys at L0/L3), IG Table-11 success, corpus membership.

The curated columns are in `qap/week1/inventory_spec.py`. I wrote them from the L0/L3 task
descriptions in `task_desc/sim_desc.json` and from the env sources. Please review them once.

**Review the object categories once.** `runtime_inventory` writes
`qap/data/category_overrides.draft.json` with an automatic category for every actor name. Some
IG actor names say nothing about the object (`food`, `cube`, `fruit_a`, numeric PartNet ids), and
these are listed under `_review_needed`. Fix them, save the file as
`qap/data/category_overrides.json`, and re-run `runtime_inventory`. The second run regenerates
`qap/data/category_vocab.json` (the frozen category ids) from the reviewed names. Do this
**before** collecting data. Episodes store category *names*, so you can rebuild the vocabulary
later, but freezing it early is cleaner.

### 1.2 Screening rule, frozen task list, splits, hypotheses

```bash
python -m qap.week1.screen_tasks --dry-run     # inspect
python -m qap.week1.screen_tasks               # writes qap/data/frozen_tasks.json + qap/data/splits.json
```

The plan's rule is: retain an L3 task if its primitive changes and IG's per-task L3 success is
below L0. The script applies it as four checks. Every exclusion is logged with its reason.

- R1: the motion primitive changes (a pure grasp change does not count, `--count-grasp-change` to include it)
- R2: mean(DP, ACT) success at L0 − at L3 is at least 10 points (`--min-gap`). One episode out
  of ten is 10 points, so a 5-point gap is noise.
- R3: the L3 variant keeps the demonstrated intent (`--no-intent-check` to disable)
- R4: an L3 env exists

With the defaults the expected result is **13 tasks**, inside the 10–15 target:

| task | L0 → L3 primitive | reversibility | IG mean L0 → L3 |
|---|---|---|---|
| PlaceBookBookcase | insert upright → unblock + place in box | reversible | 100 → 60 |
| PlaceClothBasket | drop in basket → unblock + insert in box | reversible | 100 → 50 |
| PlaceMagazineFolder | open folder + insert → stack | reversible | 90 → 70 |
| PlaceCommodityRack | place on shelf → hang on rack | reversible | 70 → 15 |
| PlacePlateRack | insert in rack → stack bowl on plate | reversible | 95 → 80 |
| PourKetchupFries | squeeze bottle → press pump | irreversible | 75 → 45 |
| TransFood | scoop with spoon → pour bowl | irreversible | 55 → 25 |
| PlaceFoodScale | place on scale → pan + counterweight | reversible | 50 → 0 |
| ScanPillBottle | handheld scanner → bring to desk scanner | event | 95 → 5 |
| PlacePillBox | drop pill → tip bottle to dispense | irreversible | 90 → 50 |
| PutCubeOnScale | place on scale → pan + counterweight | reversible | 100 → 65 |
| OpenLiquidCap | unscrew → lift lid | reversible | 75 → 25 |
| PutBox | hinged lid → sliding lid | reversible | 100 → 55 |

Near misses: ScanMilkBox and PlaceMugRack (5-point gap) and PressStapler (intent changes at
L3, stapling → filing).

The split is by task, and I wrote it out by hand in `screen_tasks.PROPOSED_SPLIT` so you can
review it:

- **test**: PourKetchupFries, PlacePillBox, ScanPillBottle, PlaceFoodScale, PlaceBookBookcase.
  Two irreversible tasks are here for H3.
- **calibration**: TransFood, PutBox, PlaceMagazineFolder. Used only for τ (conformal) and λ.
- **train**: PlaceClothBasket, PlaceCommodityRack, PlacePlateRack, PutCubeOnScale, OpenLiquidCap.

If you change a rule flag and the retained set changes, the script falls back to a
deterministic stratified split.

Both files carry a hash. After they are written the script refuses to overwrite them
(`--force` only before anything has been evaluated). H1–H5 and the ablation predictions are in
`qap/data/hypotheses.md`.

The per-task IG numbers come from Table 11 (Appendix H, page 32) of arXiv:2608.22301. I
transcribed them into `qap/week1/ig_reference.py`. Every row's mean matches the paper's Mean
column, and the row alignment agrees with the paper's own narrative. Spot-check a few rows
against the PDF before freezing.

### Running on the cluster

The long Week-1 jobs run as Slurm batch jobs (no `nohup`), submitted from the repository root.
The QOS allows one running job per user, and a 32-CPU job is refused (`QOSMaxCpuPerJobLimit`),
so the jobs below use 16 CPUs and run one after another. Check the node first:

```bash
squeue -u $USER
scontrol show node worker-0 | grep -E "Gres|AllocTRES"     # free CPUs / GPUs
```

### 1.3 Shortcut filter (DP on L0–L2 per retained task)

```bash
python -m qap.week1.shortcut_filter prepare                    # tasks.txt + per-task scripts (seconds)
# one array task per line of tasks.txt (test -> calibration -> train); %1 = one at a time
sbatch --array=1-8%1 --cpus-per-task=16 --mem=64G --time=0 --nodelist=worker-0 qap/scripts/shortcut.sbatch
# logs: shortcut_<job>_<i>.out ; when all array tasks have finished:
python -m qap.week1.shortcut_filter summarize                  # KEEP if DP L3 success < 50%
python -m qap.week1.screen_tasks --apply-shortcut-results $QAP_DATA_ROOT/shortcut/summary.json --allow-missing-train
```

- `--array=1-8` covers the 8 held-out tasks (5 test + 3 calibration). The 5 train tasks
  (lines 9–13) are optional: `--array=9-13%1` runs them. `--allow-missing-train` keeps
  unevaluated train tasks and records them as `not_evaluated`.
- Each task takes about 1–1.5 h on one H100 (≈50 min training at 100 epochs, then L0 + L3
  evaluation), so the 8 held-out tasks take roughly 8–12 h in sequence.
- Resumable: a task with `final_model.pt` skips training, and a killed task resumes from its
  newest epoch checkpoint. Delete `$QAP_DATA_ROOT/shortcut/runs` to start over.
- Each task has its own DINOv2 feature cache (`$QAP_DATA_ROOT/shortcut/te_cache/<task>`); a
  shared cache causes `CachedFrozenBackbone Cache miss` on the second task.

This uses the IG DP trainer/evaluator unchanged: video-only, DINOv2, the same flags as
`exp_scripts/diffusion_policy`. Training data per task is L0–L2 × 50 episodes, and evaluation
covers L0 (sanity) and L3. Evaluation renders with `--shader default`; IG's own runs use
`rt-fast`, which needs the OptiX denoiser and fails on drivers without `libnvoptix`
(`prepare --shader rt-fast` restores it). IG trains C45 for 10 epochs (about 80k steps); one
task's L0–L2 corpus is much smaller, so the default here is 100 epochs (`--epochs`). The filter
can only *remove* tasks. Apply its results before any QAP evaluation.

### 1.4 Robot data as object-centric states

The released IG-10K LeRobot files hold images, joint states and actions only, with no object
poses. So the 45-task corpus is re-generated with IG's own planner solutions while the new
extractor records the states:

```bash
sbatch -J qap_mp --export=ALL,WHAT=mp --cpus-per-task=16 --mem=64G --gres=gpu:1 qap/scripts/collect.sbatch
grep '^{' qap_mp_<jobid>.out | tail -3          # progress: one JSON line per finished scene
grep -c '^{' qap_mp_<jobid>.out                 # 172 when done
```

- This is 45 tasks × 4 levels × 50 successful episodes (172 scenes after the held-out
  exclusion), 15 simulator processes. Reference run: 96 min, ≈2.95M transitions, 394 MB.
- SAPIEN creates a Vulkan renderer even without images, so this job asked for a GPU. Without
  one it fails with `vk::createInstanceUnique: ErrorIncompatibleDriver`; when no GPU is free,
  use the CPU-only form shown in 1.5.
- Failed planner attempts are kept too, because they are valid robot transitions
  (`--no-failures` to drop them).
- It is resumable: resubmit the same command and seeds whose episode exists are skipped.
- **Held-out policy:** L3 scenes of calibration and test tasks are *not* collected (and graded
  G2–G4 scenes of those tasks are filtered out of training), so substitute connectivity of
  held-out tasks never enters d_θ. Use `--include-heldout` to change this.
- Benign log lines: `screw plan failed`, `env.agent ... deprecated`, `pynvml ... deprecated`.

### 1.5 Robot play (∼1M transitions)

CPU-only: `QAP_RENDER_BACKEND=none` disables the renderer and `--gres=none` overrides the
script's GPU request, so it starts even when every GPU is allocated.

```bash
sbatch -J qap_play --export=ALL,WHAT=play,QAP_RENDER_BACKEND=none --cpus-per-task=16 --mem=64G --gres=none qap/scripts/collect.sbatch
grep '^{' qap_play_<jobid>.out | tail -3        # progress: one JSON line per batch of 10 episodes
python -m qap.collect.play --count              # transitions saved so far
grep '^{' qap_play_<jobid>.out | grep -c fatal  # should be 0
```

Each play episode resets a random IG scene (any of the 50 tasks, L0–L3, same held-out
exclusion). It then runs 3–8 random primitives through the IG motion planner with the nearer
arm: pick, place, place-on-another-object, tilt (pour-like), push, press, drag an articulated
part, random reach, home. Reference run: 46 min, ≈1.5M transitions (all 400 batches finish, so
the total overshoots the 1M target). Resubmit the same command to top up.

The log is full of planner warnings (`Fail to parametrize path`, `The path is not controllable`,
`controllable set ... can't be computed`, `TOPP failed in screw planning`). They mean a random
target was unreachable and that action was skipped; they are expected. What matters is
`"errors": 0` and no `"fatal"` in the JSON lines.

**Week-1 deliverables:**

- `qap/data/frozen_tasks.json`
- `qap/data/splits.json`
- `qap/data/hypotheses.md`
- `qap/data/task_inventory.{csv,md}`
- the state extractor (`qap/state/`)
- `qap/data/category_vocab.json`

Commit these after the freeze.

---

## Week 2

Week 2 builds the learned distance d_θ that QAP uses to price "can this object still get us to
the goal?", and checks it twice before Week 3:

- **PR-AUC go/no-go:** d_θ predicts planner-certified feasibility better than DINOv2 visual cosine.
- **Goal-reaching go/no-go:** a policy driven by d_θ reaches the oracle object in most trials.

Every stage is one Slurm job of `qap/scripts/week2.sbatch` (16 CPUs, 96G; 1 GPU unless
`--gres=none`). With one running job per user they run strictly one after another, in the order
below. Each job's log is `<job-name>_<jobid>.out`. Times are estimates for 15 simulator
processes / one H100 and are not measured yet; the first log lines show the real speed.

| # | Stage | Job | GPU | Needs | Estimate |
|---|---|---|---|---|---|
| 0 | apply shortcut results | login node | – | Week 1 | seconds |
| 1 | validate graded scenes | `STAGE=validate` | yes¹ | 0 | 10–15 min (28 min measured with the old 300 s limit) |
| 2 | graded episodes + planner feasibility | `STAGE=graded` | yes¹ | 1 (checked) | 2.5–3.5 h |
| 3 | feasibility triples (+ images) | `STAGE=labels` | yes | 1 | 1–2 h |
| 4 | train QRL (main distance) | `STAGE=train,RUN=qrl_iqe` | yes | 2 | 1.5–3 h |
| 5 | train TMD (rival for selection) | `STAGE=train,RUN=tmd_iqe` | yes | 2 | 1.5–3 h |
| 6 | PR-AUC + select QRL vs TMD | `STAGE=prauc` | yes | 3, 4, 5 | 15–30 min |
| 7 | *optional* MRN / HILP variants, then re-run 6 | `STAGE=train,RUN=qrl_mrn` / `hilp_l2` | yes | 2 | 1.5–3 h each |
| 8 | policy extraction (AWR) | `STAGE=awr` | yes | 6 | 2–4 h |
| 9 | oracle-goal reaching | `STAGE=goal` | yes¹ | 8 | 30–60 min |

¹ These stages simulate on the CPU, but are submitted with a GPU so SAPIEN's renderer starts
(`QAP_RENDER_BACKEND=` keeps the CPU-only switch off even if your shell has it set).

Required path (0–6, 8, 9): roughly **11–17 h** of jobs; the optional variants add 3–6 h.

Why this order: the graded scenes (1–2) are the yardstick for everything after them, and their
episodes are added to training, so they come first. The labels (3) need only the scenes. Only
QRL and TMD are needed for the go/no-go (6), so it runs before spending GPU time on the
comparison-only variants (7). AWR (8) uses whichever distance stage 6 selects.

To queue a stage behind the running one without watching, add
`--dependency=afterok:<previous jobid>` to `sbatch`. Do not chain past stage 1 or stage 6:
both need you to read a result first.

### 0. Apply the shortcut-filter results (login node)

```bash
python -m qap.week1.screen_tasks --apply-shortcut-results $QAP_DATA_ROOT/shortcut/summary.json --allow-missing-train
```

Every Week 2 stage reads `retained_after_shortcut` from `qap/data/frozen_tasks.json`; the job
prints a warning if this has not been done. All 8 evaluated tasks passed, so the 13 tasks stay.

### 1. Validate the graded scenes G0–G4 (CPU)

```bash
sbatch -J w2_validate --export=ALL,STAGE=validate,QAP_RENDER_BACKEND= qap/scripts/week2.sbatch
```

Each retained task is built at five difficulty grades, from "copy the demo" to "abstain":

| Grade | Scene | Solver | Correct decision |
|---|---|---|---|
| G0 | IG L0 | L0 | C (copy) |
| G1 | IG L2 (new instance) | L0 | C |
| G2 | L0 with an asset swap | L0 | S (substitute) |
| G3 | IG L3 | L3 | S |
| G4 | IG L3 with the substitute parked out of reach | L3 | A (abstain) |

No task file is edited. G2 patches the loader calls of the task module, and G4 parks actors
after every reset (it resolves `@l3_only` from `runtime_inventory.json`).

G2 exists only where the assets allow a different category with the same affordance.
`qap/scenes/graded_specs.py` keeps 2 swaps after validation; the other 11 tasks have no G2:

- PlaceClothBasket: basket → plastic box (5/5)
- PlaceCommodityRack: shelf → tray (5/5)
- dropped: PourKetchupFries squeeze → spray bottle (0/5), TransFood spoon → spatula (0/5),
  PlacePillBox box → bowl (asset does not build)

G4 normally hides the actors present at L3 but not at L0 (`@l3_only`). For TransFood,
ScanPillBottle and PlacePillBox the names coincide, so `G4_HIDE` names the task attribute
instead (`attr:bowl_src`, `attr:scanner`, `attr:pill`). G4 trials never succeed and run into the
per-trial limit, which `week2.sbatch` sets to 120 s (`QAP_TRIAL_TIMEOUT`).

Run graded stages **with a GPU** (omit `--gres=none`): the CPU-only renderer switch has not
built an IG scene successfully yet.

`validate` runs 5 trials per (task, grade) and writes `$QAP_DATA_ROOT/graded/validation.json`.
**Read the table before stage 2:** G0–G3 should mostly succeed and G4 must fail. Remove or fix
G2 entries that do not build or never succeed, then re-validate.

### 2. Graded episodes and planner feasibility (CPU)

```bash
sbatch -J w2_graded --export=ALL,STAGE=graded,QAP_RENDER_BACKEND= qap/scripts/week2.sbatch
```

- `episodes`: 50 robot episodes per (task, grade) →
  `$QAP_DATA_ROOT/episodes/graded/<task>/<grade>/`. They show the distance what substitution
  looks like (G2/G3). G2–G4 of calibration and test tasks are filtered out of training.
- `feasibility`: 10 planner trials per (task, grade, step, affordance ∈ {copy, substitute}) →
  `$QAP_DATA_ROOT/labels/planner_feasibility.json`, with the ground-truth label C/S/A and
  status `n/a` when a solver cannot run in that scene at all.
- Not resumable: before re-running, delete `$QAP_DATA_ROOT/episodes/graded` (otherwise the
  episodes are added twice).

### 3. Feasibility triples for PR-AUC (GPU)

```bash
sbatch -J w2_labels --export=ALL,STAGE=labels qap/scripts/week2.sbatch
```

Test cases (s, a, g) certified by the IG planner, 20 seeds per (task, grade) starting at seed
5000 (disjoint from every training seed). Four sources:

- `planner`: reset → solver. `w` is the first gripper-contact onset, `g` the final state.
  Successes are positives.
- `planner_mid`: the solver restarted from a restored mid-episode state.
- `reverse_irr`: g → s after a latched irreversible event. Label 0.
- `g4_vs_g3`: a G4 start with the G3 goal of the same seed. Label 0.

`--images` stores 224×224 zed2i frames for the DINOv2 baseline, so this stage needs the
renderer and therefore a GPU. Output: `$QAP_DATA_ROOT/labels/triples/`.

### 4–5. Train the distance d_θ: QRL, then TMD (GPU)

```bash
sbatch -J w2_qrl --export=ALL,STAGE=train,RUN=qrl_iqe qap/scripts/week2.sbatch
sbatch -J w2_tmd --export=ALL,STAGE=train,RUN=tmd_iqe qap/scripts/week2.sbatch
```

- **Data:** robot data only: `episodes/mp` + `episodes/play` + `episodes/graded`, held-out
  scenes excluded.
- **Encoder:** CLS + 2 robot tokens + ≤16 object tokens (pose, half extents, 14 discrete
  flags, category embedding) + 1 latched event token. 4 pre-LN layers, d = 256. Head: IQE
  (64 × 32 components).
- **Units:** frame skip 5 (one unit of distance is 5 control steps). QRL constraint:
  E[relu(d(s,s') − 1)²] ≤ ε² (ε = 0.25). Objective: −softplus(200 − d) with β = 0.01; tune
  `--softplus-offset` if distances saturate.
- **Ensemble:** 5 members trained in lockstep on shared batches. Pessimistic price = mean + 1·std.
- **Length:** 200k steps at batch 512. Training does not resume: a killed job starts over.
- **Output:** `$QAP_DATA_ROOT/runs/<run>_k5/{ckpt.pt,log.jsonl,config.json}`.
- **TMD caveat:** `objectives.TMD` is a state-only port of the TMD *loss structure*
  (InfoNCE on future-state positives plus a temporal backup matching e^{−d(s,g)} to
  γ·e^{−d(s',g)} under the Bregman divergence of exp). Check its constants against the
  official code before reporting TMD numbers; all weights are flags.

### 6. PR-AUC and the first go/no-go (GPU)

```bash
sbatch -J w2_prauc --export=ALL,STAGE=prauc qap/scripts/week2.sbatch
```

Scores every trained distance (whichever `runs/*_k5/ckpt.pt` exist) and the DINOv2 visual
cosine on the triples. The score is −[d(s,w) + d(w,g)] with the pessimistic price. It reports
PR-AUC and ROC-AUC with bootstrap CIs by split, grade, source and reversibility, and writes
`$QAP_DATA_ROOT/labels/prauc_results.json`.

- **Selection:** QRL vs TMD on the calibration split → `summary.selection.chosen`.
- **Go/no-go:** the 95% CI of PR-AUC(selected) − PR-AUC(visual cosine) must exclude 0 on the
  calibration and test splits.
- **If it fails**, train the distillation fallback and re-run this stage:
  `sbatch -J w2_distill --export=ALL,STAGE=train,RUN=distill_iqe qap/scripts/week2.sbatch`
  (trains on `episodes/mp` + `episodes/graded`).

### 7. Optional: comparison variants (GPU)

```bash
sbatch -J w2_mrn  --export=ALL,STAGE=train,RUN=qrl_mrn qap/scripts/week2.sbatch
sbatch -J w2_hilp --export=ALL,STAGE=train,RUN=hilp_l2 qap/scripts/week2.sbatch
```

Same encoder with a different head (MRN) or objective (HILP), for the comparison table only.
Re-run stage 6 afterwards to add them to `prauc_results.json`.

### 8. Policy extraction with AWR (GPU)

```bash
sbatch -J w2_awr --export=ALL,STAGE=awr qap/scripts/week2.sbatch                 # QRL selected
sbatch -J w2_awr --export=ALL,STAGE=awr,DIST=tmd_iqe qap/scripts/week2.sbatch    # if TMD was selected
```

A = d(s_t, g) − d(s_{t+k}, g) − 1 and weight = min(e^{βA}, 100). Hindsight goals are future
states of the same episode, plus 10% random goals. The policy predicts joint deltas and the
gripper command for both arms. 200k steps at batch 1024 → `$QAP_DATA_ROOT/runs/awr_main/policy.pt`.

### 9. Oracle-goal reaching, the second go/no-go (CPU)

```bash
sbatch -J w2_goal --export=ALL,STAGE=goal,QAP_RENDER_BACKEND= qap/scripts/week2.sbatch
```

For each task × {G0, G1, G3} × 10 seeds (from 9000), the planner first solves the seed once to
get the oracle entry state w (first contact) and goal g. The policy then starts from the same
initial state with w as its subgoal, then g. Output:
`$QAP_DATA_ROOT/labels/goal_reaching_awr_main.json`.

**Go/no-go:** `summary.go_no_go_majority`: `reach_w` above 0.5 (a majority of trials touch the
oracle object). `reach_g` (task success) is reported as well.

---

## Decisions and assumptions to review

1. **Screening refinements** (R2 gap ≥ 10 and R3 intent) go beyond the plan's wording. Both are
   flags, and their effect is listed in `frozen_tasks.json`.
2. **Hand-written split:** see 1.2.
3. **Held-out scenes:** L3 and G2–G4 of calibration and test tasks are excluded from all
   training data (d_θ, play, AWR). The paper's d_θ uses "robot episodes of the 45-task corpus",
   which would include those L3 episodes. Excluding them keeps the test honest about unseen
   substitutes. Pass `--include-heldout` to follow the paper literally.
4. **Events:** liquids and sauces are not simulated, so irreversible effects (poured, cut,
   dispensed, cleaned, …) enter the state through a latched event token. It is derived from
   evaluate() keys (`state/schema.py: EVENT_PATTERNS`) and from success at irreversible tasks
   (`TASK_SUCCESS_EVENT`). Per-object flags such as `filled` are set only if you add
   `qap/data/flag_map.json` (`{"Task": {"pred_key": {"object": "<actor name>", "flag": "filled"}}}`).
5. **Distance units:** one unit is `--frame-skip` control steps (default 5).
6. **G2 coverage** is partial (5 of 13 tasks). The rest of IG-Sub uses G0, G1, G3 and G4.
7. **Feasibility labels** are planner certificates. A planner failure counts as infeasible, as
   in SoRB. Failed runs never provide goal states.

## Output layout (`$QAP_DATA_ROOT`)

```
inventory/runtime_inventory.json      shortcut/{configs,eval,summary.json,run_shortcut.sh}
episodes/{mp,play,graded}/<task>/<level|grade>/*.npz|.json
runs/<name>/{ckpt.pt|policy.pt,log.jsonl,config.json}
graded/validation.json                labels/{planner_feasibility.json,triples/,prauc_results.json,goal_reaching_*.json}
logs/*.log
```

## Troubleshooting

- **A stage fails with an attribute error inside ManiSkill:** run the smoke test and send the
  traceback. The code targets the ManiSkill version vendored in this repo (CPU PhysX,
  `pd_joint_pos`, gymnasium 0.29).
- **A job hangs on one planner trial (seen on G4, where the L3 solver plans towards a hidden,
  unreachable object):** every solver/play call runs under a wall-clock limit,
  `QAP_TRIAL_TIMEOUT` seconds (default 300; the smoke test uses 120; `0` disables it). A
  timeout counts as a planning failure (`timeouts` in graded_runner output), not an error.
- **numpy-2-only wheels (`np.long`, `StringDType`, `cbuffer_sizes` import errors):** the
  `[tool.uv] override-dependencies` in `pyproject.toml` cap scipy<1.16, zarr<3,
  numcodecs<0.16 because the project pins numpy<2. Re-run `uv sync` after changing them.
- **DP training crashes at `from torch.utils.tensorboard import SummaryWriter`** with
  `compiled against NumPy C-API version 0x12` (from `ml_dtypes` / `tensorflow`): tensorboard
  imports TensorFlow if it is installed, and the installed TF / ml_dtypes wheels need NumPy 2.
  TensorFlow is only used by the OpenVLA baseline, so remove it from the venv:
  `uv pip uninstall tensorflow tensorflow-cpu tensorflow-datasets`, and from then on sync with
  `uv sync --no-install-package tensorflow --no-install-package tensorflow-cpu --no-install-package tensorflow-datasets`
  (a plain `uv sync` puts it back).
- **`no motion-planning solution registered`:** that (task, level) is missing from
  `two_robot_run.MP_SOLUTIONS`. It is skipped and listed in the logs.
- **Memory when loading many episodes:** use `--obj-dtype float16` and/or `--max-episodes N`
  in `quasimetric.train` / `policy.awr`.
- **Distances saturate at the softplus offset:** raise `--softplus-offset`, or raise
  `--frame-skip`.
