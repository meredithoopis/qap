"""Build the Week-1 task inventory table.

    python -m qap.week1.inventory [--runtime qap_runs/inventory/runtime_inventory.json]

Writes qap/data/task_inventory.csv and qap/data/task_inventory.md with, for every task:
    demonstrated object, L3 substitute, role, motion primitive at L0 vs L3, motion/grasp
    change, intent preserved, reversibility, substitute assets (static scan of the L0/L3
    env sources; runtime actor diff if --runtime is given), substitute coverage in other
    tasks, predicate availability (evaluate() keys at L0 and L3), IG Table-11 success,
    corpus membership and benchmark role.
No simulator is needed (static analysis only); --runtime adds the simulator view.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Dict, List, Set

from qap.common.io import load_json
from qap.common.paths import FROZEN_DIR, TASKS_DIR
from qap.common.tasks import load_task_table
from qap.week1.ig_reference import TABLE11, level_success
from qap.week1.inventory_spec import INVENTORY

ASSET_RE = re.compile(r"[\"'](\d{3}_[a-z][a-z_]*)[\"']")
SKETCHFAB_RE = re.compile(r"(?:load_sketchfab_object|create_sketchfab_actor)\([^)]*?[\"']([a-z][a-z0-9_]+)[\"']", re.S)
GENERIC_PRED = {"success", "is_robot_static", "is_left_static", "is_right_static", "is_grasped", "is_grasping"}


def _task_files() -> Dict[str, Dict[str, Path]]:
    files: Dict[str, Dict[str, Path]] = {}
    camel = {n.lower(): n for n in load_task_table()}
    for lvl, sub in (("L0", "dual_tasks"), ("L3", "dual_tasks_l3")):
        for p in sorted((TASKS_DIR / sub).glob("_0*.py")):
            src = p.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"@register_env\(\s*[\"']TwoRobot([A-Za-z]+?)(L3)?-v1", src)
            if not m:
                continue
            name = m.group(1)
            name = camel.get(name.lower(), name)
            files.setdefault(name, {})[lvl] = p
    return files


def scan_assets(path: Path) -> Set[str]:
    src = path.read_text(encoding="utf-8", errors="replace")
    return set(ASSET_RE.findall(src)) | {f"sketchfab:{s}" for s in SKETCHFAB_RE.findall(src)}


def evaluate_keys(path: Path) -> List[str]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    keys: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            for m in ast.walk(node):
                if isinstance(m, ast.Dict):
                    keys |= {k.value for k in m.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                if isinstance(m, ast.Call) and getattr(m.func, "id", "") == "dict":
                    keys |= {kw.arg for kw in m.keywords if kw.arg}
                if isinstance(m, ast.Subscript) and isinstance(m.ctx, ast.Store) and isinstance(m.slice, ast.Constant):
                    if isinstance(m.slice.value, str):
                        keys.add(m.slice.value)
    return sorted(keys)


def build_rows(runtime: Dict | None = None) -> List[Dict]:
    tasks = load_task_table()
    files = _task_files()
    assets = {n: {lvl: scan_assets(p) for lvl, p in d.items()} for n, d in files.items()}
    rows = []
    for e in INVENTORY:
        info = tasks.get(e.name)
        f = files.get(e.name, {})
        a0, a3 = assets.get(e.name, {}).get("L0", set()), assets.get(e.name, {}).get("L3", set())
        sub_assets = sorted(a3 - a0)
        coverage = {}
        for asset in sub_assets:
            # only the IG tasks themselves (task_mapping.json), not helper envs such as PickCubeYCB
            others = sorted(t for t, d in assets.items()
                            if t != e.name and t in tasks and any(asset in s for s in d.values()))
            coverage[asset] = others
        cov_tasks = sorted({t for v in coverage.values() for t in v})
        p0 = evaluate_keys(f["L0"]) if "L0" in f else []
        p3 = evaluate_keys(f["L3"]) if "L3" in f else []
        step0 = [k for k in p0 if k not in GENERIC_PRED and not re.search(r"dist|pos|steps|progress|height", k)]
        step3 = [k for k in p3 if k not in GENERIC_PRED and not re.search(r"dist|pos|steps|progress|height", k)]
        row = e.row()
        row.update(
            human_id=info.human_id if info else "",
            corpus=info.corpus if info else "",
            eval_role=info.eval_role if info else "",
            has_l3_env="L3" in f,
            sub_assets_static=";".join(sub_assets),
            sub_coverage_tasks=";".join(cov_tasks),
            sub_coverage_n=len(cov_tasks),
            preds_l0=";".join(p0), preds_l3=";".join(p3),
            step_preds_l0=len(step0), step_preds_l3=len(step3),
            predicate_available=bool("success" in p0 and "success" in p3 and step0 and step3),
        )
        if e.name in TABLE11:
            for lvl in ("L0", "L1", "L2", "L3"):
                row[f"ig_dp_{lvl}"] = level_success(e.name, lvl, "dp")
                row[f"ig_act_{lvl}"] = level_success(e.name, lvl, "act")
            row["ig_mean_L0"] = level_success(e.name, "L0")
            row["ig_mean_L3"] = level_success(e.name, "L3")
        if runtime and e.name in runtime:
            rt = runtime[e.name]
            n0 = set(rt.get("L0", {}).get("obj_names", []))
            n3 = set(rt.get("L3", {}).get("obj_names", []))
            c0 = set(rt.get("L0", {}).get("obj_categories", []))
            c3 = set(rt.get("L3", {}).get("obj_categories", []))
            row["rt_l3_only_actors"] = ";".join(sorted(n3 - n0))
            row["rt_l3_only_categories"] = ";".join(sorted(c3 - c0))
            row["rt_l0_only_categories"] = ";".join(sorted(c0 - c3))
            row["rt_sub_category_coverage"] = ";".join(sorted(rt.get("sub_category_coverage", [])))
        rows.append(row)
    return rows


def write_outputs(rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_dir / "task_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    cols = ["name", "corpus", "demo", "sub", "prim_l0", "prim_l3", "motion_changed", "grasp_changed",
            "intent_preserved", "reversibility", "sub_coverage_n", "predicate_available", "ig_mean_L0", "ig_mean_L3"]
    lines = ["# IG task inventory (generated by qap.week1.inventory)", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (out_dir / "task_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, default=None, help="runtime_inventory.json from qap.week1.runtime_inventory")
    ap.add_argument("--out", type=Path, default=FROZEN_DIR)
    args = ap.parse_args()
    runtime = load_json(args.runtime) if args.runtime and args.runtime.exists() else None
    rows = build_rows(runtime)
    write_outputs(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}/task_inventory.(csv|md)")


if __name__ == "__main__":
    main()
