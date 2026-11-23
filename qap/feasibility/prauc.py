"""PR-AUC of each distance against planner-certified realisation feasibility (SoRB protocol).

    python -m qap.feasibility.prauc \
        --ckpt qrl_iqe=qap_runs/runs/qrl_iqe_k5/ckpt.pt tmd_iqe=qap_runs/runs/tmd_iqe_k5/ckpt.pt \
               qrl_mrn=qap_runs/runs/qrl_mrn_k5/ckpt.pt hilp_l2=qap_runs/runs/hilp_l2_k5/ckpt.pt \
        --visual --select qrl_iqe tmd_iqe

Score of a triple (s, a, g) = -realisation cost:
    cost = d(s, w_a) + d(w_a, g)   if the entry state w_a is known, else d(s, g)
with the ensemble's pessimistic price (mean + 1 std; --kappa). Visual cosine uses
1 - cos(DINOv2(s), DINOv2(g)) (needs label_triples --images).
Reports PR-AUC (average precision) and ROC-AUC per split / grade / source / reversibility,
bootstrap 95% CIs, picks QRL vs TMD on the calibration split, and checks the Week-2 go/no-go
(quasimetric PR-AUC clearly above visual cosine: CI of the difference excludes 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from qap.common.io import load_json, save_json
from qap.common.paths import FROZEN_DIR, labels_dir
from qap.week1.inventory_spec import INVENTORY_BY_NAME


# ------------------------------------------------------------------ metrics
def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yy = y[order].astype(float)
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                      # average ranks for ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_ci(y, s, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = fn(y[i], s[i])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def bootstrap_diff(y, s1, s2, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a, b = average_precision(y[i], s1[i]), average_precision(y[i], s2[i])
        if not (np.isnan(a) or np.isnan(b)):
            d.append(a - b)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d else (float("nan"),) * 2


# ------------------------------------------------------------------ data
def load_triples(root: Path) -> Dict[str, np.ndarray]:
    parts: List[Dict[str, np.ndarray]] = []
    for f in sorted(root.glob("*.npz")):
        meta = load_json(f.with_suffix(".json"))
        a = dict(np.load(f))
        n = len(a["label"])
        a["task"] = np.array([meta["task"]] * n)
        a["grade"] = np.array([meta["grade"]] * n)
        if "img_s" not in a:
            a["has_img"] = np.zeros(n, bool)
        parts.append(a)
    if not parts:
        raise FileNotFoundError(f"no triples in {root}; run qap.feasibility.label_triples first")
    keys = set.intersection(*[set(p) for p in parts]) | {"img_s", "img_g"}
    out = {}
    for k in keys:
        if all(k in p for p in parts):
            out[k] = np.concatenate([p[k] for p in parts])
        elif k in ("img_s", "img_g"):
            ref = next((p[k] for p in parts if k in p), None)
            if ref is not None:
                out[k] = np.concatenate([p[k] if k in p else np.zeros((len(p["label"]),) + ref.shape[1:], ref.dtype)
                                         for p in parts])
    keep = out.get("goal_valid", np.ones(len(out["label"]), bool)).astype(bool)
    return {k: v[keep] for k, v in out.items()}


def _states(T: Dict[str, np.ndarray], prefix: str, idx: np.ndarray, device):
    obj = torch.as_tensor(T[f"{prefix}_obj"][idx], device=device)
    cat = torch.as_tensor(T["cat"][idx], dtype=torch.long, device=device)
    mask = (obj.abs().sum(-1) > 0) | (cat > 0)
    return dict(robot=torch.as_tensor(T[f"{prefix}_robot"][idx], device=device), obj=obj, cat=cat, mask=mask,
                events=torch.as_tensor(T[f"{prefix}_events"][idx], dtype=torch.float32, device=device))


@torch.no_grad()
def quasimetric_costs(ckpt: Path, T, kappa: float, device, batch: int = 1024) -> Dict[str, np.ndarray]:
    from qap.quasimetric.model import load_checkpoint

    ens = load_checkpoint(ckpt, device)
    n = len(T["label"])
    out = {"pess": np.zeros(n), "mean": np.zeros(n), "direct": np.zeros(n)}
    for i in range(0, n, batch):
        idx = np.arange(i, min(n, i + batch))
        s, w, g = (_states(T, p, idx, device) for p in ("s", "w", "g"))
        zs, zw, zg = ens.embed(s), ens.embed(w), ens.embed(g)
        d_sg = torch.stack([m.dist_z(a, b) for m, a, b in zip(ens.members, zs, zg)])
        d_sw = torch.stack([m.dist_z(a, b) for m, a, b in zip(ens.members, zs, zw)])
        d_wg = torch.stack([m.dist_z(a, b) for m, a, b in zip(ens.members, zw, zg)])
        has_w = torch.as_tensor(T["has_w"][idx], device=device)
        via = torch.where(has_w[None], d_sw + d_wg, d_sg)                 # [K, B]
        out["mean"][idx] = via.mean(0).cpu().numpy()
        out["pess"][idx] = (via.mean(0) + (kappa * via.std(0) if via.shape[0] > 1 else 0)).cpu().numpy()
        out["direct"][idx] = d_sg.mean(0).cpu().numpy()
    return out


def evaluate(name: str, y: np.ndarray, score: np.ndarray, groups: Dict[str, np.ndarray]) -> Dict:
    res = {}
    for gname, m in groups.items():
        if m.sum() < 5:
            continue
        ap = average_precision(y[m], score[m])
        res[gname] = dict(n=int(m.sum()), pos=int(y[m].sum()), pr_auc=ap, roc_auc=roc_auc(y[m], score[m]),
                          pr_auc_ci=bootstrap_ci(y[m], score[m], average_precision, n=300))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[], help="name=path pairs")
    ap.add_argument("--visual", action="store_true", help="add DINOv2 visual cosine (needs images)")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--select", nargs=2, default=None, help="two names to choose between on the calibration split")
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    T = load_triples(args.labels or labels_dir() / "triples")
    y = T["label"].astype(int)
    splits = load_json(FROZEN_DIR / "splits.json")["splits"] if (FROZEN_DIR / "splits.json").exists() else {}
    task_split = {t: k for k, v in splits.items() for t in v}
    split_of = np.array([task_split.get(t, "none") for t in T["task"]])
    rev = np.array([INVENTORY_BY_NAME[t].reversibility if t in INVENTORY_BY_NAME else "?" for t in T["task"]])
    from qap.feasibility.label_triples import SOURCES

    groups = {"all": np.ones(len(y), bool)}
    for k in ("train", "calibration", "test"):
        groups[f"split:{k}"] = split_of == k
    for gr in np.unique(T["grade"]):
        groups[f"grade:{gr}"] = T["grade"] == gr
    for i, src in enumerate(SOURCES):
        groups[f"source:{src}"] = T["source"] == i
    for r in np.unique(rev):
        groups[f"reversibility:{r}"] = rev == r

    results: Dict[str, Dict] = {}
    scores: Dict[str, np.ndarray] = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        c = quasimetric_costs(Path(path), T, args.kappa, device)
        scores[name] = -c["pess"]
        scores[f"{name}[direct]"] = -c["direct"]
    if args.visual:
        if "img_s" not in T or not T["has_img"].any():
            print("no images in triples (run label_triples --images); skipping visual cosine")
        else:
            from qap.quasimetric.visual_cosine import get_dino, visual_cosine_distance

            dino = get_dino(str(device))
            m = T["has_img"].astype(bool)
            fs, fg = dino(T["img_s"][m]), dino(T["img_g"][m])
            vc = np.full(len(y), np.nan)
            vc[m] = visual_cosine_distance(fs, fg).numpy()
            vc[~m] = np.nanmax(vc[m]) if m.any() else 1.0
            scores["visual_cosine"] = -vc
    for name, sc in scores.items():
        results[name] = evaluate(name, y, sc, groups)

    summary = {}
    if args.select:
        a, b = args.select
        cal = groups["split:calibration"]
        pa, pb = average_precision(y[cal], scores[a][cal]), average_precision(y[cal], scores[b][cal])
        summary["selection"] = dict(candidates=[a, b], calibration_pr_auc={a: pa, b: pb},
                                    chosen=a if (pb != pb or pa >= pb) else b)
    if "visual_cosine" in scores:
        heldout = groups["split:calibration"] | groups["split:test"]
        best = summary.get("selection", {}).get("chosen") or next((n for n in scores if n != "visual_cosine" and "[" not in n), None)
        if best:
            ci = bootstrap_diff(y[heldout], scores[best][heldout], scores["visual_cosine"][heldout])
            summary["go_no_go_prauc"] = dict(model=best, diff_vs_visual_ci95=ci, passes=bool(ci[0] > 0))
    out = labels_dir() / "prauc_results.json"
    save_json(dict(results=results, summary=summary, n_triples=int(len(y)), n_pos=int(y.sum())), out)

    cols = ["all", "split:calibration", "split:test", "reversibility:irreversible", "grade:G3", "grade:G4xG3"]
    print(f"{'distance':28s} " + " ".join(f"{c[:18]:>18s}" for c in cols))
    for name, r in results.items():
        print(f"{name:28s} " + " ".join(f"{r.get(c, {}).get('pr_auc', float('nan')):18.3f}" for c in cols))
    print(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
