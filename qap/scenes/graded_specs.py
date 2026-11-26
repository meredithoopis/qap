"""IG-Sub graded scenes G0-G4 for the retained tasks (proposal Sec. 6.1).

    G0 same instance                         -> IG L0 scene, L0 solver
    G1 same category, new instance           -> IG L2 scene, L0 solver
    G2 different category, same affordance   -> L0 scene with an asset remap, L0 solver
    G3 different affordance, new strategy    -> IG L3 scene, L3 solver
    G4 no feasible substitute (abstain)      -> IG L3 scene with the substitute removed

Built from the existing task template/envs without editing them: G2 swaps assets by
patching the loader calls the task module makes (create_actor / get_actor_builder), G4
parks the substitute actors out of reach after every reset (qap.scenes.graded).

G2 needs a *different category with the same affordance*; only some tasks have one in
the released asset sets. Entries marked status='proposed' are first guesses to be
validated with `python -m qap.scenes.build_graded_scenes --validate` (instantiation +
L0-solver success). Entries with g2=None have no G2 in IG-Sub (the per-grade analysis
then uses the remaining grades for that task). Edit here, re-validate, then freeze.

Remap keys:
    "ycb:<id>"            -> "ycb:<new id>"                 (actors.get_actor_builder)
    "robotwin:<name>"     -> {"modelname": ..., "model_id": int, "scale": [x,y,z]?}  (create_actor)
G4 'hide': list of actor names, or "@l3_only" = actors present at L3 but not at L0
(resolved from qap_runs/inventory/runtime_inventory.json).
"""

from __future__ import annotations

from typing import Dict, Optional

GradeSpec = Dict[str, object]

G2: Dict[str, Optional[GradeSpec]] = {
    "PlaceClothBasket": dict(remap={"robotwin:076_breadbasket": {"modelname": "062_plasticbox", "model_id": 0}},
                             note="basket -> plastic storage box (open container, drop-in)", status="proposed"),
    "PlaceCommodityRack": dict(remap={"robotwin:074_displaystand": {"modelname": "008_tray", "model_id": 0}},
                               note="shelf -> tray (flat support, place-on)", status="proposed"),
    # dropped after validation (job 67666): the L0 solver never succeeds with the swapped asset
    # (0/5), or the asset does not build. Kept here for the record.
    "PourKetchupFries": None,      # squeeze bottle -> ycb:022_windex_bottle: 0/5
    "TransFood": None,             # spoon -> ycb:033_spatula: 0/5 (spatula cannot scoop)
    "PlacePillBox": None,          # box -> robotwin 002_bowl id 0: create_actor returned None
    "PlaceBookBookcase": None,     # no other 'upright slot' container in the asset sets
    "PlaceMagazineFolder": None,   # folder is a PartNet articulation with no same-affordance sibling
    "PlacePlateRack": None,        # dumbbell-rack used as plate rack; no same-affordance alternative
    "PlaceFoodScale": None,        # weighing has no different-category same-affordance object
    "ScanPillBottle": None,        # only one handheld scanner category
    "PutCubeOnScale": None,
    "OpenLiquidCap": None,
    "PutBox": None,                # other hinged boxes are the same category (that is G1)
}


# G4 'hide' overrides where "@l3_only" cannot work: the L3 substitute has the same actor name
# (or category) as an L0 object, so it is named by the task attribute that holds it instead
# ("attr:<name>" -> env.unwrapped.<name>).
G4_HIDE: Dict[str, list] = {
    "TransFood": ["attr:bowl_src"],         # the source bowl that is tilted to pour
    "ScanPillBottle": ["attr:scanner"],     # the fixed desktop scanner (static actor)
    "PlacePillBox": ["attr:pill"],          # the pill bottle that is tipped to dispense
}


def grade_spec(task: str, grade: str) -> Optional[GradeSpec]:
    g = grade.upper()
    if g == "G0":
        return dict(level="L0", solver="L0", expected="copy")
    if g == "G1":
        return dict(level="L2", solver="L0", expected="copy")
    if g == "G2":
        spec = G2.get(task)
        if spec is None:
            return None
        return dict(level="L0", solver="L0", expected="substitute", **spec)
    if g == "G3":
        return dict(level="L3", solver="L3", expected="substitute")
    if g == "G4":
        return dict(level="L3", solver="L3", expected="abstain", hide=G4_HIDE.get(task, "@l3_only"))
    # plain IG levels are accepted too
    if g in ("L0", "L1", "L2"):
        return dict(level=g, solver="L0", expected="copy")
    if g == "L3":
        return dict(level="L3", solver="L3", expected="substitute")
    raise ValueError(grade)


GRADES = ("G0", "G1", "G2", "G3", "G4")
