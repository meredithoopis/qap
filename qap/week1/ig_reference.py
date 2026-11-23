"""IG per-task simulation success (Zhou et al., arXiv:2608.22301, Appendix H, Table 11).

Success rate (%) over 10 evaluation episodes, DP and ACT trained on the full IG-10K
simulation corpus (50 tasks x 4 levels). Transcribed from the PDF text layer; the
rows were matched to the alphabetical task column and cross-checked against the
paper's own narrative (FoldBox/GrindFood/PlaceMugRack/PickRemoteControl high;
PickRemoteControl drops at L3; PlaceCupPlate/PlaceFoodScale/PlaceCommodityRack low)
and against the 'Mean' columns, which equal the row means for every row.
Re-check against page 32 of the PDF before freezing if in doubt.
"""

from __future__ import annotations

from typing import Dict, Tuple

# task: (DP L0, L1, L2, L3, ACT L0, L1, L2, L3)
TABLE11: Dict[str, Tuple[int, ...]] = {
    "CleanCup":                 (40, 50, 20, 100, 30, 30, 70, 90),
    "CleanDesk":                (10, 60, 40, 60, 80, 100, 100, 100),
    "CutFruit":                 (100, 50, 0, 100, 100, 100, 0, 100),
    "FoldBox":                  (100, 100, 100, 100, 100, 100, 100, 100),
    "FoldTowel":                (60, 40, 10, 40, 100, 100, 100, 80),
    "GrindFood":                (100, 100, 100, 100, 90, 100, 100, 70),
    "KnifeBowlFork":            (0, 60, 60, 90, 100, 100, 80, 50),
    "LiftLidFromSkillet":       (0, 80, 80, 100, 100, 100, 100, 100),
    "OpenBox":                  (100, 90, 0, 100, 100, 100, 0, 100),
    "OpenLiquidCap":            (60, 50, 70, 50, 90, 60, 90, 0),
    "PickAppleBananaToBaskets": (50, 100, 50, 100, 90, 90, 50, 50),
    "PickAppleBasket":          (70, 70, 100, 80, 100, 90, 100, 100),
    "PickAppleToScale":         (40, 90, 40, 80, 90, 10, 90, 80),
    "PickFood":                 (40, 100, 10, 90, 100, 70, 30, 90),
    "PickFruitsToPlate":        (90, 90, 100, 50, 20, 40, 100, 0),
    "PickPillToRegions":        (40, 60, 50, 10, 90, 90, 70, 80),
    "PickRemoteControl":        (100, 100, 100, 40, 100, 100, 100, 80),
    "PickTennisBallGolfBall":   (90, 90, 90, 100, 60, 80, 70, 50),
    "PickWash":                 (70, 100, 30, 80, 90, 100, 60, 90),
    "PlaceBookBookcase":        (100, 100, 90, 90, 100, 80, 100, 30),
    "PlaceBrushRest":           (90, 70, 100, 100, 80, 100, 100, 100),
    "PlaceBurgerTray":          (80, 90, 80, 70, 90, 80, 90, 70),
    "PlaceChipsRack":           (90, 80, 100, 80, 70, 70, 70, 100),
    "PlaceClothBasket":         (100, 100, 80, 10, 100, 100, 90, 90),
    "PlaceCommodityRack":       (40, 50, 30, 30, 100, 100, 90, 0),
    "PlaceCupPlate":            (40, 20, 70, 30, 50, 60, 70, 0),
    "PlaceFileFolder":          (80, 70, 100, 80, 80, 80, 60, 100),
    "PlaceFoodScale":           (40, 20, 90, 0, 60, 100, 100, 0),
    "PlaceFruitBox":            (80, 20, 90, 30, 70, 30, 90, 60),
    "PlaceMagazineFolder":      (80, 100, 0, 70, 100, 100, 0, 70),
    "PlaceMugRack":             (100, 70, 80, 90, 100, 100, 80, 100),
    "PlacePillBox":             (80, 70, 60, 30, 100, 60, 80, 70),
    "PlacePlateRack":           (90, 40, 30, 80, 100, 90, 50, 80),
    "PlaceScrewdriver":         (30, 20, 30, 30, 100, 80, 40, 100),
    "PlaceShoeBox":             (90, 90, 60, 100, 100, 100, 90, 90),
    "PourCup":                  (10, 100, 100, 90, 100, 100, 100, 100),
    "PourKetchupFries":         (50, 50, 100, 30, 100, 100, 80, 60),
    "PourKettle":               (90, 80, 0, 80, 100, 90, 0, 100),
    "PourLiquidCup":            (10, 0, 0, 10, 10, 10, 0, 0),
    "PourLiquidFilter":         (60, 60, 20, 50, 100, 100, 90, 90),
    "PourLiquidMug":            (60, 10, 100, 20, 0, 90, 70, 0),
    "PressJuicer":              (60, 80, 0, 100, 20, 100, 0, 100),
    "PressStapler":             (100, 70, 100, 90, 50, 90, 40, 40),
    "PutBox":                   (100, 90, 90, 10, 100, 90, 100, 100),
    "PutCubeOnScale":           (100, 90, 90, 60, 100, 100, 100, 70),
    "ScanMilkBox":              (100, 100, 100, 90, 100, 100, 100, 100),
    "ScanPillBottle":           (90, 50, 90, 10, 100, 100, 90, 0),
    "StirSpoon":                (20, 0, 40, 10, 50, 50, 40, 80),
    "TransFood":                (50, 90, 30, 30, 60, 90, 0, 20),
    "WipePot":                  (10, 0, 0, 100, 20, 0, 0, 80),
}


def level_success(task: str, level: str, model: str = "mean") -> float:
    row = TABLE11[task]
    i = int(level[-1])
    dp, act = row[i], row[4 + i]
    return {"dp": dp, "act": act, "mean": (dp + act) / 2.0}[model.lower()]


def l0_minus_l3(task: str, model: str = "mean") -> float:
    return level_success(task, "L0", model) - level_success(task, "L3", model)
