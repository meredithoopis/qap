"""Curated task inventory (Week 1).

One entry per IG task: what the human demonstrates (L0), what replaces it at L3, the
motion primitive at L0 vs L3, and reversibility. Written from the L0/L3 task
descriptions in examples/baselines/lerobot_dataset/task_desc/sim_desc.json and the
L0/L3 environment sources. Columns that can be computed (asset ids, substitute
coverage in other tasks, predicate availability, IG success) are added by
qap.week1.inventory, not typed here.

Definitions used for the screening rule
---------------------------------------
motion_changed  : the core motion primitive differs (e.g. insert->stack, scoop->pour,
                  hinge-close->slide-close, handheld-tool->bring-to-fixture), or L3 adds /
                  removes a step (unblocking, counterweight). A pure grasp change (handle vs
                  body) is recorded separately in grasp_changed and does NOT count.
intent_preserved: the L3 description pursues the same goal as L0 (False for
                  PressStapler L3 = filing, PressJuicer L3 = grinding).
reversibility   : 'irreversible' = physical change of the patient the robot cannot undo
                  (pour, cut, dispense, clean, grind); 'event' = latched predicate without a
                  lasting physical change (scan, stir, press); 'reversible' otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class InvEntry:
    name: str
    demo: str                 # demonstrated object that L3 replaces (or the changed element)
    sub: str                  # L3 substitute
    role: str                 # tool | container | support | fixture | patient | articulated | source
    prim_l0: str
    prim_l3: str
    motion_changed: bool
    grasp_changed: bool
    intent_preserved: bool
    reversibility: str
    notes: str = ""

    def row(self) -> Dict:
        return asdict(self)


E = InvEntry
INVENTORY: List[InvEntry] = [
    E("StirSpoon", "spoon", "chopsticks", "tool", "grasp tool + stir in bowl + return", "grasp tool + stir in bowl + return", False, True, True, "event"),
    E("PlaceBookBookcase", "bookcase", "storage box (occluded by a book)", "container", "pick book + insert upright into shelf", "unblock (move book) + pick + place into box", True, False, True, "reversible", "adds an unblocking step"),
    E("PlaceClothBasket", "bamboo basket", "storage box (occluded by a book)", "container", "pick clothes + drop into basket", "unblock (move book) + pick + insert into box", True, False, True, "reversible", "adds an unblocking step"),
    E("PickRemoteControl", "table surface (source)", "black box (elevated source)", "source", "pick + place into box", "pick from elevated support + place into box", False, False, True, "reversible", "unseen eval task"),
    E("PlaceMagazineFolder", "folder (articulated)", "white magazine (flat support)", "container", "open folder (bimanual) + insert magazine", "pick magazine + stack on magazine", True, False, True, "reversible"),
    E("PickWash", "free spot between bottles", "storage box", "container", "pick + place on surface", "pick + place into box", False, False, True, "reversible"),
    E("PickAppleBasket", "fruit basket", "storage box", "container", "pick + place into container", "pick + place into container", False, False, True, "reversible"),
    E("PickAppleBananaToBaskets", "fruit baskets", "storage boxes", "container", "bimanual pick + place into containers", "bimanual pick + place into containers", False, False, True, "reversible"),
    E("PickAppleToScale", "electronic scale", "balance scale + weight", "fixture", "pick + place on scale", "pick + place on pan + other arm places counterweight", True, False, True, "reversible", "adds a counterweight step"),
    E("PlaceChipsRack", "chips (instance)", "chips (other instance)", "patient", "pick + place on shelf", "pick + place on shelf", False, False, True, "reversible", "L3 is effectively an instance swap"),
    E("PlaceCommodityRack", "shelf + packaged goods", "mug racks + cups", "fixture", "pick + place on shelf (x2)", "pick + hang cup on rack (x2)", True, True, True, "reversible"),
    E("PlaceFruitBox", "fruit basket (source)", "storage box (source)", "source", "pick + place into box (x2)", "pick + place into box (x2)", False, False, True, "reversible"),
    E("ScanMilkBox", "handheld scanner gun", "desktop scanner (fixed)", "tool", "bimanual: grasp scanner + present product", "unimanual: pick product + present to fixed scanner", True, False, True, "event", "unseen eval task"),
    E("PlacePlateRack", "plate rack", "plate (bowl stacked on it)", "fixture", "pick plate + insert upright into rack", "pick bowl + stack on plate", True, True, True, "reversible", "patient also changes (plate->bowl)"),
    E("CutFruit", "kitchen knife", "scissors (pre-grasped)", "tool", "grasp knife + slice + return", "snip with pre-grasped scissors", True, True, True, "irreversible"),
    E("PickFruitsToPlate", "round plate", "square plate", "container", "bimanual pick + place on plate", "bimanual pick + place on plate", False, False, True, "reversible"),
    E("PourKetchupFries", "squeeze bottle", "pump dispenser", "tool", "grasp bottle + squeeze over fries + set down", "press pump over fries (no grasp)", True, True, True, "irreversible"),
    E("WipePot", "rag", "brush", "tool", "grasp rag + wipe pot x3", "grasp brush + scrub pot x3", False, True, True, "irreversible"),
    E("CleanDesk", "rag", "tissue", "tool", "grasp rag + wipe stain", "grasp tissue + wipe stain", False, False, True, "irreversible"),
    E("PourKettle", "kettle (handle)", "water bottle (body grasp)", "tool", "grasp handle + tilt-pour", "grasp body + tilt-pour", False, True, True, "irreversible", "unseen eval task"),
    E("PickFood", "trash bin", "bamboo basket", "container", "bimanual pick + drop into bin", "bimanual pick + place into basket", False, False, True, "reversible", "unseen eval task"),
    E("TransFood", "spoon", "source bowl (poured)", "tool", "grasp spoon + scoop + deposit", "grasp bowl + tilt-pour into bowl", True, True, True, "irreversible"),
    E("PlaceFoodScale", "electronic scale", "balance scale + weight", "fixture", "pick + place on scale", "pick + place on pan + other arm places counterweight", True, False, True, "reversible", "adds a counterweight step"),
    E("PressStapler", "stapler", "folder", "tool", "press stapler on document", "pick document + insert into folder", True, True, False, "irreversible", "intent changes (stapling -> filing)"),
    E("PlaceBurgerTray", "tray", "plate", "container", "pick + place into tray", "pick + place onto plate", False, False, True, "reversible"),
    E("PickTennisBallGolfBall", "lidded boxes", "open storage boxes", "container", "pick + place into box + close lid (x2)", "pick + place into box (x2)", True, False, True, "reversible", "L3 drops the lid-closing step"),
    E("ScanPillBottle", "handheld scanner gun", "desktop scanner (fixed)", "tool", "bimanual: grasp scanner + present bottle", "pick bottle + present to fixed scanner", True, False, True, "event"),
    E("PlaceShoeBox", "shoe box", "storage box", "container", "bimanual pick + place into box", "bimanual pick + place into box", False, False, True, "reversible"),
    E("OpenBox", "large box", "small box", "articulated", "lift lid open", "lift lid open", False, False, True, "reversible", "L3 is effectively an instance swap"),
    E("FoldBox", "cardboard box flaps", "storage box lid", "articulated", "push-fold flaps closed", "push lid closed", False, False, True, "reversible", "unseen eval task"),
    E("PickPillToRegions", "table regions", "double-layer shelf", "support", "bimanual pick + place in region", "bimanual pick + place on shelf layer", False, False, True, "reversible"),
    E("PlaceFileFolder", "folder (articulated)", "tray", "container", "open folder (bimanual) + insert document", "pick document + place into tray", True, False, True, "reversible", "L3 drops the opening step"),
    E("PourLiquidCup", "cup", "mug", "container", "grasp bottle + tilt-pour", "grasp bottle + tilt-pour", False, False, True, "irreversible", "L3 omitted in the real suite (hazardous)"),
    E("PlacePillBox", "loose pill", "pill bottle (dispensed)", "patient", "pick pill + drop into box", "grasp bottle + tip to dispense into box", True, True, True, "irreversible"),
    E("PlaceBrushRest", "brush rest", "brush pot", "support", "pick brush + lay on rest", "pick brush + insert upright into pot", True, False, True, "reversible"),
    E("PlaceCupPlate", "tray", "double-layer rack", "support", "bimanual pick + place on tray", "bimanual pick + place on rack layer", False, False, True, "reversible"),
    E("PlaceScrewdriver", "screwdriver", "hammer", "tool", "grasp tool + screw (rotate) into block", "grasp tool + strike block", True, True, True, "event"),
    E("PlaceMugRack", "mug rack", "plate", "fixture", "pick mug + hang on rack", "pick mug + place on plate", True, False, True, "reversible"),
    E("PourLiquidMug", "mug / test tube (instance)", "cup / test tube (other)", "container", "grasp tube + tilt-pour + return", "grasp tube + tilt-pour + return", False, False, True, "irreversible"),
    E("PutCubeOnScale", "electronic scale", "balance scale + second object", "fixture", "pick + place on scale", "pick + place on pan + other arm places counterweight", True, False, True, "reversible", "adds a counterweight step"),
    E("PourCup", "cup", "cup with filter", "container", "grasp mug + pour + return", "grasp mug + pour + return", False, False, True, "irreversible"),
    E("CleanCup", "brush", "whiteboard eraser", "tool", "grasp brush + scrub cup", "grasp eraser + scrub cup", False, True, True, "irreversible"),
    E("OpenLiquidCap", "screw cap", "cup used as a lid", "articulated", "unscrew cap (rotate) + place on plate", "lift cup off bottle + place on plate", True, True, True, "reversible"),
    E("GrindFood", "grinding stick", "bottle used as pestle", "tool", "grasp stick + grind in bowl", "grasp bottle + grind in bowl", False, True, True, "irreversible"),
    E("PourLiquidFilter", "mug", "kettle (handle)", "tool", "grasp mug + pour through filter", "grasp kettle + pour through filter", False, True, True, "irreversible"),
    E("PutBox", "hinged-lid box", "sliding-lid storage box", "container", "pick + place into box + close hinged lid", "pick + place into box + slide lid shut", True, False, True, "reversible"),
    E("KnifeBowlFork", "knife", "chopsticks (in bowl)", "patient", "bimanual pick + place cutlery beside bowl", "bimanual pick + place cutlery beside bowl", False, True, True, "reversible"),
    E("LiftLidFromSkillet", "skillet lid", "insulated-bag zipper", "articulated", "grasp lid + lift + set down", "unzip / open bag (articulated)", True, True, True, "reversible"),
    E("FoldTowel", "storage box (instance)", "storage box (other)", "container", "fold towel + pick + place into box", "fold towel + pick + place into box", False, False, True, "reversible"),
    E("PressJuicer", "juicer button", "grinding stick + bowl", "tool", "press button", "grasp stick + grind fruit", True, True, False, "event", "intent changes (press -> grind)"),
]

INVENTORY_BY_NAME: Dict[str, InvEntry] = {e.name: e for e in INVENTORY}
