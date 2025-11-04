from dataclasses import dataclass
from core.lattice import LatticeState

@dataclass
class GrowthResult:
    new_state: LatticeState
    new_polarity: float
    rule: str
    note: str = ""
