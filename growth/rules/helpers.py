from core.conservation import verify_conservation, conserve_ledger
from core.lattice import LatticeState

@conserve_ledger
def _blend_lattices(a: LatticeState, b: LatticeState, w: float = 0.5) -> LatticeState:
    s = a.clone()
    s.weights = (a.weights * w) + (b.weights * (1.0 - w))
    for x in range(3):
        for y in range(3):
            for z in range(3):
                ca, cb = a.cells[x, y, z], b.cells[x, y, z]
                s.cells[x, y, z] = ca if ca <= cb else cb
    s.normalize()
    assert verify_conservation(s), "Merged lattice violated ΣSW!"
    return s

def _clip_phi(phi: float) -> float:
    return float(max(-1.0, min(1.0, phi)))
