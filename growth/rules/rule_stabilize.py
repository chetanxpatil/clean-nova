from core.lattice import rebalance
from core.semantic import compute_intent
from core.audit import audit_cycle
from core.conservation import conserve_ledger, verify_conservation
from .result import GrowthResult

@conserve_ledger
def rule_stabilize(state, log=None, observer="Om"):
    before = state.clone()
    rebalance(state)
    phi = compute_intent(before, state, observer=observer).polarity
    assert verify_conservation(state)
    audit_cycle(before, state, "rule_stabilize", observer=observer, log=log)
    return GrowthResult(state, phi, "G3:stabilize"), log
