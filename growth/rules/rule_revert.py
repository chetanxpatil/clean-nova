from core.semantic import compute_intent
from core.audit import audit_cycle
from core.conservation import conserve_ledger
from .result import GrowthResult

@conserve_ledger
def rule_revert(current, previous, log=None, observer="Om"):
    phi = compute_intent(current, previous, observer=observer).polarity
    audit_cycle(current, previous, "rule_revert", observer=observer, log=log)
    return GrowthResult(previous.clone(), phi, "G4:revert"), log
