from core.semantic import IntentVector, compute_intent
from core.audit import AuditLog, audit_cycle
from core.conservation import conserve_ledger
from .helpers import _blend_lattices, _clip_phi
from .result import GrowthResult

@conserve_ledger
def rule_merge(A, B, intent=None, log=None, observer="Om"):
    if intent is None:
        intent = compute_intent(A, B, observer=observer)
    phi = _clip_phi(intent.polarity)
    weight = 0.5 + 0.5 * phi
    merged = _blend_lattices(A, B, w=weight)
    audit_cycle(A, merged, "rule_merge", observer=observer, log=log)
    return GrowthResult(merged, phi, "G1:merge", note=f"w={weight:.3f}"), log
