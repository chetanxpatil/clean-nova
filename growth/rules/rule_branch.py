import numpy as np
from core.rotation import rotate_sequence
from core.semantic import IntentVector, compute_intent
from core.audit import AuditLog, audit_cycle
from core.conservation import conserve_ledger, verify_conservation
from .helpers import _clip_phi
from .result import GrowthResult

@conserve_ledger
def rule_branch(parent, intent, divergence=0.5, log=None, observer="Lo"):
    phi = _clip_phi(intent.polarity)
    if phi >= 0:
        return GrowthResult(parent.clone(), phi, "G2:noop", note="no divergence"), log or AuditLog()
    k = int(round(abs(phi) * 3 * divergence)) or 1
    seq = "XYZ"[:k]
    new_state = rotate_sequence(parent, seq)
    new_state.normalize()
    assert verify_conservation(new_state)
    audit_cycle(parent, new_state, f"rule_branch({seq})", observer=observer, log=log)
    return GrowthResult(new_state, phi, "G2:branch", note=f"seq={seq}"), log
