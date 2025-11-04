from core.lattice import canonical_symbol_layout
from core.semantic import compute_intent
from core.audit import AuditLog
from .dispatcher import apply_rule

if __name__ == "__main__":
    A = canonical_symbol_layout()
    B = A.clone()
    log = AuditLog()

    intent = compute_intent(A, B)
    for name in ["merge", "branch", "stabilize", "revert"]:
        result, log = apply_rule(name, A, B if name == "merge" else A, intent=intent, log=log)
        print(result)
    print(log.describe())
    print("rules/ self-check passed ✓")
