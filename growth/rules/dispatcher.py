from core.audit import AuditLog
from .rule_merge import rule_merge
from .rule_branch import rule_branch
from .rule_stabilize import rule_stabilize
from .rule_revert import rule_revert
from .result import GrowthResult

RULE_MAP = {
    "merge": rule_merge,
    "branch": rule_branch,
    "stabilize": rule_stabilize,
    "revert": rule_revert,
}

def apply_rule(name: str, *args, **kwargs) -> tuple[GrowthResult, AuditLog]:
    if name not in RULE_MAP:
        raise ValueError(f"Unknown rule '{name}'. Valid: {list(RULE_MAP.keys())}")
    return RULE_MAP[name](*args, **kwargs)
