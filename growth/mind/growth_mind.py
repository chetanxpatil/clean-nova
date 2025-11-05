# growth/mind/growth_mind.py
"""
Livnium Growth — growth_mind.py
===============================
Recursive tree-of-thought manager built on GrowthNode and rules.py.

This file defines the core GrowthMind class, which inherits functionality
from various mixins:
  - GrowthMindMemoryMixin: Adapters for MemoryCoupling.
  - GrowthMindExpansionMixin: Motif-aware expansion logic.
  - GrowthMindIntrospectionMixin: Stats, reflection, and traversal.
  - GrowthMindPersistenceMixin: Saving and loading state.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import time, random, numpy as np

# --- Core Livnium Imports ---
from core.lattice import LatticeState, canonical_symbol_layout
from core.semantic import IntentVector, compute_intent
from core.conservation import verify_conservation

# --- Imports relative to the 'growth/' directory ---
from growth.node.core_node import GrowthNode
from growth.rules import apply_rule

# --- Top-level import ---
from growth.memory_coupling import MemoryCoupling

# --- Refactored Module Imports (relative to this 'mind/' dir) ---
from growth.mind.policy import PolicyPi
from growth.mind.growth_mind_memory import GrowthMindMemoryMixin
from growth.mind.growth_mind_expansion import GrowthMindExpansionMixin
from growth.mind.growth_mind_introspection import GrowthMindIntrospectionMixin
from growth.mind.growth_mind_persistence import GrowthMindPersistenceMixin


@dataclass
class GrowthMind(
    GrowthMindMemoryMixin,
    GrowthMindExpansionMixin,
    GrowthMindIntrospectionMixin,
    GrowthMindPersistenceMixin
):
    root: GrowthNode
    active: GrowthNode
    policy: PolicyPi = field(default_factory=PolicyPi)
    journal: List[dict] = field(default_factory=list)

    # metabolic parameters
    temperature: float = 0.07
    phi_damping: float = 0.95
    branch_var_threshold: float = 0.002

    # calibration parameters
    phi_sign: float = 1.0
    neutral_band: float = 0.45  # This is our "base" static band

    # --- Attributes for adaptive reward & field awareness ---
    last_phi: float = field(default=0.0)
    reward_history: List[float] = field(default_factory=list)
    phi_field: np.ndarray = field(default_factory=lambda: np.zeros((3, 3, 3)))
    last_logged_pi: Optional[Dict[str, float]] = None

    # --- External Memory Manager ---
    memory: MemoryCoupling = field(default_factory=lambda: MemoryCoupling(alpha=0.2, max_size=512))

    # Motif influence strength
    motif_eta: float = 0.15

    # -------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "GrowthMind":
        mind = cls.from_state()
        mind.temperature = cfg.get("temperature", 0.04)
        mind.phi_damping = cfg.get("phi_damping", 0.95)
        mind.branch_var_threshold = cfg.get("branch_var_threshold", 0.02)
        mind.motif_eta = float(cfg.get("motif_eta", 0.15))
        return mind

    @staticmethod
    def from_state(state: Optional[LatticeState] = None, polarity: float = 1.0) -> "GrowthMind":
        if state is None:
            state = canonical_symbol_layout()
        node = GrowthNode(state, polarity=polarity, depth=0, rule="origin")
        return GrowthMind(root=node, active=node)

    # -------------------------------------------------------------------
    # Core ToT control
    # -------------------------------------------------------------------
    def choose_rule(self, phi: float) -> str:
        """
        Our final, 53%-accurate "Dumb Mind" model.
        It uses a simple, static neutral band and IGNORES all
        Q-learning and adaptive "smart" logic.
        """
        band = float(self.neutral_band)  # 0.45

        # --- ================================== ---
        # ---      OUR FINAL 53% MODEL         ---
        # --- ================================== ---

        if phi > band:
            base = "merge"
        elif phi < -band:
            base = "branch"
        else:
            base = "stabilize"

        return base

        # --- ================================== ---
        # ---         END OF FINAL MODEL       ---
        # --- ================================== ---

        # (This Q-policy logic is now "dead code" because we return early)
        weighted = sorted(self.policy.Q.items(), key=lambda kv: (-kv[1], kv[0]))
        top = weighted[0][0]

        if random.random() < 0.05:  # 5% chance to try something random
            return random.choice(list(self.policy.Q.keys()))

        return base if self.policy.Q[base] >= self.policy.Q[top] else top

    def step(self, intent: Optional[IntentVector] = None, note: str = "",
             external_correct: Optional[bool] = None) -> GrowthNode:

        A = self.active.state
        if intent is None:
            intent = compute_intent(A, A, observer="Lo")

        phi = float(np.clip(intent.polarity, -1.0, 1.0)) * self.phi_damping

        # 'rule' is now *always* the base rule from the "dumb mind"
        rule = self.choose_rule(phi)

        # (This exploration code is now "dead" because choose_rule returns early)
        if random.random() < self.temperature * 4:
            # Exploration: override the 'best' rule
            probs = self.policy.softmax_probs(self.temperature)
            rule = np.random.choice(list(probs.keys()), p=list(probs.values()))

        # --- Apply Rule ---
        if rule == "merge":
            partner = self.active.parent.state if self.active.parent else self.active.state
            result, _ = apply_rule("merge", self.active.state, partner, intent=intent, log=self.active.log)
        elif rule == "branch":
            result, _ = apply_rule("branch", self.active.state, intent, log=self.active.log)
        elif rule == "stabilize":
            result, _ = apply_rule("stabilize", self.active.state, log=self.active.log)
        else:  # rule == "revert"
            rule = "revert"
            result, _ = apply_rule("revert", self.active.state,
                                   self.active.parent.state if self.active.parent else self.active.state,
                                   log=self.active.log)

        child = GrowthNode(result.new_state, result.new_polarity * self.phi_damping,
                           depth=self.active.depth + 1, rule=result.rule,
                           note=result.note, parent=self.active, log=self.active.log)
        self.active.children.append(child)
        self.active = child
        assert verify_conservation(child.state), "ΣSW violated in GrowthMind.step"

        # --- FIELD UPDATE ---
        self.phi_field = child.state.weights
        self.last_phi = child.polarity  # Still useful to track this
        current_entropy = self.policy.entropy(self.temperature)  # Still useful for logging

        # --- ================================== ---
        # ---         SIMPLE REWARD            ---
        # --- =S================================= ---

        # We still calculate a reward so we can *plot* the
        # (unused) Q-policy learning in the background.

        if external_correct:
            total_reward = 1.0
        elif external_correct is False:
            total_reward = -1.0
        else:
            total_reward = 0.0  # No reward if no external signal

        # --- ================================== ---
        # ---          END REWARD              ---
        # --- ================================== ---

        # --- FINAL POLICY UPDATE ---
        self.policy.update(rule, total_reward)

        # --- ENTROPY-COUPLED TEMPERATURE FEEDBACK ---
        # (This block is fine, as it only affects exploration temperature)
        try:
            self.temperature *= (0.995 if current_entropy > 1.5 else 1.002)
            self.temperature = np.clip(self.temperature, 0.02, 0.12)
        except Exception:
            pass
        # --- END TEMPERATURE FEEDBACK ---

        # --- JOURNALING ---
        if self.last_logged_pi is None:
            self.last_logged_pi = dict(self.policy.Q)
            self.journal.append({
                "t": time.time(),
                "event": "init_policy",
                "π": dict(self.last_logged_pi),
                "entropy": current_entropy
            })

        current_pi = dict(self.policy.Q)
        delta_pi = {k: round(current_pi[k] - self.last_logged_pi.get(k, 0.0), 6)
                    for k in current_pi}
        self.last_logged_pi = current_pi

        delta_q_norm = float(np.linalg.norm(np.array(list(delta_pi.values())))) if delta_pi else 0.0

        self.journal.append({
            "t": time.time(),
            "rule": child.rule,
            "Φ": round(child.polarity, 6),
            "note": note or child.note,
            "Δπ": delta_pi,
            "ΔQ_norm": delta_q_norm,
            "entropy": current_entropy
        })

        return child