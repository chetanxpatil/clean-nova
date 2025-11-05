# growth/mind/growth_mind.py
"""
Livnium Growth — growth_mind.py
===============================
Recursive tree-of-thought manager built on GrowthNode and rules.py.
This is our champion model, now with smarter meta-cognition.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import time
import numpy as np

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
    """
    Core recursive tree-of-thought controller.
    Orchestrates rule selection, node creation, and adaptive temperature control.
    """
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

    # --- STEP 7: META-COGNITION ATTRIBUTE ---
    std_dev_factor: float = 0.5  # This is no longer used, but fine to keep.

    # -------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "GrowthMind":
        """Instantiate GrowthMind from configuration dictionary."""
        mind = cls.from_state()
        mind.temperature = cfg.get("temperature", 0.04)
        mind.phi_damping = cfg.get("phi_damping", 0.95)
        mind.branch_var_threshold = cfg.get("branch_var_threshold", 0.02)
        mind.motif_eta = float(cfg.get("motif_eta", 0.15))
        return mind

    @staticmethod
    def from_state(state: Optional[LatticeState] = None, polarity: float = 1.0) -> "GrowthMind":
        """Create a GrowthMind starting from an initial LatticeState."""
        if state is None:
            state = canonical_symbol_layout()
        node = GrowthNode(state, polarity=polarity, depth=0, rule="origin")
        return GrowthMind(root=node, active=node)

    # -------------------------------------------------------------------
    # Core ToT control
    # -------------------------------------------------------------------

    def choose_rule(self, phi: float) -> str:
        """
        Our final, champion "Dumb Core" model.
        It uses a simple, static neutral band that is tuned
        by the meta-cognition step.
        """

        # --- ================================== ---
        # ---      OUR FINAL STABLE CORE       ---
        # --- ================================== ---

        band = float(self.neutral_band)  # This is 0.45 or our new, tuned value

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

    def step(
            self,
            intent: Optional[IntentVector] = None,
            note: str = "",
            external_correct: Optional[bool] = None
    ) -> GrowthNode:

        A = self.active.state
        if intent is None:
            intent = compute_intent(A, A, observer="Lo")

        phi = float(np.clip(intent.polarity, -1.0, 1.0)) * self.phi_damping

        # 'rule' is now *always* the base rule from the "dumb mind"
        rule = self.choose_rule(phi)

        parent_state = self.active.parent.state if self.active.parent else self.active.state

        # --- Apply Rule ---
        if rule == "merge":
            result, _ = apply_rule("merge", self.active.state, parent_state, intent=intent, log=self.active.log)
        elif rule == "branch":
            result, _ = apply_rule("branch", self.active.state, intent, log=self.active.log)
        elif rule == "stabilize":
            result, _ = apply_rule("stabilize", self.active.state, log=self.active.log)
        else:
            rule = "revert"
            result, _ = apply_rule("revert", self.active.state, parent_state, log=self.active.log)

        # --- Node creation ---
        child = GrowthNode(
            result.new_state,
            result.new_polarity * self.phi_damping,
            depth=self.active.depth + 1,
            rule=result.rule,
            note=result.note,
            parent=self.active,
            log=self.active.log
        )
        self.active.children.append(child)
        self.active = child

        assert verify_conservation(child.state), "ΣSW violated in GrowthMind.step"

        # --- FIELD UPDATE ---
        self.phi_field = child.state.weights
        self.last_phi = child.polarity
        current_entropy = self.policy.entropy(self.temperature)

        # --- SIMPLE REWARD ---
        if external_correct:
            total_reward = 1.0
        elif external_correct is False:
            total_reward = -1.0
        else:
            total_reward = 0.0

        # --- FINAL POLICY UPDATE ---
        self.policy.update(rule, total_reward)

        # --- ENTROPY-COUPLED TEMPERATURE FEEDBACK ---
        try:
            self.temperature *= (0.995 if current_entropy > 1.5 else 1.002)
            self.temperature = np.clip(self.temperature, 0.02, 0.12)
        except Exception:
            pass

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
        delta_pi = {
            k: round(current_pi[k] - self.last_logged_pi.get(k, 0.0), 6)
            for k in current_pi
        }
        self.last_logged_pi = current_pi

        delta_q_norm = float(np.linalg.norm(np.array(list(delta_pi.values())))) if delta_pi else 0.0

        self.journal.append({
            "t": time.time(),
            "rule": child.rule,
            "Φ": round(child.polarity, 6),
            "note": note or child.note,
            "Δπ": delta_pi,
            "ΔQ_norm": delta_q_norm,
            "entropy": current_entropy,
        })

        return child

    # -------------------------------------------------------------------
    # STEP 7: META-COGNITION
    # -------------------------------------------------------------------

    def metacog_reflect_and_tune(self, cm: np.ndarray, train_acc: float) -> None:
        """
        Think about our own thinking.
        Analyze the confusion matrix (cm) from the last run and
        tune the mind's internal NEUTRAL_BAND for the *next* run.
        """
        print("\n" + "=" * 20 + " META-COGNITION REFLECTION " + "=" * 20)

        # Calculate accuracy for "neutral"
        neutral_total = cm[1, :].sum()
        neutral_correct = cm[1, 1]

        if neutral_total == 0:
            print("🧠 [MetaCog] No neutral samples seen. No tuning applied.")
            return

        neutral_acc = neutral_correct / neutral_total
        print(f"🧠 [MetaCog] Last run total accuracy: {train_acc:.2%}")
        print(f"🧠 [MetaCog] Last run 'neutral' accuracy: {neutral_acc:.2%}")
        print(f"🧠 [MetaCog] Current neutral_band: {self.neutral_band:.3f}")

        # --- NEW "AMBITIOUS" META-COGNITION RULE ---
        # The mind now expects its neutral accuracy to be
        # reasonably close to its total accuracy.

        # We're lagging if neutral acc is 10% or more behind total acc
        is_lagging = neutral_acc < (train_acc - 0.10)

        # We're too cautious if we're *over*-performing on neutral
        is_too_cautious = neutral_acc > (train_acc + 0.05)

        if is_lagging:
            # We are being too aggressive (not guessing neutral enough).
            # WIDEN the neutral band to be more cautious.
            self.neutral_band = min(self.neutral_band + 0.05, 0.60)  # Cap at 0.60
            print(f"🧠 [MetaCog] RESULT: Neutral performance is lagging. Widening band to {self.neutral_band:.3f}")

        elif is_too_cautious:
            # We are being too cautious (guessing neutral too much).
            # NARROW the neutral band to be more aggressive.
            self.neutral_band = max(self.neutral_band - 0.05, 0.20)  # Min at 0.20
            print(f"🧠 [MetaCog] RESULT: Too cautious. Narrowing band to {self.neutral_band:.3f}")

        else:
            print("🧠 [MetaCog] RESULT: Performance is balanced. No tuning required.")

        print("=" * 66)