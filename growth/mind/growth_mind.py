# growth/mind/growth_mind.py
"""
Livnium Growth — growth_mind.py
===============================
Recursive tree-of-thought manager built on GrowthNode and rules.py.
This is our champion model, now with smarter meta-cognition and R2R learning.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import time
import numpy as np
import random  # kept for parity; may be used by rules/policy elsewhere

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


# -------------------------------------------------------------------
# Tunables & constants (kept equal to original semantics)
# -------------------------------------------------------------------

_TEMP_MIN = 0.02
_TEMP_MAX = 0.12

# Risk-to-Reward (R2R) curiosity shaping constants
_R2R_MULTIPLIER = 0.1
_R2R_MAX_BONUS = 0.5


# -------------------------------------------------------------------
# GrowthMind
# -------------------------------------------------------------------


@dataclass
class GrowthMind(
    GrowthMindMemoryMixin,
    GrowthMindExpansionMixin,
    GrowthMindIntrospectionMixin,
    GrowthMindPersistenceMixin,
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
    memory: MemoryCoupling = field(
        default_factory=lambda: MemoryCoupling(alpha=0.2, max_size=512)
    )

    # Motif influence strength
    motif_eta: float = 0.15

    # --- STEP 7: META-COGNITION ATTRIBUTE ---
    std_dev_factor: float = 0.5

    # -------------------------------------------------------------------
    # Construction
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

    # @staticmethod
    # def from_state(
    #     state: Optional[LatticeState] = None, polarity: float = 1.0
    # ) -> "GrowthMind":
    #     """Create a GrowthMind starting from an initial LatticeState."""
    #     if state is None:
    #         state = canonical_symbol_layout()
    #     node = GrowthNode(state, polarity=polarity, depth=0, rule="origin")
    #     return GrowthMind(root=node, active=node)

    @staticmethod
    def from_state(
            state: Optional[LatticeState] = None, polarity: float = 1.0
    ) -> "GrowthMind":
        """Create a GrowthMind starting from an initial LatticeState."""
        if state is None:
            state = canonical_symbol_layout()

        # --- FIX: Explicitly set node_id=0 and parent_id=-1 for the Root ---
        node = GrowthNode(state,
                          node_id=0,  # The first node gets ID 0
                          parent_id=-1,  # The root has no parent
                          polarity=polarity,
                          depth=0,
                          rule="origin")
        # -------------------------------------------------------------------

        return GrowthMind(root=node, active=node)

    # -------------------------------------------------------------------
    # Core ToT control
    # -------------------------------------------------------------------

    def choose_rule(self, phi: float) -> str:
        """
        Final, champion "Dumb Core" model:
        uses a simple neutral band (tuned by meta-cognition).
        """
        band = float(self.neutral_band)  # 0.45 or tuned value
        if phi > band:
            return "merge"
        if phi < -band:
            return "branch"
        return "stabilize"

    # def step(
    #     self,
    #     intent: Optional[IntentVector] = None,
    #     note: str = "",
    #     external_correct: Optional[bool] = None
    # ) -> GrowthNode:
    #     """
    #     One thinking step:
    #     - compute/clip φ
    #     - choose/apply rule
    #     - create child node
    #     - compute reward (accuracy + R2R)
    #     - update policy and temperature
    #     - journal deltas
    #     """
    #     A = self.active.state
    #     if intent is None:
    #         intent = compute_intent(A, A, observer="Lo")
    #
    #     # φ after damping (kept identical to original)
    #     phi = float(np.clip(intent.polarity, -1.0, 1.0)) * self.phi_damping
    #
    #     # Decision by core
    #     rule = self.choose_rule(phi)
    #
    #     # Apply rule & create child
    #     result = self._apply_rule(rule, intent)
    #     child = self._spawn_child_from_result(result, note)
    #     self._post_apply_invariants(child)
    #
    #     # Update field & diagnostics
    #     self._update_field_and_entropy(child)
    #
    #     # Reward shaping (base accuracy + curiosity)
    #     total_reward = self._compute_total_reward(external_correct=external_correct, phi=phi)
    #
    #     # Final policy update (same timing/order)
    #     self.policy.update(rule, total_reward)
    #
    #     # Temperature feedback (entropy-coupled)
    #     self._update_temperature()
    #
    #     # Journal (policy deltas + entropy snapshot)
    #     self._journal_step(child, note)
    #
    #     return child
    def step(
        self,
        intent: Optional[IntentVector] = None,
        note: str = "",
        external_correct: Optional[bool] = None,
    ) -> GrowthNode:
        """
        One thinking step:
        - compute/clip φ
        - choose/apply rule
        - create child node (with persistent IDs)
        - compute reward (accuracy + R2R)
        - update policy and temperature
        - journal deltas (with node structure)
        """
        A = self.active.state
        if intent is None:
            intent = compute_intent(A, A, observer="Lo")

        # φ after damping (kept identical to original)
        phi = float(np.clip(intent.polarity, -1.0, 1.0)) * self.phi_damping

        # Decision by core
        rule = self.choose_rule(phi)

        # -------------------------------------------------------------------
        # 1. CALCULATE NODE IDs AND PACK PARAMS FOR CHILD SPAWNING
        # -------------------------------------------------------------------
        # The new node's ID is the next index in the journal (simple sequential counter)
        next_node_id = len(self.journal)

        # The parent ID is the ID of the current active node
        parent_node_id = self.active.node_id

        # Pack the IDs to pass to the helper
        node_params = {"node_id": next_node_id, "parent_id": parent_node_id}

        # -------------------------------------------------------------------

        # Apply rule & create child
        result = self._apply_rule(rule, intent)
        # PASS THE NEW node_params TO SPAWN THE CHILD
        child = self._spawn_child_from_result(result, note, **node_params)
        self._post_apply_invariants(child)

        # Update field & diagnostics
        self._update_field_and_entropy(child)

        # Reward shaping (base accuracy + curiosity)
        total_reward = self._compute_total_reward(
            external_correct=external_correct, phi=phi
        )

        # Final policy update (same timing/order)
        self.policy.update(rule, total_reward)

        # Temperature feedback (entropy-coupled)
        self._update_temperature()

        # Journal (policy deltas + entropy snapshot)
        self._journal_step(child, note)

        return child

    # -------------------------------------------------------------------
    # META-COGNITION (unchanged logic, tidied prints)
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

        neutral_acc = neutral_correct / max(1, neutral_total)
        print(f"🧠 [MetaCog] Last run total accuracy: {train_acc:.2%}")
        print(f"🧠 [MetaCog] Last run 'neutral' accuracy: {neutral_acc:.2%}")
        print(f"🧠 [MetaCog] Current neutral_band: {self.neutral_band:.3f}")

        # --- FINE-GRAINED TUNING LOGIC ---
        TUNING_STEP = 0.01

        # Lagging if neutral acc is 10%+ behind total
        is_lagging = neutral_acc < (train_acc - 0.10)

        # Too cautious if neutral acc 5% above total
        is_too_cautious = neutral_acc > (train_acc + 0.05)

        if is_lagging:
            # WIDEN the neutral band (be more cautious)
            self.neutral_band = min(self.neutral_band + TUNING_STEP, 0.60)
            print(
                f"🧠 [MetaCog] RESULT: Neutral lagging → widen to {self.neutral_band:.3f}"
            )
        elif is_too_cautious:
            # NARROW the neutral band (be more aggressive)
            self.neutral_band = max(self.neutral_band - TUNING_STEP, 0.20)
            print(
                f"🧠 [MetaCog] RESULT: Too cautious → narrow to {self.neutral_band:.3f}"
            )
        else:
            print("🧠 [MetaCog] RESULT: Balanced. No tuning required.")

        print("=" * 66)

    # -------------------------------------------------------------------
    # Internal helpers (modularized; preserve original semantics/order)
    # -------------------------------------------------------------------

    def _apply_rule(self, rule: str, intent: IntentVector):
        """
        Apply the chosen rule to the active node/state.
        Mirrors the original if/elif chain including 'revert' fallback.
        """
        parent_state = (
            self.active.parent.state if self.active.parent else self.active.state
        )

        if rule == "merge":
            result, _ = apply_rule(
                "merge",
                self.active.state,
                parent_state,
                intent=intent,
                log=self.active.log,
            )
        elif rule == "branch":
            result, _ = apply_rule(
                "branch", self.active.state, intent, log=self.active.log
            )
        elif rule == "stabilize":
            result, _ = apply_rule("stabilize", self.active.state, log=self.active.log)
        else:
            # Fallback: revert
            result, _ = apply_rule(
                "revert", self.active.state, parent_state, log=self.active.log
            )

        return result

    #
    # def _spawn_child_from_result(self, result, note: str) -> GrowthNode:
    #     """
    #     Create the next GrowthNode from an apply_rule result, damping polarity.
    #     """
    #     child = GrowthNode(
    #         result.new_state,
    #         result.new_polarity * self.phi_damping,
    #         depth=self.active.depth + 1,
    #         rule=result.rule,
    #         note=result.note if result.note else note,
    #         parent=self.active,
    #         log=self.active.log
    #     )
    #     self.active.children.append(child)
    #     self.active = child
    #     return child

    # def _spawn_child_from_result(self, result, note: str, **kwargs) -> GrowthNode:
    #     """
    #     Create the next GrowthNode from an apply_rule result, damping polarity.
    #     Accepts **kwargs (which contains node_id and parent_id) to pass to GrowthNode.
    #     """
    #     child = GrowthNode(
    #         result.new_state,
    #         result.new_polarity * self.phi_damping,
    #         depth=self.active.depth + 1,
    #         rule=result.rule,
    #         note=result.note if result.note else note,
    #         parent=self.active,
    #         log=self.active.log,
    #         **kwargs,  # Pass node_id and parent_id here
    #     )
    #     self.active.children.append(child)
    #     # self.active = child
    #     return child

    # --- UPDATE IN growth/mind/growth_mind.py ---

    def _spawn_child_from_result(self, result, note: str, **kwargs) -> GrowthNode:
        """
        Create the next GrowthNode from an apply_rule result, damping polarity.
        NOTE: Removed self.active = child to enable true branching in ToT.
        """
        child = GrowthNode(
            result.new_state,
            result.new_polarity * self.phi_damping,
            depth=self.active.depth + 1,
            rule=result.rule,
            note=result.note if result.note else note,
            parent=self.active,
            log=self.active.log,
            **kwargs,  # Pass node_id and parent_id here
        )
        self.active.children.append(child)
        # -------------------------------------------------------------
        # REMOVED: self.active = child
        # The active node must be updated externally by the search logic.
        # -------------------------------------------------------------
        return child

    def _post_apply_invariants(self, child: GrowthNode) -> None:
        """
        Run invariants and store quick references after node transition.
        """
        assert verify_conservation(child.state), "ΣSW violated in GrowthMind.step"

    def _update_field_and_entropy(self, child: GrowthNode) -> None:
        """
        Keep φ-field and diagnostics synced.
        """
        self.phi_field = child.state.weights
        self.last_phi = child.polarity
        # compute and cache current policy entropy for journaling/feedback
        self._current_entropy = self.policy.entropy(self.temperature)

    def _compute_total_reward(
        self, *, external_correct: Optional[bool], phi: float
    ) -> float:
        """
        Original reward composition:
        1) base accuracy (+1/-1)
        2) curiosity bonus with R2R
        """
        total_reward = 0.0

        # 1. Base accuracy reward (extrinsic)
        if external_correct:
            total_reward += 1.0
        elif external_correct is False:
            total_reward -= 1.0

        # 2. Curiosity / R2R bonus (intrinsic)
        confidence = abs(phi)
        uncertainty = 1.0 - confidence
        curiosity_bonus = _R2R_MAX_BONUS * uncertainty * _R2R_MULTIPLIER
        total_reward += curiosity_bonus

        return total_reward

    def _update_temperature(self) -> None:
        """
        Entropy-coupled temperature feedback (kept identical to original).
        """
        try:
            self.temperature *= 0.995 if self._current_entropy > 1.5 else 1.002
            self.temperature = float(np.clip(self.temperature, _TEMP_MIN, _TEMP_MAX))
        except Exception:
            # stay fail-quiet as in original
            pass

    def _journal_step(self, child: GrowthNode, note: str) -> None:
        """
        Policy delta journaling with entropy snapshot; initialize once.
        --- NOW INCLUDES NODE ID, PARENT ID, AND LATTICE STATE DATA ---
        """
        # ... (initialization logic remains unchanged) ...
        if self.last_logged_pi is None:
            self.last_logged_pi = dict(self.policy.Q)
            self.journal.append(
                {
                    "t": time.time(),
                    "event": "init_policy",
                    "π": dict(self.last_logged_pi),
                    "entropy": self._current_entropy,
                }
            )

        # ... (delta_pi and delta_q_norm calculation remains unchanged) ...
        current_pi = dict(self.policy.Q)
        delta_pi = {
            k: round(current_pi[k] - self.last_logged_pi.get(k, 0.0), 6)
            for k in current_pi
        }
        self.last_logged_pi = current_pi

        delta_q_norm = (
            float(np.linalg.norm(np.array(list(delta_pi.values()))))
            if delta_pi
            else 0.0
        )

        self.journal.append(
            {
                "t": time.time(),
                # --- NEW FIELDS FOR VISUALIZATION ---
                "node_id": child.node_id,
                "parent_id": child.parent_id,
                "depth": child.depth,
                # --- CRITICAL LATTICE STATE FOR HEATMAP ---
                "phi_field_flat": child.state.weights.flatten().tolist(),
                # ------------------------------------
                "rule": child.rule,
                "Φ": round(child.polarity, 6),
                "note": note or child.note,
                "Δπ": delta_pi,
                "ΔQ_norm": delta_q_norm,
                "entropy": self._current_entropy,
            }
        )

    # def _journal_step(self, child: GrowthNode, note: str) -> None:
    #     """
    #     Policy delta journaling with entropy snapshot; initialize once.
    #     """
    #     # Initialize policy snapshot once
    #     if self.last_logged_pi is None:
    #         self.last_logged_pi = dict(self.policy.Q)
    #         self.journal.append({
    #             "t": time.time(),
    #             "event": "init_policy",
    #             "π": dict(self.last_logged_pi),
    #             "entropy": self._current_entropy
    #         })
    #
    #     current_pi = dict(self.policy.Q)
    #     delta_pi = {
    #         k: round(current_pi[k] - self.last_logged_pi.get(k, 0.0), 6)
    #         for k in current_pi
    #     }
    #     self.last_logged_pi = current_pi
    #
    #     delta_q_norm = float(np.linalg.norm(np.array(list(delta_pi.values())))) if delta_pi else 0.0
    #
    #     self.journal.append({
    #         "t": time.time(),
    #         "rule": child.rule,
    #         "Φ": round(child.polarity, 6),
    #         "note": note or child.note,
    #         "Δπ": delta_pi,
    #         "ΔQ_norm": delta_q_norm,
    #         "entropy": self._current_entropy,
    #     })
