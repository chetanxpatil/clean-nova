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
# These are OK, assuming you run your project from the root (the 'clean-nova' folder)
from core.lattice import LatticeState, canonical_symbol_layout
from core.semantic import IntentVector, compute_intent
from core.conservation import verify_conservation

# --- Imports relative to the 'growth/' directory ---
from growth.node.core_node import GrowthNode  # <-- FIXED
from growth.rules import apply_rule           # <-- This was OK

# --- Top-level import ---
from growth.memory_coupling import MemoryCoupling # <-- FIXED

# --- Refactored Module Imports (relative to this 'mind/' dir) ---
from growth.mind.policy import PolicyPi                  # <-- FIXED
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
    neutral_band: float = 0.35

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
        Adaptive rule chooser with ε-greedy exploration.
        """
        band = float(self.neutral_band)
        lo, hi = -band / 2, band / 2

        # dynamic thresholds
        if phi < lo:
            base = "branch"
        elif phi > hi:
            base = "merge"
        else:
            base = "stabilize"

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
        rule = self.choose_rule(phi)

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

        # --- BALANCED ADAPTIVE REWARD SYSTEM ---
        phi_reward = np.tanh(child.polarity)
        if external_correct is not None:
            acc_reward = +1.2 if external_correct else -0.8
        else:
            acc_reward = 0.0

        phi_variance = abs(child.polarity - self.last_phi)
        self.last_phi = child.polarity
        curiosity_bonus = 0.12 * np.tanh(phi_variance * 4.0)
        exploration_bonus = 0.08 if (not external_correct and rule in ["branch", "merge"]) else 0.0
        stability_bonus = 0.08 if (external_correct and rule == "stabilize") else 0.0

        total_reward = (
                0.3 * phi_reward
                + 0.5 * acc_reward
                + stability_bonus
                + curiosity_bonus
                + exploration_bonus
        )
        # --- END REWARD SYSTEM ---

        self.phi_field = child.state.weights
        kernel = None

        # --- MEMORY-AWARE REWARD MODULATION ---
        if hasattr(self, "memory"):
            # Recall the most relevant kernel for this RULE (e.g., "merge" or "branch")
            kernel = self.memory.recall(src_shape=(3, 3, 3), dst_shape=(3, 3, 3), tag=rule)

            if kernel is not None:
                try:
                    # Calculate gradient magnitude of the new state's phi_field
                    phi_grad_vec = np.array(np.gradient(self.phi_field))
                    # Add epsilon to prevent division by zero or noisy gradients
                    phi_grad_mag = np.linalg.norm(phi_grad_vec, axis=0) + 1e-8  # Shape (3,3,3)

                    # Calculate effect (dot product of kernel and gradient)
                    phi_effect = np.sum(kernel * phi_grad_mag)
                    weighted_phi = np.tanh(phi_effect)
                    total_reward += 0.2 * weighted_phi
                except Exception:
                    pass  # Failed modulation, continue
        # --- END MEMORY-AWARE REWARD MODULATION ---

        # --- MEMORY-COUPLING ENTROPY FEEDBACK ---
        if hasattr(self, "memory"):
            try:
                if kernel is None:
                    kernel = self.memory.recall(src_shape=(3, 3, 3), dst_shape=(3, 3, 3), tag=rule)

                if kernel is not None:
                    coupling_strength = np.mean(np.abs(np.array(kernel)))
                else:
                    coupling_strength = 0.0

                # 🔧 NEW FEEDBACK BALANCE
                entropy_adjust = np.clip((0.25 - coupling_strength), -0.08, 0.03)

                self.temperature = np.clip(
                    self.temperature + entropy_adjust * 0.5,  # damped reaction
                    0.02, 0.10  # narrower top
                )
            except Exception:
                pass
        # --- END ENTROPY FEEDBACK ---

        # --- SMOOTH CREDIT ASSIGNMENT ---
        self.reward_history.append(total_reward)
        if len(self.reward_history) > 50:
            self.reward_history.pop(0)
        total_reward = float(np.mean(self.reward_history))
        # --- END CREDIT SMOOTHING ---

        # --- REWARD POLISHING (Clamping & Entropy Scaling) ---
        # Clamp total_reward to avoid explosions from noisy external signals
        total_reward = np.clip(total_reward, -2.0, 2.0)

        # Compute entropy ONCE, using the temperature *before* the feedback loop
        current_entropy = self.policy.entropy(self.temperature)

        # Optional: Scale reward based on current entropy (smoother decay)
        try:
            # entropy_factor is 0 at high entropy, 1 at low
            entropy_factor = max(0.0, 1.0 - (current_entropy / 2.0))
            # Boost reward when policy is certain (low entropy)
            total_reward *= (1.0 + 0.3 * entropy_factor)
        except Exception:
            pass  # Fail safe, use clamped reward
        # --- END REWARD POLISHING ---

        # --- FINAL POLICY UPDATE ---
        self.policy.update(rule, total_reward)

        # --- ENTROPY-COUPLED TEMPERATURE FEEDBACK ---
        # Use the entropy calculated *before* this feedback loop
        self.temperature *= (0.995 if current_entropy > 1.5 else 1.002)
        self.temperature = np.clip(self.temperature, 0.02, 0.12)
        # --- END TEMPERATURE FEEDBACK ---

        # Ensure we have an absolute-π baseline in the journal
        if self.last_logged_pi is None:
            self.last_logged_pi = dict(self.policy.Q)
            self.journal.append({
                "t": time.time(),
                "event": "init_policy",
                "π": dict(self.last_logged_pi),
                "entropy": current_entropy  # Log the entropy from this step
            })

        # Compute Δπ against last-logged π and advance the baseline
        current_pi = dict(self.policy.Q)
        delta_pi = {k: round(current_pi[k] - self.last_logged_pi.get(k, 0.0), 6)
                    for k in current_pi}
        self.last_logged_pi = current_pi

        # Log the magnitude of the policy change (defensively)
        delta_q_norm = float(np.linalg.norm(np.array(list(delta_pi.values())))) if delta_pi else 0.0

        self.journal.append({
            "t": time.time(),
            "rule": child.rule,
            "Φ": round(child.polarity, 6),
            "note": note or child.note,
            "Δπ": delta_pi,
            "ΔQ_norm": delta_q_norm,
            "entropy": current_entropy  # Log the entropy *used* for this step's calcs
        })

        return child