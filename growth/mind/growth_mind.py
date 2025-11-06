"""
Livnium Growth — growth_mind.py
===============================
Refactored to be a "controller" that holds Policy, Memory, and Journal.
The search logic has been externalized to SearchOrchestrator.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np

# --- Core Livnium Imports ---
from core.lattice import LatticeState, canonical_symbol_layout

# --- Imports relative to the 'growth/' directory ---
from growth.node.core_node import GrowthNode

# --- Top-level import ---
from growth.memory_coupling import MemoryCoupling

# --- Refactored Module Imports (relative to this 'mind/' dir) ---
from growth.mind.policy import PolicyPi
from growth.mind.growth_mind_memory import GrowthMindMemoryMixin
from growth.mind.growth_mind_expansion import GrowthMindExpansionMixin
from growth.mind.growth_mind_introspection import GrowthMindIntrospectionMixin
from growth.mind.growth_mind_persistence import GrowthMindPersistenceMixin


# -------------------------------------------------------------------
# Tunables & constants (RESTORED)
# -------------------------------------------------------------------

_TEMP_MIN = 0.02
_TEMP_MAX = 0.50 # Using the aggressive exploration max (was 0.12)

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
    Core controller for Policy, Memory, and Journaling.
    The search logic is now handled by SearchOrchestrator.
    """

    # --- Core AI Components ---
    policy: PolicyPi = field(default_factory=PolicyPi)
    journal: List[dict] = field(default_factory=list)
    memory: MemoryCoupling = field(
        default_factory=lambda: MemoryCoupling(alpha=0.2, max_size=512)
    )

    # --- Parameters used by external components (like SearchOrchestrator) ---
    temperature: float = 0.07
    phi_damping: float = 0.95
    neutral_band: float = 0.45

    # --- Other attributes ---
    last_logged_pi: Optional[Dict[str, float]] = None
    motif_eta: float = 0.15

    # -------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "GrowthMind":
        """Instantiate GrowthMind from configuration dictionary."""
        mind = cls() # Create an empty mind
        mind.temperature = cfg.get("temperature", 0.04)
        mind.phi_damping = cfg.get("phi_damping", 0.95)
        mind.motif_eta = float(cfg.get("motif_eta", 0.15))
        mind.neutral_band = cfg.get("neutral_band", 0.45)
        return mind

    # -------------------------------------------------------------------
    # META-COGNITION (Still relevant)
    # -------------------------------------------------------------------

    def metacog_reflect_and_tune(self, cm: np.ndarray, train_acc: float) -> None:
        """
        Tunes the mind's internal NEUTRAL_BAND for the *next* run.
        """
        print("\n" + "=" * 20 + " META-COGNITION REFLECTION " + "=" * 20)

        neutral_total = cm[1, :].sum()
        neutral_correct = cm[1, 1]

        if neutral_total == 0:
            print("🧠 [MetaCog] No neutral samples seen. No tuning applied.")
            return

        neutral_acc = neutral_correct / max(1, neutral_total)
        print(f"🧠 [MetaCog] Last run total accuracy: {train_acc:.2%}")
        print(f"🧠 [MetaCog] Last run 'neutral' accuracy: {neutral_acc:.2%}")
        print(f"🧠 [MetaCog] Current neutral_band: {self.neutral_band:.3f}")

        # --- FINE-GRAINED TUNING LOGIC (Unchanged) ---
        TUNING_STEP = 0.01
        is_lagging = neutral_acc < (train_acc - 0.10)
        is_too_cautious = neutral_acc > (train_acc + 0.05)

        if is_lagging:
            self.neutral_band = min(self.neutral_band + TUNING_STEP, 0.60)
            print(
                f"🧠 [MetaCog] RESULT: Neutral lagging → widen to {self.neutral_band:.3f}"
            )
        elif is_too_cautious:
            self.neutral_band = max(self.neutral_band - TUNING_STEP, 0.20)
            print(
                f"🧠 [MetaCog] RESULT: Too cautious → narrow to {self.neutral_band:.3f}"
            )
        else:
            print("🧠 [MetaCog] RESULT: Balanced. No tuning required.")

        print("=" * 66)

    # -------------------------------------------------------------------
    # Metabolism
    # -------------------------------------------------------------------

    def update_temperature(self) -> None:
        """
        Entropy-coupled temperature feedback.
        This is called by the orchestrator to keep exploration dynamic.
        """
        try:
            current_entropy = self.policy.entropy(self.temperature)
            # Adjust temperature based on entropy level
            self.temperature *= 0.995 if current_entropy > 1.5 else 1.002
            # Clamp temperature to defined limits
            self.temperature = float(np.clip(self.temperature, _TEMP_MIN, _TEMP_MAX))
        except Exception:
            # stay fail-quiet
            pass