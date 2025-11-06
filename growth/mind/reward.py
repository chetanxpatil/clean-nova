# --- New File: growth/mind/reward.py ---
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from growth.mind.growth_mind import GrowthMind
    from growth.node.core_node import GrowthNode


# Define the parameters for reward shaping
@dataclass
class RewardParams:
    correct_reward: float = +1.0
    incorrect_penalty: float = -1.0
    # Bonus for being uncertain/curious (Risk-to-Reward)
    r2r_multiplier: float = 0.1
    r2r_max_bonus: float = 0.5

    # --- CRITICAL FIX: Non-Terminal Penalty (Negative) ---
    non_terminal_penalty: float = -0.1
    depth_penalty: float = -0.01


def compute_total_reward(
        *,
        mind: GrowthMind,
        solution_node: GrowthNode,
        is_correct: bool,
        params: RewardParams,
) -> float:
    """
    Computes the total reward for a path, including shaping.
    This replaces the simple +1/-1 and provides the rich signal needed for learning.
    """

    # 1. Base (Extrinsic) Reward
    total = params.correct_reward if is_correct else params.incorrect_penalty
    final_rule = solution_node.rule.split(':')[-1]

    # --- FIX 1: Curiosity / R2R Bonus ---
    # We use the mind's temperature as a proxy for uncertainty.
    # We must ensure TEMP_MAX (0.50) is used in normalization.
    uncertainty = (mind.temperature - 0.02) / (0.50 - 0.02)
    uncertainty = np.clip(uncertainty, 0, 1)

    curiosity = params.r2r_max_bonus * uncertainty * params.r2r_multiplier
    total += curiosity

    # 3. Apply Penalties

    # A) Penalize if the *final answer* is non-terminal (stabilize/revert/origin/noop)
    # This prevents the AI from defaulting to the root node (origin/noop).
    if final_rule in ("stabilize", "revert", "noop", "origin"):
        # ...UNLESS it was the correct answer!
        if not is_correct:
            total += params.non_terminal_penalty
        # (If it IS correct, it gets the +1.0 base reward, fixing the 0% neutral bug)

    # B) Penalize for depth
    # This is the term that often cancels the curiosity bonus.
    total += solution_node.depth * params.depth_penalty

    return float(total)  # Ensure return type is float