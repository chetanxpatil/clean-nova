# --- File: growth/reward.py (Accuracy-Only Reward) ---
from dataclasses import dataclass
import numpy as np
import time
from typing import TYPE_CHECKING
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from growth.mind.growth_mind import GrowthMind
    from growth.node.core_node import GrowthNode

# --- Global Initialization and Constants ---
TENSORBOARD_WRITER = SummaryWriter(f'runs/growth_exp_{time.strftime("%Y%m%d-%H%M%S")}')

# Logging State
_REWARD_COUNTER = 0
_TOTAL_REWARD_SUM = 0.0
_PRINT_INTERVAL = 5000  # Keeping speed optimization
_STATS_BUFFER = []


# --- Reward Parameters (Accuracy Only) ---
@dataclass
class RewardParams:
    # ONLY these two parameters remain
    correct_reward: float = +1.0
    incorrect_penalty: float = -1.0
    # All intrinsic (r2r_multiplier, depth_penalty, etc.) parameters removed.


# --- Core Logic Functions ---

def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) > 0 else 0.0


def _log_stats_to_tensorboard(total: float, is_correct: bool, curiosity: float):
    """Updates the global buffer and logs to TensorBoard on the interval."""
    global _REWARD_COUNTER, _STATS_BUFFER, _TOTAL_REWARD_SUM

    # curiosity is always 0.0 in this setup but is included for consistent logging structure
    _STATS_BUFFER.append({"total": total, "is_correct": is_correct, "curiosity": curiosity})

    if _REWARD_COUNTER % _PRINT_INTERVAL == 0 and _REWARD_COUNTER != 0:
        stats = np.array([d["total"] for d in _STATS_BUFFER])
        corrects = sum(d["is_correct"] for d in _STATS_BUFFER)
        curiosity_vals = [d["curiosity"] for d in _STATS_BUFFER]
        buffer_len = len(_STATS_BUFFER)

        _TOTAL_REWARD_SUM += np.sum(stats)
        avg_cumulative_reward = _TOTAL_REWARD_SUM / _REWARD_COUNTER

        # TensorBoard Logging
        TENSORBOARD_WRITER.add_scalar('Performance/Accuracy', corrects / buffer_len, _REWARD_COUNTER)
        TENSORBOARD_WRITER.add_scalar('Reward/Mean_Total_Reward', _safe_mean(stats), _REWARD_COUNTER)
        TENSORBOARD_WRITER.add_scalar('Intrinsic/Avg_Curiosity_Bonus', _safe_mean(curiosity_vals), _REWARD_COUNTER)
        TENSORBOARD_WRITER.add_scalar('Reward/Cumulative_Reward', avg_cumulative_reward, _REWARD_COUNTER)
        TENSORBOARD_WRITER.flush()

        _STATS_BUFFER.clear()


def compute_total_reward(
        *,
        mind: "GrowthMind",
        solution_node: "GrowthNode",
        is_correct: bool,
        params: RewardParams,
) -> float:
    """Calculates the total reward for a path based ONLY on extrinsic correctness."""
    global _REWARD_COUNTER
    _REWARD_COUNTER += 1

    # 1. Base (Extrinsic) Reward: +1.0 for correct, -1.0 for incorrect (Pure Accuracy)
    total = params.correct_reward if is_correct else params.incorrect_penalty

    # 2. Curiosity Bonus (REMOVED)
    curiosity = 0.0

    # 3. Non-Terminal Penalty (REMOVED)

    # 4. Depth Penalty (REMOVED)

    # 5. Logging to TensorBoard
    _log_stats_to_tensorboard(total, is_correct, curiosity)

    return total