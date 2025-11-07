"""Reward computation utilities for the Growth mind."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Tuple, TYPE_CHECKING
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from growth.mind.growth_mind import GrowthMind
    from growth.node.core_node import GrowthNode


__all__ = ["compute_total_reward", "RewardParams", "TENSORBOARD_WRITER"]


TENSORBOARD_WRITER = SummaryWriter(
    f"runs/growth_exp_{time.strftime('%Y%m%d-%H%M%S')}"
)


@dataclass
class RewardParams:
    """Extrinsic reward configuration."""

    correct_reward: float = +1.0
    incorrect_penalty: float = -1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _counter(*, start: int = 0, interval: int = 5000) -> Callable[[], Tuple[int, bool]]:
    """Create a simple step counter that emits (step, should_log)."""

    step = start

    def advance() -> Tuple[int, bool]:
        nonlocal step
        step += 1
        should_log = (step % interval) == 0
        return step, should_log

    return advance


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


_step_counter = _counter(interval=5000)
_total_reward_sum: float = 0.0
_correct_count: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_total_reward(
    *,
    mind: "GrowthMind",
    solution_node: "GrowthNode",
    is_correct: bool,
    params: RewardParams,
) -> float:
    """Compute the total reward for the current step."""

    global _total_reward_sum, _correct_count

    total = params.correct_reward if is_correct else params.incorrect_penalty
    _total_reward_sum += total
    if is_correct:
        _correct_count += 1

    step, should_log = _step_counter()

    # Per-step logging for the new dashboards
    TENSORBOARD_WRITER.add_scalar("Reward/Total", total, step)

    if should_log:
        accuracy = _safe_divide(_correct_count, step)
        avg_reward = _safe_divide(_total_reward_sum, step)

        TENSORBOARD_WRITER.add_scalar("Performance/Accuracy", accuracy, step)
        TENSORBOARD_WRITER.add_scalar("Reward/Mean_Total_Reward", avg_reward, step)
        TENSORBOARD_WRITER.add_scalar("Reward/Cumulative_Reward", _total_reward_sum, step)
        TENSORBOARD_WRITER.flush()

    return total
