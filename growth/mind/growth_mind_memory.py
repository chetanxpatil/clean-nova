# --- File: growth/growth_mind_memory.py ---
"""
Mixin class for GrowthMind to handle MemoryCoupling adapters.
Now includes TensorBoard logging for long-term memory evolution.
Throttled console logs every 1000 updates.
"""
from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from growth.mind.growth_mind import GrowthMind

# -------------------------------------------------------------------
# TensorBoard Writer Initialization
# -------------------------------------------------------------------
# Creates a dedicated run directory for memory tracking
MEMORY_TENSORBOARD_WRITER = SummaryWriter(
    f"runs/memory_coupling_{time.strftime('%Y%m%d-%H%M%S')}"
)

# -------------------------------------------------------------------
# Global Counters for Throttled Logging
# -------------------------------------------------------------------
_MEMORY_UPDATE_COUNTER = 0
_PRINT_INTERVAL = 1000  # console update frequency


def _log_every_1000(msg: str):
    """Prints a memory status line every 1000 successful updates."""
    global _MEMORY_UPDATE_COUNTER
    if _MEMORY_UPDATE_COUNTER % _PRINT_INTERVAL == 0 and _MEMORY_UPDATE_COUNTER != 0:
        t = time.strftime("%H:%M:%S")
        print(f"[{t}] {msg}")


# -------------------------------------------------------------------
# Memory Adapter Mixin
# -------------------------------------------------------------------
class GrowthMindMemoryMixin:
    def note_coupling(
        self: "GrowthMind",
        key: str,
        C: np.ndarray,
        phi: float,
        src_shape: Tuple[int, ...],
        dst_shape: Tuple[int, ...],
    ) -> None:
        """Store coupling matrix and log both locally and to TensorBoard."""
        global _MEMORY_UPDATE_COUNTER

        try:
            self.memory.remember(self.memory.key(src_shape, dst_shape, tag=key), C, phi)
            stats = self.memory.stats()
            pool_count = stats.get("matrix_count", stats.get("count", 0))

            # Journal entry for traceability
            self.journal.append({
                "t": time.time(),
                "event": "memory_update",
                "key": key,
                "Φ": float(phi),
                "pool_count": pool_count,
            })

            # TensorBoard logging
            MEMORY_TENSORBOARD_WRITER.add_scalar(
                "Memory/Pool_Size", pool_count, _MEMORY_UPDATE_COUNTER
            )
            MEMORY_TENSORBOARD_WRITER.add_scalar(
                "Memory/Latest_Φ", phi, _MEMORY_UPDATE_COUNTER
            )

            # Optional: track magnitude/variance of coupling matrix
            if C is not None and np.size(C) > 0:
                MEMORY_TENSORBOARD_WRITER.add_scalar(
                    "Memory/Coupling_Mean", float(np.mean(C)), _MEMORY_UPDATE_COUNTER
                )
                MEMORY_TENSORBOARD_WRITER.add_scalar(
                    "Memory/Coupling_Std", float(np.std(C)), _MEMORY_UPDATE_COUNTER
                )

            MEMORY_TENSORBOARD_WRITER.flush()

            # Increment counter and throttled console print
            _MEMORY_UPDATE_COUNTER += 1
            _log_every_1000(
                f"🧠 {_MEMORY_UPDATE_COUNTER:,} memory couplings stored | "
                f"key='{key}', Φ={phi:+.3f}, pool_size={pool_count}"
            )

        except Exception as e:
            self.journal.append({
                "t": time.time(),
                "event": "memory_error",
                "key": key,
                "error": repr(e),
            })

    def suggest_coupling(
        self: "GrowthMind",
        key: str,
        src_shape: Tuple[int, ...],
        dst_shape: Tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """Retrieve prior coupling from long-term memory."""
        return self.memory.recall(src_shape, dst_shape, tag=key)
