# growth/growth_mind_memory.py
"""
Mixin class for GrowthMind to handle MemoryCoupling adapters.
"""
from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING
import time
import numpy as np

if TYPE_CHECKING:
    from growth.mind.growth_mind import GrowthMind
#
# class GrowthMindMemoryMixin:
#     def note_coupling(self: 'GrowthMind', key: str, C: np.ndarray, phi: float,
#                       src_shape: Tuple[int, ...], dst_shape: Tuple[int, ...]) -> None:
#         """Delegate coupling memory update to MemoryCoupling."""
#         try:
#             self.memory.remember(self.memory.key(src_shape, dst_shape, tag=key), C, phi)
#             self.journal.append({
#                 "t": time.time(),
#                 "event": "memory_update",
#                 "key": key,
#                 "Φ": float(phi),
#                 "pool_count": self.memory.stats()["count"]
#             })
#         except Exception as e:
#             self.journal.append({
#                 "t": time.time(),
#                 "event": "memory_error",
#                 "key": key,
#                 "error": repr(e),
#             })
#
#     def suggest_coupling(self: 'GrowthMind', key: str, src_shape: Tuple[int, ...], dst_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
#         """Retrieve prior coupling from long-term memory."""
#         return self.memory.recall(src_shape, dst_shape, tag=key)

class GrowthMindMemoryMixin:
    def note_coupling(self: 'GrowthMind', key: str, C: np.ndarray, phi: float,
                      src_shape: Tuple[int, ...], dst_shape: Tuple[int, ...]) -> None:
        """Delegate coupling memory update to MemoryCoupling."""
        try:
            self.memory.remember(self.memory.key(src_shape, dst_shape, tag=key), C, phi)

            # --- FIX: Change 'count' to 'matrix_count' ---
            self.journal.append({
                "t": time.time(),
                "event": "memory_update",
                "key": key,
                "Φ": float(phi),
                "pool_count": self.memory.stats()["matrix_count"]  # <-- CORRECTED KEY
            })

        except Exception as e:
            # We still log errors if they happen elsewhere (e.g., during memory.remember)
            self.journal.append({
                "t": time.time(),
                "event": "memory_error",
                "key": key,
                "error": repr(e),
            })

    def suggest_coupling(self: 'GrowthMind', key: str, src_shape: Tuple[int, ...], dst_shape: Tuple[int, ...]) -> \
    Optional[np.ndarray]:
        """Retrieve prior coupling from long-term memory."""
        # NOTE: This function is safe as it only calls memory.recall, which is robust.
        return self.memory.recall(src_shape, dst_shape, tag=key)