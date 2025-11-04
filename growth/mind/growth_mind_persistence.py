# growth/growth_mind_persistence.py
"""
Mixin class for GrowthMind to handle state persistence (saving/loading).
"""
from __future__ import annotations
import os
import json
from typing import TYPE_CHECKING
from core.paths import PATHS, BRAIN_DIR

if TYPE_CHECKING:
    from growth.mind.growth_mind import GrowthMind

class GrowthMindPersistenceMixin:
    def save_state(self: 'GrowthMind', path: str | None = None):
        """Save GrowthMind policy, journal, and coupling pool to brain/."""
        if path is None:
            path = str(BRAIN_DIR)
        os.makedirs(path, exist_ok=True)

        with open(PATHS["growth_policy"], "w", encoding="utf-8") as f:
            json.dump(self.policy.Q, f, indent=2)

        with open(PATHS["growth_journal"], "w", encoding="utf-8") as f:
            for rec in self.journal:
                f.write(json.dumps(rec) + "\n")

        try:
            self.memory.save(os.path.join(path, "memory_coupling.npz"))
        except Exception as e:
            print(f"⚠️ Failed to save MemoryCoupling: {e}")

        print(f"🌱 GrowthMind state saved to {BRAIN_DIR}/")

    def load_state(self: 'GrowthMind', path: str | None = None):
        """Load GrowthMind policy, journal, and coupling pool from brain/."""
        if path is None:
            path = str(BRAIN_DIR)
        if not os.path.exists(path):
            print("⚙️ No saved GrowthMind brain state found.")
            return

        pol_path = PATHS["growth_policy"]
        jr_path = PATHS["growth_journal"]

        if os.path.exists(pol_path):
            with open(pol_path, "r", encoding="utf-8") as f:
                self.policy.Q = json.load(f)
            print("🔁 Growth policy restored from brain/growth_policy.json")

        if os.path.exists(jr_path):
            with open(jr_path, "r", encoding="utf-8") as f:
                self.journal = [json.loads(line) for line in f if line.strip()]
            print(f"🧩 Growth journal restored ({len(self.journal)} entries)")

        mem_path = os.path.join(path, "memory_coupling.npz")
        if os.path.exists(mem_path):
            try:
                self.memory.load(mem_path)
                print(f"🧠 MemoryCoupling restored ({self.memory.stats()['count']} entries)")
            except Exception as e:
                print(f"⚠️ Failed to load MemoryCoupling: {e}")