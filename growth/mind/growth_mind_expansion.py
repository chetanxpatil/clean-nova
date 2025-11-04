# growth/growth_mind_expansion.py
"""
Mixin class for GrowthMind to handle motif-aware expansion.
"""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np
import time
from core.semantic import compute_intent

if TYPE_CHECKING:
    from growth.node.core_node import GrowthNode
    from growth.mind.growth_mind import GrowthMind
    from core.lattice import LatticeState # Assuming this is the type for 'lattice'

class GrowthMindExpansionMixin:
    def expand_from_lattice(self: 'GrowthMind', lattice: 'LatticeState') -> Optional['GrowthNode']:
        if not lattice.entries:
            return None

        recent = lattice.entries[-min(5, len(lattice.entries)):]
        phis = [e.phi for e in recent]
        variance = float(np.var(phis))
        avg_phi = float(np.mean(phis))

        # Base rule by statistics
        if variance > 0.05:
            rule_hint = "branch"
        elif avg_phi > 0.1:
            rule_hint = "merge"
        else:
            rule_hint = "stabilize"

        # --- Motif bias: use motifs near avg Φ to nudge intent polarity
        motif_bias = 0.0
        motif_meta = None
        try:
            motifs = getattr(lattice, "motifs", None)
            if motifs is not None:
                found = motifs.find(avg_phi, top_k=3)
                if found:
                    motif_bias = float(np.mean([m["phi"] for m in found]))
                    motif_meta = {"count": len(found), "Φ̄_motifs": motif_bias}
        except Exception:
            pass

        # Build a blended state and compute intent
        blended_state = lattice.blend()
        intent = compute_intent(self.active.state, blended_state)

        # Softly push Φ toward motif polarity (tanh-compressed)
        if motif_bias != 0.0:
            phi_push = float(self.motif_eta * np.tanh(motif_bias))
            intent.polarity = float(np.clip(intent.polarity + phi_push, -1.0, 1.0))

        # Journal the expansion decision with motif info
        self.journal.append({
            "t": time.time(),
            "event": "expand_from_lattice",
            "rule_hint": rule_hint,
            "Φ̄_recent": avg_phi,
            "σ²_recent": variance,
            "motif_bias": motif_bias,
            "motif_eta": self.motif_eta,
            "Φ_after_bias": float(intent.polarity),
            "π": dict(self.policy.Q),
            **(motif_meta or {}),
        })

        # Let step() re-evaluate rule with the biased Φ
        new_node = self.step(intent, note=f"auto:{rule_hint}", external_correct=None)
        return new_node