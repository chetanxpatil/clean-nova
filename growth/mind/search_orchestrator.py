"""Search orchestration logic for GrowthMind.

This module now relies on the real GrowthMind and GrowthTree types and
implements the lightweight, linear heuristic that scores every candidate rule
exactly once.  The previous A*/beam search scaffolding and placeholder shims
have been removed in favour of the intended single-pass selector.
"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

import time
import numpy as np

from core.semantic import IntentVector
from growth.node.core_node import GrowthNode

if TYPE_CHECKING:
    from growth.mind.growth_mind import GrowthMind
    from growth.mind.growth_tree import GrowthTree


class SearchOrchestrator:
    """Linear heuristic orchestrator for rule selection."""

    _RULE_BIAS: Dict[str, float] = {
        "merge": 0.15,
        "branch": -0.10,
        "stabilize": 0.05,
        "revert": -0.20,
    }

    _RULE_SIGN: Dict[str, float] = {
        "merge": 1.0,
        "branch": -1.0,
        "stabilize": 0.0,
        "revert": -1.0,
    }

    def __init__(self, mind: "GrowthMind", tree: "GrowthTree"):
        self.mind = mind
        self.tree = tree
        existing_ids = getattr(tree, "nodes", {})
        self._node_counter = (max(existing_ids.keys()) + 1) if existing_ids else 1

    def _next_node_id(self) -> int:
        node_id = self._node_counter
        self._node_counter += 1
        return node_id

    def _score_rule(self, rule: str, phi: float, q_val: float) -> float:
        """Compute the linear heuristic score for a rule."""
        bias = self._RULE_BIAS.get(rule, 0.0)
        signed_phi = self._RULE_SIGN.get(rule, 0.0) * phi
        exploratory = float(np.random.gumbel() * self.mind.temperature)
        return 0.6 * signed_phi + 0.3 * q_val + bias + exploratory

    def _project_polarity(self, rule: str, phi: float) -> float:
        """Project the polarity expected after applying a rule."""
        damp = self.mind.phi_damping
        neutral = self.mind.neutral_band

        if rule == "merge":
            return float(phi * damp)
        if rule == "branch":
            if phi < -neutral:
                return float(phi * damp)
            return float(-abs(phi) * 0.5 * damp)
        if rule == "stabilize":
            return float(phi * 0.25 * damp)
        if rule == "revert":
            return float(-phi * damp)

        return float(phi)

    def run_search(self, initial_intent: IntentVector, note_prefix: str) -> GrowthNode:
        """Evaluate each candidate rule once and return the best-scoring node."""

        root = self.tree.root
        root.search_state = "ROOT"
        self.mind.root = root
        self.mind.active = root

        phi = float(initial_intent.polarity)
        q_values = self.mind.policy.Q
        timestamp = time.time()

        candidates: List[GrowthNode] = []

        for rule in ("merge", "branch", "stabilize", "revert"):
            if rule == "revert" and (root.parent is None or root.parent.parent is None):
                continue

            q_val = float(q_values.get(rule, 0.0))
            score = self._score_rule(rule, phi, q_val)
            projected_phi = self._project_polarity(rule, phi)

            node = GrowthNode(
                state=root.state,
                polarity=projected_phi,
                note=f"{note_prefix}:{rule}",
                node_id=self._next_node_id(),
                parent_id=root.node_id,
                parent=root,
                depth=root.depth + 1,
                rule=self._rule_label(rule, phi),
                log=root.log,
                search_state="EVALUATED",
            )
            node.score = score

            self.tree.add_node(node, root.node_id)
            candidates.append(node)

            self.mind.journal.append(
                {
                    "t": timestamp,
                    "event": "search_candidate",
                    "note": note_prefix,
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "rule": rule,
                    "rule_label": node.rule,
                    "score": float(score),
                    "Φ_in": phi,
                    "Φ_out": float(projected_phi),
                    "π": q_val,
                    "temperature": float(self.mind.temperature),
                }
            )

        if not candidates:
            root.search_state = "NO_CANDIDATES"
            self.mind.journal.append(
                {
                    "t": timestamp,
                    "event": "search_fallback",
                    "note": note_prefix,
                    "reason": "no_rules_available",
                    "node_id": root.node_id,
                }
            )
            return root

        best = max(candidates, key=lambda n: n.score or float("-inf"))
        best.search_state = "SELECTED"
        self.mind.active = best

        self.mind.journal.append(
            {
                "t": timestamp,
                "event": "search_decision",
                "note": note_prefix,
                "selected_node": best.node_id,
                "selected_rule": best.rule,
                "Φ_in": phi,
                "Φ_out": float(best.polarity),
                "score": float(best.score or 0.0),
                "candidates": [
                    {
                        "node_id": node.node_id,
                        "rule": node.rule,
                        "score": float(node.score or 0.0),
                    }
                    for node in candidates
                ],
            }
        )

        return best

    def _rule_label(self, rule: str, phi: float) -> str:
        """Return the canonical label for a rule."""
        if rule == "merge":
            return "G1:merge"
        if rule == "branch":
            if phi < -self.mind.neutral_band:
                return "G2:branch"
            return "G2:noop"
        if rule == "stabilize":
            return "G3:stabilize"
        if rule == "revert":
            return "G4:revert"
        return rule
