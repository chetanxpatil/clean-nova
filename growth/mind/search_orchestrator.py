from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Dict, Any, Tuple
import time
import numpy as np

# Make sure IntentVector is imported
from core.semantic import IntentVector, compute_intent
from growth.node.core_node import GrowthNode
from growth.rules import apply_rule, GrowthResult

if TYPE_CHECKING:
    from growth.mind.growth_mind import GrowthMind  # For policy/memory
    from growth.mind.growth_tree import GrowthTree  # For node storage


class SearchOrchestrator:
    """
    Implements a HEURISTIC-GUIDED LINEAR search.
    This restores the fast, 60%-accurate "System 1" logic
    inside the new, correct architecture.
    """

    def __init__(self, mind: GrowthMind, tree: GrowthTree):
        self.mind = mind  # The "Mind" holds the Policy, Memory, and Journal
        self.tree = tree  # The "Tree" holds the node data
        self.node_counter = 1

    def _get_next_node_id(self) -> int:
        id = self.node_counter
        self.node_counter += 1
        return id

    def _choose_rule(self, phi: float) -> str:
        """
        This is the "smart" heuristic (our 60% logic).
        We've moved it from GrowthMind to the orchestrator.
        """
        band = float(self.mind.neutral_band)
        if phi > band:
            return "merge"
        if phi < -band:
            return "branch"
        return "stabilize"  # Default to stabilize if in doubt

    def run_search(self, initial_intent: IntentVector, note_prefix: str) -> GrowthNode:
        """
        Runs a fast, single-path search guided by the 'choose_rule' heuristic.
        This simulates the old 'step' logic but correctly builds a tree.
        """

        current_node = self.tree.root
        current_intent = initial_intent

        max_depth = 10  # Safety limit

        for depth in range(max_depth):
            current_node.search_state = "EXPANDED"

            # 1. Calculate Polarity (Phi)
            # Damping is applied to the polarity from the *previous* node
            phi = float(np.clip(current_intent.polarity, -1.0, 1.0)) * self.mind.phi_damping

            # 2. Choose ONE "smart" rule
            rule_name = self._choose_rule(phi)

            # 3. Check for Termination (is the chosen rule a final answer?)
            is_terminal = rule_name in ("merge", "branch")

            # 4. Apply ONLY the chosen rule
            parent_state = current_node.state
            grandparent_state = current_node.parent.state if current_node.parent else parent_state

            args: Tuple = ()
            kwargs: Dict[str, Any] = {"log": current_node.log}

            if rule_name == "merge":
                args = (parent_state, grandparent_state)
                kwargs["intent"] = current_intent
            elif rule_name == "branch":
                args = (parent_state,)
                kwargs["intent"] = current_intent
            elif rule_name == "stabilize":
                args = (parent_state,)

            result, log = apply_rule(rule_name, *args, **kwargs)

            # 5. Create the ONE new child
            child_node = GrowthNode(
                state=result.new_state,
                polarity=result.new_polarity,  # Damping was already applied to intent
                depth=current_node.depth + 1,
                rule=result.rule,
                note=result.note,
                parent=current_node,
                log=log,
                node_id=self._get_next_node_id(),
                parent_id=current_node.node_id,
                search_state="GENERATED",
                h_score=self.mind.policy.Q.get(result.rule.split(':')[-1], 0.0)
            )

            # 6. Add to tree and journal
            self.tree.add_node(child_node, current_node.node_id)
            self.mind.journal.append({
                "t": time.time(),
                "node_id": child_node.node_id,
                "parent_id": child_node.parent_id,
                "depth": child_node.depth,
                "rule": child_node.rule,
                "Φ": round(child_node.polarity, 6),
                "note": f"{note_prefix}_{child_node.rule}",
                "heuristic_score": child_node.g_score,
                "phi_field_flat": child_node.state.weights.flatten().tolist()
            })

            # 7. Navigate to the new node
            current_node = child_node

            # 8. Prepare for next loop
            # --- FIX: Provide all required arguments to IntentVector ---
            current_intent = IntentVector(
                polarity=child_node.polarity,
                raw_polarity=0.0,  # Raw polarity is reset after the first step
                delta_energy=0.0,
                rotation_seq="",
                observer="Om"
            )
            # ---------------------------------------------------------

            if is_terminal:
                break  # We found our answer, stop the search.

        # Return the final node in the single, smart path
        return current_node