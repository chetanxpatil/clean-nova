from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Dict, Any, Tuple
from growth.mind.policy import PolicyPi
import time
import numpy as np
import heapq


# -------------------------------------------------------------------
# Placeholder Classes (for standalone execution)
# -------------------------------------------------------------------
class IntentVector:
    pass


def compute_intent(state1: Any, state2: Any, observer: str) -> IntentVector:
    return IntentVector()


class GrowthNode:
    def __init__(self, state, polarity, depth, rule, note, parent, log, node_id, parent_id, search_state):
        self.state = state
        self.polarity = polarity
        self.depth = depth
        self.rule = rule
        self.note = note
        self.parent = parent
        self.log = log
        self.node_id = node_id
        self.parent_id = parent_id
        self.search_state = search_state
        self.score = 0.0
        self.children: List['GrowthNode'] = []
        self.g_score: float = 0.0
        self.h_score: float = 0.0


class GrowthResult:
    def __init__(self, new_state, new_polarity, rule, note):
        self.new_state = new_state
        self.new_polarity = new_polarity
        self.rule = rule
        self.note = note


def apply_rule(rule_name, *args, **kwargs) -> Tuple[Optional[GrowthResult], Dict]:
    return GrowthResult(new_state=args[0], new_polarity=0.5, rule=rule_name, note="applied"), {}


# -------------------------------------------------------------------
# TYPE CHECKING IMPORTS
# -------------------------------------------------------------------
if TYPE_CHECKING:
    class GrowthMind:
        phi_damping: float
        temperature: float
        journal: List[Dict]
        policy: PolicyPi

    class GrowthTree:
        root: GrowthNode
        def add_node(self, child: GrowthNode, parent_id: int): ...

    from growth.mind.growth_mind import GrowthMind
    from growth.mind.growth_tree import GrowthTree


# -------------------------------------------------------------------
# GLOBAL COUNTERS FOR THROTTLED LOGGING
# -------------------------------------------------------------------
_CHILD_COUNTER = 0
_PRINT_INTERVAL = 100_000  # reduced output noise


def _log_every_n(msg: str):
    """Prints once every N child nodes for monitoring long runs."""
    global _CHILD_COUNTER
    if _CHILD_COUNTER % _PRINT_INTERVAL == 0 and _CHILD_COUNTER != 0:
        t = time.strftime("%H:%M:%S")
        print(f"[{t}] {msg}")


# -------------------------------------------------------------------
# A* CONFIG
# -------------------------------------------------------------------
MAX_DEPTH = 10
BEAM_WIDTH = 5
REVERT_EXTRA_COST = 1.0
NOISE_SCALE = 1.0
A_STAR_PRINT_INTERVAL = 50_000  # larger to quiet normal runs


# -------------------------------------------------------------------
# Search Orchestrator (A* with Beam)
# -------------------------------------------------------------------
class SearchOrchestrator:
    """
    A* search with per-level beam pruning.
    f(n) = g(n) + h(n)
      • g(n): path cost (depth-based with higher cost for 'revert')
      • h(n): -Q(rule) with small Gumbel noise (scaled by temperature)
    """

    def __init__(self, mind: 'GrowthMind', tree: 'GrowthTree'):
        self.mind = mind
        self.tree = tree
        self.node_counter = 1

    def _get_next_node_id(self) -> int:
        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    # ---------------------------------------------------------------
    # Child Generation
    # ---------------------------------------------------------------
    def _generate_children(self, parent_node: GrowthNode, intent: IntentVector) -> List[GrowthNode]:
        """Generates all possible child nodes for a given parent."""
        global _CHILD_COUNTER
        children: List[GrowthNode] = []
        parent_state = parent_node.state
        grandparent_state = parent_node.parent.state if parent_node.parent else parent_state

        possible_rules = ["merge", "branch", "stabilize", "revert"]

        for rule_name in possible_rules:
            args: Tuple = ()
            kwargs: Dict[str, Any] = {"log": parent_node.log}

            if rule_name == "merge":
                args = (parent_state, grandparent_state)
                kwargs["intent"] = intent
            elif rule_name == "branch":
                args = (parent_state,)
                kwargs["intent"] = intent
            elif rule_name == "stabilize":
                args = (parent_state,)
            elif rule_name == "revert":
                if parent_node.parent is None or parent_node.parent.parent is None:
                    continue
                args = (parent_node.parent.parent.state,)
            else:
                continue

            try:
                result, log = apply_rule(rule_name, *args, **kwargs)
            except Exception:
                continue

            if result is None or result.new_state is None:
                continue

            child_node = GrowthNode(
                state=result.new_state,
                polarity=result.new_polarity * self.mind.phi_damping,
                depth=parent_node.depth + 1,
                rule=result.rule,
                note=result.note,
                parent=parent_node,
                log=log,
                node_id=self._get_next_node_id(),
                parent_id=parent_node.node_id,
                search_state="GENERATED"
            )
            children.append(child_node)
            _CHILD_COUNTER += 1

            # Throttled progress indicator (kept for transparency)
            _log_every_n(
                f"🌿 {_CHILD_COUNTER:,} child nodes created | "
                f"id={child_node.node_id}, rule='{child_node.rule}', Φ={child_node.polarity:+.3f}, depth={child_node.depth}"
            )

        return children

    # ---------------------------------------------------------------
    # A* + Beam Execution
    # ---------------------------------------------------------------
    def run_search(self, initial_intent: IntentVector, note_prefix: str) -> Optional[GrowthNode]:
        """
        A* search with beam pruning.
        Keeps top-K candidates per level based on f(n) = g(n) + h(n).
        """
        global _CHILD_COUNTER
        root = self.tree.root
        root.g_score = 0.0
        root.h_score = -self.mind.policy.Q.get("origin", 0.0)
        root.search_state = "ROOT"

        frontier_pq: List[Tuple[float, int, GrowthNode]] = []
        heapq.heappush(frontier_pq, (root.g_score + root.h_score, root.node_id, root))

        best_solution_node: Optional[GrowthNode] = root
        best_h = root.h_score
        visited_nodes = 0

        for depth in range(MAX_DEPTH):
            if not frontier_pq:
                break

            next_level_candidates: List[Tuple[float, int, GrowthNode]] = []

            while frontier_pq:
                _, _, current_node = heapq.heappop(frontier_pq)
                current_node.search_state = "EXPANDED"
                visited_nodes += 1

                if current_node.h_score < best_h:
                    best_h = current_node.h_score
                    best_solution_node = current_node

                rule_tail = current_node.rule.split(':')[-1]
                is_terminal = rule_tail in ("merge", "branch")
                if is_terminal or current_node.depth >= MAX_DEPTH:
                    continue

                current_intent = (
                    initial_intent
                    if current_node.depth == 0
                    else compute_intent(
                        current_node.state,
                        current_node.parent.state if current_node.parent else current_node.state,
                        observer="Lo"
                    )
                )

                child_nodes = self._generate_children(current_node, current_intent)

                for child in child_nodes:
                    rule_key = child.rule.split(':')[-1]

                    g_score = current_node.g_score + 1
                    h_score = -self.mind.policy.Q.get(rule_key, 0.0)
                    h_score -= np.random.gumbel() * self.mind.temperature
                    f_score = g_score + h_score

                    # Minimal A* diagnostics only at long intervals
                    if _CHILD_COUNTER % A_STAR_PRINT_INTERVAL == 0 and _CHILD_COUNTER != 0:
                        t = time.strftime("%H:%M:%S")
                        print(f"[{t}] A* progress → {_CHILD_COUNTER:,} nodes processed | "
                              f"latest ID={child.node_id}, rule={rule_key}, f={f_score:+.2f}")

                    child.g_score = g_score
                    child.h_score = h_score
                    self.tree.add_node(child, current_node.node_id)

                    heapq.heappush(next_level_candidates, (f_score, child.node_id, child))

                    phi_field_flat = (
                        child.state.weights.flatten().tolist()
                        if hasattr(child.state, 'weights')
                        else []
                    )
                    self.mind.journal.append({
                        "t": time.time(),
                        "node_id": child.node_id,
                        "parent_id": child.parent_id,
                        "depth": child.depth,
                        "rule": child.rule,
                        "Φ": round(child.polarity, 6),
                        "note": f"{note_prefix}_{child.rule}",
                        "g_score": round(g_score, 6),
                        "h_score": round(h_score, 6),
                        "f_score": round(f_score, 6),
                        "phi_field_flat": phi_field_flat
                    })

            if not next_level_candidates:
                break

            best_k = heapq.nsmallest(BEAM_WIDTH, next_level_candidates)
            frontier_pq = best_k

        return best_solution_node
