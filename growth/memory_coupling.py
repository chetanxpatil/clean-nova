from __future__ import annotations
import numpy as np
import time
import json
import os
from typing import Dict, Tuple, Optional, List
from collections import defaultdict


class MemoryCoupling:
    def __init__(self, alpha: float = 0.2, max_size: int = 512, history_per_rule: int = 200) -> None:
        self.alpha = alpha
        self.max_size = max_size

        # --- Old (Step 4) Storage for C-Matrices ---
        self.store: Dict[str, np.ndarray] = {}
        self.meta: Dict[str, dict] = {}

        # --- (Step 5.4) Storage for Phi History ---
        # This will store ALL raw Phi values for each rule.
        self.phi_history: Dict[str, List[float]] = defaultdict(list)
        self.max_history_per_rule = history_per_rule

    def key(self, src_shape: Tuple[int, ...], dst_shape: Tuple[int, ...], tag: str = "") -> str:
        """Generate a unique key for a given source/destination shape and tag."""
        return f"{tag}|src={src_shape}|dst={dst_shape}"

    def remember(self, key: str, C: np.ndarray, phi: float) -> None:
        """
        Remembers two things:
        1. (Step 4) The coupling matrix 'C' via an EMA update.
        2. (Step 5.4) The raw 'phi' value, filed under its ground-truth rule.
        """

        # --- 1. (Step 4) Update Coupling Matrix (Original Logic) ---
        C = np.asarray(C, dtype=float)
        phi_eff = np.tanh(phi)
        alpha = self.alpha * (0.5 + 0.5 * abs(phi_eff))

        if key in self.store:
            old = self.store[key]
            C_new = (1 - alpha) * old + alpha * C
        else:
            if len(self.store) >= self.max_size:
                oldest = sorted(self.meta.items(), key=lambda kv: kv[1]["t"])[0][0]
                self.store.pop(oldest, None)
                self.meta.pop(oldest, None)
            C_new = C

        self.store[key] = C_new / (np.linalg.norm(C_new) + 1e-8)
        self.meta[key] = {"t": time.time(), "Φ": float(phi)}

        # --- 2. (Step 5.4) Update Full Phi History ---
        # "Purification" logic is REMOVED. We store ALL data.
        rule = key.split('_')[0]

        if rule in ("merge", "branch", "stabilize"):
            history = self.phi_history[rule]
            history.append(phi)
            # Prune old entries to keep memory fresh
            if len(history) > self.max_history_per_rule:
                self.phi_history[rule] = history[-self.max_history_per_rule:]

    # --- NEW (Step 5.4) Phi Bias Calculation ---

    def get_phi_bias(
            self,
            phi: float,
            window_size: float = 0.1,
            min_samples: int = 10,
            confidence_threshold: float = 0.7
    ) -> float:
        """
        Calculates a "bias" nudge for a given phi value based on
        the ground truth of its historical neighbors.
        """
        upper_bound = phi + window_size / 2
        lower_bound = phi - window_size / 2

        # Find all historical neighbors in this "window"
        merge_count = len([p for p in self.phi_history.get("merge", []) if lower_bound < p < upper_bound])
        branch_count = len([p for p in self.phi_history.get("branch", []) if lower_bound < p < upper_bound])
        stabilize_count = len([p for p in self.phi_history.get("stabilize", []) if lower_bound < p < upper_bound])

        total_count = merge_count + branch_count + stabilize_count

        if total_count < min_samples:
            return 0.0  # Not enough data to be confident

        # Calculate confidence
        merge_confidence = merge_count / total_count
        branch_confidence = branch_count / total_count

        bias = 0.0

        # If we are highly confident this window is "merge"...
        if merge_confidence > confidence_threshold and merge_confidence > branch_confidence:
            # Add a small positive bias to push it over the threshold
            bias = 0.05 * (merge_confidence - 0.5)

        # If we are highly confident this window is "branch"...
        elif branch_confidence > confidence_threshold and branch_confidence > merge_confidence:
            # Add a small negative bias
            bias = -0.05 * (branch_confidence - 0.5)

        return float(np.clip(bias, -0.1, 0.1))  # Safety clip

    # --- DELETED get_adaptive_thresholds() method ---

    # --- Original Methods Below ---

    def recall(
            self,
            src_shape: Tuple[int, ...],
            dst_shape: Tuple[int, ...],
            tag: str = ""
    ) -> Optional[np.ndarray]:
        """Retrieve best matching coupling matrix if available."""
        key = self.key(src_shape, dst_shape, tag)
        if key in self.store:
            return self.store[key]
        for k in self.store.keys():
            if k.startswith(tag):
                return self.store[k]
        return None

    def stats(self) -> Dict[str, object]:
        """Return summary statistics for stored matrices and Phi history."""
        phi_stats: Dict[str, float] = {}
        for rule, history in self.phi_history.items():
            if history:
                phi_stats[f"{rule}_phi_mean"] = float(np.mean(history))
                phi_stats[f"{rule}_phi_var"] = float(np.var(history))
                phi_stats[f"{rule}_n"] = len(history)

        return {
            "matrix_count": len(self.store),
            "phi_history_stats": phi_stats,
            "mean_Φ_all": float(np.mean([m["Φ"] for m in self.meta.values()])) if self.meta else 0.0,
        }

    def save(self, path: str = "brain/memory_coupling.npz") -> None:
        """
        Save coupling matrices, metadata, and Phi history to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **self.store)

        meta_and_history = {
            "meta": self.meta,
            "phi_history": dict(self.phi_history)
        }
        with open(path + ".meta.json", "w") as f:
            json.dump(meta_and_history, f, indent=2)

    def load(self, path: str = "brain/memory_coupling.npz") -> None:
        """
        Load previously saved matrices and metadata from disk.
        """
        if not os.path.exists(path):
            return

        try:
            data = np.load(path)
            for k in data.files:
                self.store[k] = np.array(data[k])
        except Exception as e:
            print(f"Warning: Could not load memory matrices from {path}. Error: {e}")

        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta_and_history = json.load(f)
                self.meta = meta_and_history.get("meta", {})
                loaded_history = meta_and_history.get("phi_history", {})
                self.phi_history = defaultdict(list, loaded_history)
            except Exception as e:
                print(f"Warning: Could not load memory metadata from {meta_path}. Error: {e}")