# memory_coupling.py
from __future__ import annotations
import numpy as np, time, json, os
from typing import Dict, Tuple, Optional
from core.conservation import verify_conservation

class MemoryCoupling:
    def __init__(self, alpha: float = 0.2, max_size: int = 512):
        self.alpha = alpha          # learning rate for updating couplings
        self.max_size = max_size    # maximum stored entries
        self.store: Dict[str, np.ndarray] = {}
        self.meta: Dict[str, dict] = {}

    def key(self, src_shape: Tuple[int,...], dst_shape: Tuple[int,...], tag: str = "") -> str:
        return f"{tag}|src={src_shape}|dst={dst_shape}"

    def remember(self, key: str, C: np.ndarray, phi: float):
        """EMA update of a coupling matrix, weighted by Φ intensity."""
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

    def recall(self, src_shape: Tuple[int,...], dst_shape: Tuple[int,...], tag: str = "") -> Optional[np.ndarray]:
        """Retrieve best matching coupling (or None)."""
        key = self.key(src_shape, dst_shape, tag)
        if key in self.store:
            return self.store[key]
        # fallback: nearest shape match
        for k in self.store.keys():
            if k.startswith(tag):
                return self.store[k]
        return None

    def stats(self) -> dict:
        return {
            "count": len(self.store),
            "mean_Φ": np.mean([m["Φ"] for m in self.meta.values()]) if self.meta else 0.0,
            "time_range": (
                min(m["t"] for m in self.meta.values()) if self.meta else 0,
                max(m["t"] for m in self.meta.values()) if self.meta else 0,
            ),
        }

    def save(self, path="brain/memory_coupling.npz"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **self.store)
        with open(path + ".meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)

    def load(self, path="brain/memory_coupling.npz"):
        if not os.path.exists(path):
            return
        data = np.load(path)
        with open(path + ".meta.json") as f:
            meta = json.load(f)
        for k in data.files:
            self.store[k] = np.array(data[k])
            self.meta[k] = meta.get(k, {"t": time.time(), "Φ": 0.0})
