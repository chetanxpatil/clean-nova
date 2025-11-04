# core/coupling.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np


# ----------------------------
# Internal utilities
# ----------------------------
def _safe_norm(x: np.ndarray) -> float:
    n = float(np.linalg.norm(x))
    return n if n > 1e-12 else 1e-12


def _ensure_class_first(x: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Heuristic to bring the class axis to the front (C, ...spatial...).
    Returns (class_first_tensor, detected_class_axis).
    """
    if x.ndim < 2:
        raise ValueError("Lattice weights must be at least 2D.")
    # Heuristic: if the last axis is small-ish and smaller than the first, treat it as class axis
    class_axis = 0
    if x.shape[-1] <= 16 and x.shape[-1] < x.shape[0]:
        class_axis = x.ndim - 1
    x_cf = np.moveaxis(x, class_axis, 0)
    return x_cf, class_axis


def _flatten_spatial(x_cf: np.ndarray) -> Tuple[np.ndarray, int]:
    """Flatten spatial dims while preserving class-first layout. Returns (C, S), S = prod(spatial)."""
    C = x_cf.shape[0]
    S = int(np.prod(x_cf.shape[1:]))
    return x_cf.reshape(C, S), S


def _orthonormal_rows_like(na: int, nb: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Return an (na x nb) matrix with orthonormal rows (Stiefel manifold).
    We sample a square Gaussian and project via SVD, then slice and re-project.
    """
    if rng is None:
        rng = np.random.default_rng()
    m = max(na, nb)
    gauss = rng.standard_normal((m, m))
    u, _, vT = np.linalg.svd(gauss, full_matrices=False)
    base = u @ vT  # orthogonal (m x m)
    C0 = base[:na, :nb]
    u2, _, vT2 = np.linalg.svd(C0, full_matrices=False)
    return u2 @ vT2  # (na x nb), rows orthonormal


# ----------------------------
# Coupling map
# ----------------------------
@dataclass
class CouplingMap:
    """
    Cross-lattice coupling between a source lattice (src_shape) and a destination lattice (dst_shape).
    Supports:
      • Discrete per-class permutations (class-preserving)  — A7 compliant
      • Continuous Φ-weighted learning with orthogonalization — reversible & conservative

    Matrix semantics:
      matrix shape = (NA, NB) where
        NA = prod(dst_shape), NB = prod(src_shape)
      apply(src) -> dst via matrix @ src_flat
      apply(dst) -> src via matrix.T @ dst_flat   (since rows are orthonormal)
    """

    # Discrete mapping data (optional):
    per_class_permutation_dst_to_src: Tuple[np.ndarray, ...]  # each array length S_dst with values in [0, S_src)

    # Shapes & class axes:
    src_shape: Tuple[int, ...]
    dst_shape: Tuple[int, ...]
    src_class_axis: int
    dst_class_axis: int

    # Continuous operator:
    matrix: Optional[np.ndarray] = None  # shape (NA, NB)
    mode: str = "identity"

    # ------------
    # Core methods
    # ------------
    def apply(self, tensor: np.ndarray) -> np.ndarray:
        """
        Bidirectional apply:
          • If tensor.size == NB (prod(src_shape)), maps src -> dst (A7 CLO)
          • If tensor.size == NA (prod(dst_shape)), maps dst -> src (transpose path)
          • Else raises.
        Discrete path is used if matrix is None; otherwise continuous path.
        """
        NA = int(np.prod(self.dst_shape))
        NB = int(np.prod(self.src_shape))

        if tensor.size == NB:
            # src -> dst
            if self.matrix is not None:
                flat = tensor.reshape(-1)
                out = self.matrix @ flat  # (NA,)
                return out.reshape(self.dst_shape)
            # Discrete fallback:
            x_cf, _ = _ensure_class_first(tensor)
            x_f, S_src = _flatten_spatial(x_cf)
            C = x_f.shape[0]
            # Build dst buffer
            S_dst = int(np.prod(self.dst_shape)) // C
            out = np.empty((C, S_dst), dtype=x_f.dtype)
            for k, idx_dst in enumerate(self.per_class_permutation_dst_to_src):
                # idx_dst maps each dst position to a src position
                out[k] = x_f[k, idx_dst % S_src]
            out_cf = out.reshape((C,) + tuple(int(s) for s in self.dst_shape if s != self.dst_shape[self.dst_class_axis]))
            return np.moveaxis(out_cf, 0, self.dst_class_axis)

        if tensor.size == NA:
            # dst -> src (inverse via transpose / reverse discrete map)
            if self.matrix is not None:
                flat = tensor.reshape(-1)
                out = self.matrix.T @ flat  # (NB,)
                return out.reshape(self.src_shape)
            # Discrete inverse: compute src from dst by "gather inverse"
            x_cf, _ = _ensure_class_first(tensor)
            x_f, S_dst = _flatten_spatial(x_cf)
            C = x_f.shape[0]
            # Build inverse index per class: for each dst pos j, src pos = idx_dst[j]
            inv_out = []
            S_src = None
            for k, idx_dst in enumerate(self.per_class_permutation_dst_to_src):
                # Infer S_src as max(idx)+1 when not known (safe for contiguous)
                if S_src is None:
                    S_src = max(int(idx_dst.max()) + 1, S_dst)
                src_vec = np.empty(S_src, dtype=x_f.dtype)
                # Many-to-one collisions are unlikely by construction; if happen, last wins.
                src_vec[idx_dst] = x_f[k, np.arange(S_dst)]
                inv_out.append(src_vec)
            out = np.stack(inv_out, axis=0)
            out_cf = out.reshape((C,) + tuple(int(s) for s in self.src_shape if s != self.src_shape[self.src_class_axis]))
            return np.moveaxis(out_cf, 0, self.src_class_axis)

        raise ValueError(
            f"Coupling.apply received tensor with size {tensor.size}, "
            f"which matches neither src ({NB}) nor dst ({NA}) sizes."
        )

    def update(self, A: np.ndarray, B: np.ndarray, phi: float, lr: float = 1e-3) -> float:
        """
        Φ-weighted continuous update, with orthogonalization to preserve reversibility:
            C_{t+1} = orth( C_t - lr * phi * g )
            where g = normalize( (C_t b - sgn(phi) a) ⊗ b ), a=vec(A), b=vec(B)
        Returns a small alignment-style loss for logging.
        """
        # Ensure matrix exists and has correct rectangular shape (NA x NB)
        NA = int(np.prod(self.dst_shape))
        NB = int(np.prod(self.src_shape))
        if self.matrix is None or self.matrix.shape != (NA, NB):
            self.matrix = _orthonormal_rows_like(NA, NB)

        # Targets and residual
        a = A.reshape(-1).astype(float, copy=False)            # (NA,)
        b = B.reshape(-1).astype(float, copy=False)            # (NB,)
        aligned = self.matrix @ b                              # (NA,)
        target = np.sign(phi) * a
        resid = aligned - target                               # (NA,)

        # Outer-product gradient (NA x NB), normalized for scale stability
        g = 2.0 * np.outer(resid, b)
        g /= _safe_norm(g)

        # Φ-weighted step (note the sign: positive phi aligns, negative anti-aligns)
        self.matrix = self.matrix - float(lr) * float(phi) * g

        # Project back to nearest row-orthonormal matrix (Stiefel)
        u, _, vT = np.linalg.svd(self.matrix, full_matrices=False)
        self.matrix = u @ vT  # (NA x NB), rows orthonormal

        # Simple signed alignment loss (for logging only)
        loss = float(np.mean(resid ** 2))
        return loss


# ----------------------------
# Builder
# ----------------------------
def build_coupling(Aw: np.ndarray, Bw: np.ndarray, mode: str = "identity") -> CouplingMap:
    """
    Build a class-preserving coupling from src=B -> dst=A.
    Discrete part provides a permutation-like mapping per class (dst positions -> src positions).
    Continuous part initializes an orthonormal-row matrix C in R^{NA x NB}.
    """
    # Compute class-first layouts and spatial sizes
    A_cf, A_cls_axis = _ensure_class_first(Aw)  # dst
    B_cf, B_cls_axis = _ensure_class_first(Bw)  # src
    A_f, S_A = _flatten_spatial(A_cf)
    B_f, S_B = _flatten_spatial(B_cf)

    if A_f.shape[0] != B_f.shape[0]:
        raise ValueError(f"Class count mismatch: {A_f.shape[0]} vs {B_f.shape[0]}")

    C_classes = A_f.shape[0]
    per_class: List[np.ndarray] = []
    rng = np.random.default_rng()

    # Build per-class index array that maps each dst spatial position j to a src position idx_dst[j]
    for k in range(C_classes):
        if mode == "identity":
            # Map dst positions to the same index modulo src size
            idx_dst = np.arange(S_A) % S_B
        elif mode == "random":
            idx_dst = np.arange(S_A)
            rng.shuffle(idx_dst)
            idx_dst = idx_dst % S_B
        elif mode == "sorted":
            # Heuristic: pair large-magnitude coordinates
            a_rank = np.argsort(-np.abs(A_f[k][:S_A]))
            b_rank = np.argsort(-np.abs(B_f[k][:S_B]))
            idx_dst = np.empty(S_A, dtype=np.int64)
            for i, apos in enumerate(a_rank):
                # place src index (from b_rank) into the dst position 'apos'
                idx_dst[apos] = b_rank[i % S_B]
        else:
            raise ValueError(f"Unknown coupling mode: {mode}")
        per_class.append(idx_dst.astype(np.int64))

    # Initialize continuous operator with orthonormal rows: shape (NA x NB)
    NA = int(np.prod(Aw.shape))
    NB = int(np.prod(Bw.shape))
    C0 = _orthonormal_rows_like(NA, NB, rng=rng)

    return CouplingMap(
        per_class_permutation_dst_to_src=tuple(per_class),
        src_shape=tuple(int(x) for x in Bw.shape),
        dst_shape=tuple(int(x) for x in Aw.shape),
        src_class_axis=B_cls_axis,
        dst_class_axis=A_cls_axis,
        matrix=C0,
        mode=mode,
    )


# ----------------------------
# Convenience wrapper
# ----------------------------
def apply_coupling(Aw: np.ndarray, Bw: np.ndarray, cmap: CouplingMap) -> np.ndarray:
    """
    Apply the coupling from src=B to dst=A (continuous if available, else discrete).
    This is equivalent to cmap.apply(Bw).
    """
    return cmap.apply(Bw)
