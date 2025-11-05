# # growth/lattice_encoding.py
# """
# Handles text embedding and conversion to LatticeState.
# """
# import numpy as np
# from core.lattice import LatticeState, canonical_symbol_layout
# from core.conservation import verify_conservation
# from growth.config import CONFIG
#
# # --- Globals for caching the embedder and projection matrix ---
# _EMBEDDER = None
# _PROJ = None  # projection matrix for multi-channel mapping
#
# def _get_embedder():
#     """Initializes and returns the SentenceTransformer model."""
#     global _EMBEDDER
#     if _EMBEDDER is None:
#         from sentence_transformers import SentenceTransformer
#         _EMBEDDER = SentenceTransformer(CONFIG["embedding_model"])
#     return _EMBEDDER
#
# def _init_proj(embed_dim: int, out_dim: int):
#     """Initializes a deterministic random projection matrix."""
#     rng = np.random.RandomState(12345)
#     W = rng.randn(embed_dim, out_dim).astype(np.float32) / np.sqrt(embed_dim)
#     return W
#
# def _encode_text(text: str) -> np.ndarray:
#     """Encodes text into a 3D weight field using projection."""
#     vec = _get_embedder().encode(text, normalize_embeddings=True)  # shape [D]
#     D = vec.shape[0]
#     C = CONFIG.get("embed_channels", 1)
#     target_shape = CONFIG["embed_shape"]  # (3,3,3)
#
#     global _PROJ
#     out_dim = int(np.prod(target_shape)) * C  # 27 * C
#     if _PROJ is None or _PROJ.shape != (D, out_dim):
#         _PROJ = _init_proj(D, out_dim)
#
#     # project to multi-channel lattice volume
#     h = vec @ _PROJ  # [27*C]
#     h = h.reshape(C, *target_shape)  # [C,3,3,3]
#
#     # per-channel standardize
#     h = (h - h.mean(axis=(1, 2, 3), keepdims=True)) / (h.std(axis=(1, 2, 3), keepdims=True) + 1e-6)
#
#     # collapse channels -> single 3x3x3 weight field
#     h = h.mean(axis=0)  # [3,3,3]
#     return h.astype(np.float32)
#
# def text_to_lattice(text: str) -> LatticeState:
#     """Converts a string of text into a normalized LatticeState."""
#     base = canonical_symbol_layout()
#     s = base.clone()
#     s.weights += CONFIG["embed_scale"] * _encode_text(text)
#     s.normalize()
#     assert verify_conservation(s), f"ΣSW violated for: {text[:40]}"
#     return s

# growth/lattice_encoding.py
"""
Handles text embedding and conversion to LatticeState.
"""
import numpy as np
from core.lattice import LatticeState, canonical_symbol_layout
from core.conservation import verify_conservation
from growth.config import CONFIG
from typing import Tuple

# --- Globals for caching the embedder and projection matrix ---
_EMBEDDER = None
_PROJ = None  # projection matrix for multi-channel mapping


def _get_embedder():
    """Initializes and returns the SentenceTransformer model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(CONFIG["embedding_model"])
    return _EMBEDDER


def _init_proj(embed_dim: int, out_dim: int):
    """Initializes a deterministic random projection matrix."""
    rng = np.random.RandomState(12345)
    W = rng.randn(embed_dim, out_dim).astype(np.float32) / np.sqrt(embed_dim)
    return W


def _encode_text(text: str) -> np.ndarray:
    """Encodes text into an N x N x N weight field using projection."""
    vec = _get_embedder().encode(text, normalize_embeddings=True)  # shape [D]
    D = vec.shape[0]
    C = CONFIG.get("embed_channels", 1)
    target_shape: Tuple[int, int, int] = CONFIG["embed_shape"]  # (N,N,N)

    global _PROJ
    # out_dim is the size of the final flattened array: N*N*N * C
    out_dim = int(np.prod(target_shape)) * C

    # Re-initialize projection matrix if the required output size has changed (e.g., N=3 -> N=5)
    if _PROJ is None or _PROJ.shape != (D, out_dim):
        _PROJ = _init_proj(D, out_dim)

    # 1. Project to multi-channel lattice volume
    h = vec @ _PROJ  # [N*N*N*C]

    # 2. Reshape to multi-channel cube
    h = h.reshape(C, *target_shape)  # [C, N, N, N]

    # 3. Per-channel standardize
    h = (h - h.mean(axis=(1, 2, 3), keepdims=True)) / (h.std(axis=(1, 2, 3), keepdims=True) + 1e-6)

    # 4. Collapse channels -> single N x N x N weight field
    h = h.mean(axis=0)  # [N, N, N]

    return h.astype(np.float32)


def text_to_lattice(text: str) -> LatticeState:
    """Converts a string of text into a normalized LatticeState."""
    base = canonical_symbol_layout()
    s = base.clone()
    s.weights += CONFIG["embed_scale"] * _encode_text(text)
    s.normalize()
    assert verify_conservation(s), f"ΣSW violated for: {text[:40]}"
    return s