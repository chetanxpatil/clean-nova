# growth/phi_computer.py
"""
Computes the normalized Phi (Φ) value for a premise/hypothesis pair.
"""
import numpy as np
from core.semantic import compute_intent
from growth.lattice_encoding import text_to_lattice
from growth.config import CONFIG

# --- Module-level state for running normalization ---
_phi_values_seen = []


def phi_raw_for_pair(premise: str, hypothesis: str) -> float:
    """
    Computes the raw intent polarity and applies a running normalization
    (tanh standardization) over the last 500 values.
    """
    A = text_to_lattice(premise)
    B = text_to_lattice(hypothesis)
    intent = compute_intent(A, B,
                            polarity_scale=CONFIG["polarity_scale"],
                            polarity_shift=CONFIG["polarity_shift"],
                            )
    phi_raw = float(intent.polarity)

    # Update running list for normalization
    _phi_values_seen.append(phi_raw)
    if len(_phi_values_seen) > 500:
        del _phi_values_seen[0]

    # Apply running standardization + tanh squashing
    phi_mean = np.mean(_phi_values_seen) if len(_phi_values_seen) > 10 else 0.0
    phi_std = np.std(_phi_values_seen) if len(_phi_values_seen) > 10 else 1.0
    phi_norm = np.tanh((phi_raw - phi_mean) / (phi_std + 1e-6))
    return float(phi_norm)