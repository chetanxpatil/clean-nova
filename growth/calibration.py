# growth/calibration.py
"""
Handles calibration of the Phi field to find optimal sign and neutral band.
"""
from typing import List, Tuple, Dict
from growth.config import CONFIG
from growth.phi_computer import phi_raw_for_pair

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


def map_label(phi_cal: float, neutral_band: float) -> str:
    """Maps a calibrated Phi value to a discrete label."""
    if phi_cal > neutral_band:
        return "entailment"
    elif phi_cal < -neutral_band:
        return "contradiction"
    return "neutral"


def calibrate_phi(train_pairs: List[Tuple[str, str, str]]) -> Dict[str, float]:
    """
    Calibrates Phi sign and neutral band on a subset of data.
    """
    n_calib = max(100, int(len(train_pairs) * CONFIG["calib_frac"]))
    calib = train_pairs[:n_calib]

    print(f"Running calibration on {len(calib)} samples...")
    raw_list = [(phi_raw_for_pair(s1, s2), g) for s1, s2, g in tqdm(calib, desc="Calibrating Φ")]

    best = {"phi_sign": +1.0, "neutral_band": 0.35, "acc": -1.0}

    for sign in (+1.0, -1.0):
        for band in CONFIG["neutral_band_grid"]:
            preds, correct = [], 0
            for raw_phi, gold in raw_list:
                phi_cal = sign * raw_phi
                pred = map_label(phi_cal, band)
                preds.append(pred)
                correct += (pred == gold)

            acc = correct / len(raw_list)
            dist = {p: preds.count(p) / len(preds) for p in ("entailment", "neutral", "contradiction")}

            # Penalize solutions that collapse to a single label
            collapse_penalty = max(dist.values())
            effective = acc * (1 - collapse_penalty + 0.33)  # +0.33 is baseline random guess

            if effective > best["acc"]:
                best = {"phi_sign": sign, "neutral_band": band, "acc": effective}

    print(f"✅ Calibration: sign={best['phi_sign']:+.0f}, band={best['neutral_band']:.2f}, score={best['acc']:.3f}")
    return best