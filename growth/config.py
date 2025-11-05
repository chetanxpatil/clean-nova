# growth/config.py
"""
Central configuration for the SNLI experiment.
"""

CONFIG = {
    "seed": 42,
    "use_embeddings": True,
    "embedding_model": "all-MiniLM-L6-v2",
    "embed_channels": 8,  # try 4 or 8
    "embed_scale": 25.0,
    "embed_shape": (3, 3, 3),
    "polarity_scale": 1.0,
    "polarity_shift": 0.0,
    "snli_train_limit": 30000,
    "snli_test_limit": 5000,
    "calib_frac": 0.1,
    "neutral_band_grid": [x / 100 for x in range(5, 61, 5)],
    "progress": True,
}