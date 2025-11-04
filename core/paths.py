import os

# -------------------------------------------------------------------
# Root directories
# -------------------------------------------------------------------

# Root of project (assuming core/ is inside nova-livnium)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BRAIN_DIR = os.path.join(ROOT, "brain")

# Ensure main brain directory exists
os.makedirs(BRAIN_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Common file paths
# -------------------------------------------------------------------

PATHS = {
    # TalkCore + MemoryLattice
    "memory": os.path.join(BRAIN_DIR, "memory.jsonl"),
    "policy": os.path.join(BRAIN_DIR, "policy.json"),
    "journal": os.path.join(BRAIN_DIR, "journal.jsonl"),

    # GrowthMind persistence
    "growth_policy": os.path.join(BRAIN_DIR, "growth_policy.json"),
    "growth_journal": os.path.join(BRAIN_DIR, "growth_journal.jsonl"),

    # Model checkpoints and backup states
    "checkpoints": os.path.join(BRAIN_DIR, "checkpoints"),

    # Motif persistence (A8 hierarchical coupling)
    "motifs": os.path.join(BRAIN_DIR, "motifs.json"),
}


# Ensure checkpoint directory exists
os.makedirs(PATHS["checkpoints"], exist_ok=True)

# -------------------------------------------------------------------
# Optional utility
# -------------------------------------------------------------------

def describe_paths():
    """Return a readable summary of all known brain paths."""
    return "\n".join(f"{k:15s} → {v}" for k, v in PATHS.items())


if __name__ == "__main__":
    print("🧠 Unified Brain Path Layout:")
    print(describe_paths())
