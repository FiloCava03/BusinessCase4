"""Global determinism utility."""
from __future__ import annotations
import os
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed every RNG we might touch. Idempotent."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  # keep cuDNN flexible; not needed on CPU
    except ImportError:
        pass
