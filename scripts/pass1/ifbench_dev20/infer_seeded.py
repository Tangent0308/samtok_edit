"""Use the frozen seed with the shared joint inference implementation."""

import random

import numpy as np
import torch

from samtok_edit21.pass1.evaluation.inference import main

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main()
