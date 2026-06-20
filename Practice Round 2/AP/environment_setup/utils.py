import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_header():
    print("="*40)
    print("CEOAI Practice Round 2 - Solution")
    print(f"Running from: {__file__}")
    print("="*40)
