"""
Seed Setting for Reproducibility

Purpose:
    - Set random seeds for all libraries
    - Ensure reproducible results
    - Important for scientific experiments
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    This ensures that:
        - Data shuffling is the same
        - Model initialization is the same
        - Dropout is the same
        - Results can be reproduced
    
    Args:
        seed: Random seed value (default: 42)
    """
    print(f"\nSetting random seed to {seed} for reproducibility")
    
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make PyTorch deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("  ✓ Python random seed set")
    print("  ✓ NumPy seed set")
    print("  ✓ PyTorch seed set")
    print("  ✓ CUDA seeds set")
    print("  ✓ Deterministic mode enabled")


if __name__ == "__main__":
    # Demo
    print("Demo: Setting seeds")
    
    set_seed(42)
    
    # Generate some random numbers to verify
    print("\nGenerating random numbers:")
    print(f"Python random: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
    
    print("\nSetting seed again...")
    set_seed(42)
    
    print("\nGenerating same random numbers (should be identical):")
    print(f"Python random: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
