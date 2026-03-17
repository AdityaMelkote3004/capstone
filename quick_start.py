"""
Quick Start Script for Stock Return Prediction

This script runs the complete pipeline:
1. Downloads data
2. Trains the model
3. Evaluates performance
4. Saves results

Usage:
    python quick_start.py
"""

import subprocess
import sys
import os

def main():
    print("="*60)
    print("Stock Return Prediction - Quick Start")
    print("="*60)
    
    # Check if requirements are installed
    print("\n1. Checking dependencies...")
    try:
        import torch
        import yfinance
        import yaml
        print("   ✓ All dependencies installed")
    except ImportError as e:
        print(f"   ✗ Missing dependency: {e}")
        print("\n   Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("   ✓ Dependencies installed")
    
    # Run training
    print("\n2. Running training script...")
    print("-"*60)
    
    training_script = os.path.join("src", "training", "train_price_model.py")
    subprocess.check_call([sys.executable, training_script])
    
    print("\n" + "="*60)
    print("Quick start completed successfully!")
    print("="*60)
    print("\nResults saved to: results/price_only/")
    print("  - metrics.txt: Test set performance")
    print("  - loss_curve.png: Training curves")
    print("  - checkpoints/best_model.pt: Best model")
    print("\nTo explore interactively, open: experiments/price_only_baseline.ipynb")

if __name__ == "__main__":
    main()
