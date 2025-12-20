"""
Dueling Double Deep Q-Network (D3QN) implementation for Checkers.

Components:
- model.py: Neural network architecture with dueling streams
- trainer.py: Training loop and optimization
"""

from .model import D3QNModel, DuelingDQN, count_parameters, init_weights
from .trainer import D3QNTrainer

__all__ = [
    "D3QNModel", 
    "DuelingDQN", 
    "count_parameters", 
    "init_weights",
    "D3QNTrainer"
]