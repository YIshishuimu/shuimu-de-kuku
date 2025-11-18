"""
Modified Tic-Tac-Toe with Reinforcement Learning

A special variant of Tic-Tac-Toe where:
- Win condition: Complete a COLUMN (not row)
- Each player has 3-4 pieces
- Players can remove opponent's pieces (max 2 times)
- Removed pieces return to player's reserve

Trained using Q-Learning algorithm.
"""

from .game_env import ModifiedTicTacToe
from .agent import QLearningAgent, RandomAgent

__version__ = "1.0.0"
__all__ = ['ModifiedTicTacToe', 'QLearningAgent', 'RandomAgent']
