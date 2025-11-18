"""
Q-Learning Agent for Modified Tic-Tac-Toe
"""

import numpy as np
import pickle
from typing import Tuple, Dict, List
import random


class QLearningAgent:
    """Q-Learning agent for playing Modified Tic-Tac-Toe"""
    
    def __init__(self, 
                 player_id: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            player_id: Player number (1 or 2)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.player_id = player_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: maps (state, action) -> Q-value
        self.q_table: Dict[Tuple, float] = {}
        
        # For tracking learning progress
        self.episode_rewards = []
    
    def get_state_key(self, state: Tuple) -> Tuple:
        """Convert state to a hashable key for Q-table"""
        return state
    
    def get_action_key(self, action: Tuple[str, int, int]) -> Tuple:
        """Convert action to a hashable key"""
        return action
    
    def get_q_value(self, state: Tuple, action: Tuple[str, int, int]) -> float:
        """Get Q-value for state-action pair"""
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        return self.q_table.get((state_key, action_key), 0.0)
    
    def set_q_value(self, state: Tuple, action: Tuple[str, int, int], value: float):
        """Set Q-value for state-action pair"""
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        self.q_table[(state_key, action_key)] = value
    
    def choose_action(self, state: Tuple, valid_actions: List[Tuple[str, int, int]], 
                     training: bool = True) -> Tuple[str, int, int]:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            valid_actions: List of valid actions
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Exploration: choose random action
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: choose best action based on Q-values
        q_values = [self.get_q_value(state, action) for action in valid_actions]
        max_q = max(q_values)
        
        # If multiple actions have same max Q-value, choose randomly among them
        best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state: Tuple, action: Tuple[str, int, int], 
                      reward: float, next_state: Tuple, 
                      next_valid_actions: List[Tuple[str, int, int]], done: bool):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        """
        current_q = self.get_q_value(state, action)
        
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Get max Q-value for next state
            if next_valid_actions:
                next_q_values = [self.get_q_value(next_state, a) for a in next_valid_actions]
                max_next_q = max(next_q_values)
            else:
                max_next_q = 0.0
            
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.set_q_value(state, action, new_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save agent's Q-table to file"""
        data = {
            'q_table': self.q_table,
            'player_id': self.player_id,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent's Q-table from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.q_table = data['q_table']
            self.player_id = data['player_id']
            self.epsilon = data.get('epsilon', self.epsilon_min)
            self.episode_rewards = data.get('episode_rewards', [])
            print(f"Agent loaded from {filepath}")
            print(f"Q-table size: {len(self.q_table)} entries")
        except FileNotFoundError:
            print(f"No saved agent found at {filepath}. Starting with empty Q-table.")


class RandomAgent:
    """Random agent that selects actions uniformly at random"""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
    
    def choose_action(self, state: Tuple, valid_actions: List[Tuple[str, int, int]], 
                     training: bool = True) -> Tuple[str, int, int]:
        """Choose a random valid action"""
        return random.choice(valid_actions)
    
    def update_q_value(self, *args, **kwargs):
        """Random agent doesn't learn"""
        pass
    
    def decay_epsilon(self):
        """Random agent doesn't have epsilon"""
        pass
