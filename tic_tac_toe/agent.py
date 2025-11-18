"""
Q-Learning智能体
Q-Learning Agent for Tic-Tac-Toe
"""
import numpy as np
import pickle
import random
from typing import Tuple, Dict
from .game import TicTacToe


class QLearningAgent:
    """Q学习智能体"""
    
    def __init__(self, player: int = 1, learning_rate: float = 0.1, 
                 discount_factor: float = 0.9, epsilon: float = 0.1):
        """
        初始化Q学习智能体
        
        参数:
            player: 玩家标识 (1 或 -1)
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率 (epsilon-greedy策略)
        """
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table: Dict[str, Dict[Tuple[int, int], float]] = {}
        
    def get_q_value(self, state_key: str, action: Tuple[int, int]) -> float:
        """获取Q值"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
        return self.q_table[state_key][action]
    
    def get_best_action(self, game: TicTacToe) -> Tuple[int, int]:
        """获取当前状态下的最佳动作"""
        state_key = game.get_state_key()
        available_actions = game.get_available_actions()
        
        if not available_actions:
            return None
        
        # 获取所有动作的Q值
        q_values = [(action, self.get_q_value(state_key, action)) 
                    for action in available_actions]
        
        # 选择Q值最大的动作
        best_action = max(q_values, key=lambda x: x[1])[0]
        return best_action
    
    def choose_action(self, game: TicTacToe, training: bool = True) -> Tuple[int, int]:
        """
        选择动作 (epsilon-greedy策略)
        
        参数:
            game: 游戏实例
            training: 是否在训练模式
        """
        available_actions = game.get_available_actions()
        
        if not available_actions:
            return None
        
        # 在训练模式下使用epsilon-greedy策略
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # 否则选择最佳动作
        return self.get_best_action(game)
    
    def update_q_value(self, state_key: str, action: Tuple[int, int], 
                       reward: float, next_state_key: str, 
                       next_available_actions: list):
        """更新Q值"""
        current_q = self.get_q_value(state_key, action)
        
        # 计算下一个状态的最大Q值
        if next_available_actions:
            max_next_q = max([self.get_q_value(next_state_key, next_action) 
                             for next_action in next_available_actions])
        else:
            max_next_q = 0.0
        
        # Q-learning更新规则
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'q_table': self.q_table,
            'player': self.player,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.player = model_data['player']
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.epsilon = model_data['epsilon']
        print(f"模型已从 {filepath} 加载")
    
    def set_epsilon(self, epsilon: float):
        """设置探索率"""
        self.epsilon = epsilon


class RandomAgent:
    """随机智能体（用于对战测试）"""
    
    def __init__(self, player: int = -1):
        self.player = player
    
    def choose_action(self, game: TicTacToe, training: bool = False) -> Tuple[int, int]:
        """随机选择一个可用动作"""
        available_actions = game.get_available_actions()
        if not available_actions:
            return None
        return random.choice(available_actions)
