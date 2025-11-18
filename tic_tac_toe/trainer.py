"""
井字棋训练器
Tic-Tac-Toe Trainer
"""
import numpy as np
from typing import Optional
from .game import TicTacToe
from .agent import QLearningAgent, RandomAgent


class TicTacToeTrainer:
    """井字棋训练器"""
    
    def __init__(self):
        """初始化训练器"""
        self.game = TicTacToe()
        self.agent1 = None
        self.agent2 = None
        self.stats = {
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'total_games': 0
        }
    
    def play_game(self, agent1, agent2, training: bool = True, 
                  verbose: bool = False) -> int:
        """
        进行一局游戏
        
        返回: 获胜者 (1, -1, 或 0表示平局)
        """
        self.game.reset()
        history = []  # 记录历史状态和动作
        
        while True:
            # 当前玩家
            current_agent = agent1 if self.game.current_player == 1 else agent2
            
            # 记录当前状态
            state_key = self.game.get_state_key()
            
            # 选择动作
            action = current_agent.choose_action(self.game, training=training)
            
            if action is None:
                break
            
            # 记录历史
            if training and isinstance(current_agent, QLearningAgent):
                history.append({
                    'agent': current_agent,
                    'state_key': state_key,
                    'action': action
                })
            
            # 执行动作
            success, winner, done = self.game.make_move(action)
            
            if verbose:
                self.game.render()
            
            if done:
                # 游戏结束，更新Q值
                if training:
                    for record in history:
                        agent = record['agent']
                        if isinstance(agent, QLearningAgent):
                            reward = self.game.get_reward(winner, agent.player)
                            agent.update_q_value(
                                record['state_key'],
                                record['action'],
                                reward,
                                self.game.get_state_key(),
                                []
                            )
                
                return winner if winner is not None else 0
            
            # 更新之前步骤的Q值
            if training and len(history) >= 2:
                prev_record = history[-2]
                if isinstance(prev_record['agent'], QLearningAgent):
                    prev_record['agent'].update_q_value(
                        prev_record['state_key'],
                        prev_record['action'],
                        0.0,  # 中间步骤奖励为0
                        state_key,
                        self.game.get_available_actions()
                    )
        
        return 0  # 平局
    
    def train(self, agent1: QLearningAgent, agent2, 
              num_episodes: int = 10000, 
              print_interval: int = 1000):
        """
        训练智能体
        
        参数:
            agent1: Q学习智能体
            agent2: 对手（可以是另一个Q学习智能体或随机智能体）
            num_episodes: 训练轮数
            print_interval: 打印间隔
        """
        print(f"开始训练，共 {num_episodes} 轮...")
        
        for episode in range(num_episodes):
            # 交替先手
            if episode % 2 == 0:
                winner = self.play_game(agent1, agent2, training=True)
                if winner == 1:
                    self.stats['player1_wins'] += 1
                elif winner == -1:
                    self.stats['player2_wins'] += 1
                else:
                    self.stats['draws'] += 1
            else:
                winner = self.play_game(agent2, agent1, training=True)
                if winner == 1:
                    self.stats['player2_wins'] += 1
                elif winner == -1:
                    self.stats['player1_wins'] += 1
                else:
                    self.stats['draws'] += 1
            
            self.stats['total_games'] += 1
            
            # 打印进度
            if (episode + 1) % print_interval == 0:
                self.print_stats(episode + 1)
        
        print("\n训练完成！")
        self.print_stats(num_episodes)
    
    def print_stats(self, episode: int):
        """打印统计信息"""
        total = self.stats['total_games']
        if total == 0:
            return
        
        p1_win_rate = self.stats['player1_wins'] / total * 100
        p2_win_rate = self.stats['player2_wins'] / total * 100
        draw_rate = self.stats['draws'] / total * 100
        
        print(f"\n第 {episode} 轮:")
        print(f"  玩家1胜率: {p1_win_rate:.2f}%")
        print(f"  玩家2胜率: {p2_win_rate:.2f}%")
        print(f"  平局率: {draw_rate:.2f}%")
    
    def evaluate(self, agent1, agent2, num_games: int = 100, verbose: bool = False):
        """
        评估智能体性能
        
        参数:
            agent1: 智能体1
            agent2: 智能体2
            num_games: 评估游戏数
            verbose: 是否打印详细信息
        """
        print(f"\n开始评估，共 {num_games} 局...")
        
        wins = {1: 0, -1: 0, 0: 0}
        
        for i in range(num_games):
            # 交替先手
            if i % 2 == 0:
                winner = self.play_game(agent1, agent2, training=False, verbose=verbose)
            else:
                winner = self.play_game(agent2, agent1, training=False, verbose=verbose)
                # 调整获胜者（因为顺序反了）
                if winner == 1:
                    winner = -1
                elif winner == -1:
                    winner = 1
            
            wins[winner] += 1
        
        print(f"\n评估结果:")
        print(f"  智能体1 (X) 胜: {wins[1]} 局 ({wins[1]/num_games*100:.1f}%)")
        print(f"  智能体2 (O) 胜: {wins[-1]} 局 ({wins[-1]/num_games*100:.1f}%)")
        print(f"  平局: {wins[0]} 局 ({wins[0]/num_games*100:.1f}%)")
        
        return wins
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'total_games': 0
        }
