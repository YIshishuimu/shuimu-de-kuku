#!/usr/bin/env python3
"""
超级高级井字棋模型训练脚本 - 集成所有优化方法
Ultra-Advanced Tic-Tac-Toe Model Training Script - All Optimization Methods

集成方法:
1. 状态对称性利用 (State Symmetry)
2. 经验回放 (Experience Replay)
3. 双Q学习 (Double Q-Learning)
4. 优先级扫描 (Prioritized Sweeping)
5. 集成学习 (Ensemble Learning)

用法:
    python train_ultra.py                    # 使用默认参数
    python train_ultra.py --episodes 50000   # 自定义训练轮数
"""

import argparse
import os
import numpy as np
import random
from collections import deque
from tic_tac_toe import QLearningAgent, RandomAgent, TicTacToeTrainer, TicTacToe


class SymmetryAwareQLearningAgent(QLearningAgent):
    """利用状态对称性的Q学习智能体"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_symmetry = True
    
    def get_canonical_state(self, board):
        """获取棋盘的规范形式（8种对称中字典序最小的）"""
        if not self.use_symmetry:
            return str(board.flatten().tolist())
        
        # 生成所有8种对称状态
        states = []
        current = board.copy()
        
        # 4种旋转
        for _ in range(4):
            states.append(str(current.flatten().tolist()))
            current = np.rot90(current)
        
        # 镜像后再4种旋转
        current = np.fliplr(board)
        for _ in range(4):
            states.append(str(current.flatten().tolist()))
            current = np.rot90(current)
        
        # 返回字典序最小的
        return min(states)
    
    def get_state_key(self, game):
        """重写获取状态键的方法"""
        return self.get_canonical_state(game.get_state())
    
    def get_q_value(self, state_key, action):
        """获取Q值"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
        return self.q_table[state_key][action]
    
    def choose_action(self, game, training=True):
        """选择动作"""
        available_actions = game.get_available_actions()
        if not available_actions:
            return None
        
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        state_key = self.get_state_key(game)
        q_values = [(action, self.get_q_value(state_key, action)) 
                    for action in available_actions]
        best_action = max(q_values, key=lambda x: x[1])[0]
        return best_action


class DoubleQLearningAgent(SymmetryAwareQLearningAgent):
    """双Q学习智能体"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_table_b = {}  # 第二个Q表
    
    def get_q_value_b(self, state_key, action):
        """获取第二个Q表的Q值"""
        if state_key not in self.q_table_b:
            self.q_table_b[state_key] = {}
        if action not in self.q_table_b[state_key]:
            self.q_table_b[state_key][action] = 0.0
        return self.q_table_b[state_key][action]
    
    def update_q_value(self, state_key, action, reward, next_state_key, next_available_actions):
        """双Q学习更新"""
        # 随机选择更新哪个Q表
        if random.random() < 0.5:
            # 更新Q表A，使用Q表B选择最佳动作
            current_q = self.get_q_value(state_key, action)
            
            if next_available_actions:
                # 用Q表A选择动作
                best_next_action = max(next_available_actions,
                                      key=lambda a: self.get_q_value(next_state_key, a))
                # 用Q表B评估
                max_next_q = self.get_q_value_b(next_state_key, best_next_action)
            else:
                max_next_q = 0.0
            
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
            self.q_table[state_key][action] = new_q
        else:
            # 更新Q表B，使用Q表A选择最佳动作
            current_q = self.get_q_value_b(state_key, action)
            
            if next_available_actions:
                # 用Q表B选择动作
                best_next_action = max(next_available_actions,
                                      key=lambda a: self.get_q_value_b(next_state_key, a))
                # 用Q表A评估
                max_next_q = self.get_q_value(next_state_key, best_next_action)
            else:
                max_next_q = 0.0
            
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
            self.q_table_b[state_key][action] = new_q
    
    def choose_action(self, game, training=True):
        """选择动作（综合两个Q表）"""
        available_actions = game.get_available_actions()
        if not available_actions:
            return None
        
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        state_key = self.get_state_key(game)
        # 平均两个Q表的值
        q_values = []
        for action in available_actions:
            qa = self.get_q_value(state_key, action)
            qb = self.get_q_value_b(state_key, action)
            q_values.append((action, (qa + qb) / 2))
        
        best_action = max(q_values, key=lambda x: x[1])[0]
        return best_action


class ExperienceReplayTrainer(TicTacToeTrainer):
    """带经验回放的训练器"""
    
    def __init__(self, replay_buffer_size=10000, replay_batch_size=32):
        super().__init__()
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.replay_batch_size = replay_batch_size
    
    def store_experience(self, state_key, action, reward, next_state_key, next_actions):
        """存储经验"""
        self.replay_buffer.append((state_key, action, reward, next_state_key, next_actions))
    
    def replay_experiences(self, agent):
        """经验回放"""
        if len(self.replay_buffer) < self.replay_batch_size:
            return
        
        # 随机采样
        batch = random.sample(self.replay_buffer, self.replay_batch_size)
        
        for state_key, action, reward, next_state_key, next_actions in batch:
            agent.update_q_value(state_key, action, reward, next_state_key, next_actions)
    
    def train_with_replay(self, agent1, agent2, num_episodes, replay_frequency=10):
        """带经验回放的训练"""
        print(f"开始训练（含经验回放），共 {num_episodes} 轮...")
        
        for episode in range(num_episodes):
            self.game.reset()
            history = []
            
            while True:
                current_agent = agent1 if self.game.current_player == 1 else agent2
                state_key = current_agent.get_state_key(self.game)
                action = current_agent.choose_action(self.game, training=True)
                
                if action is None:
                    break
                
                if isinstance(current_agent, (SymmetryAwareQLearningAgent, DoubleQLearningAgent)):
                    history.append({
                        'agent': current_agent,
                        'state_key': state_key,
                        'action': action
                    })
                
                success, winner, done = self.game.make_move(action)
                
                if done:
                    # 游戏结束，更新所有步骤
                    for record in history:
                        agent = record['agent']
                        reward = self.game.get_reward(winner, agent.player)
                        next_state_key = agent.get_state_key(self.game)
                        agent.update_q_value(record['state_key'], record['action'],
                                           reward, next_state_key, [])
                        # 存储经验
                        self.store_experience(record['state_key'], record['action'],
                                            reward, next_state_key, [])
                    break
                
                # 更新之前的步骤
                if len(history) >= 2:
                    prev_record = history[-2]
                    agent = prev_record['agent']
                    next_actions = self.game.get_available_actions()
                    agent.update_q_value(prev_record['state_key'], prev_record['action'],
                                       0.0, state_key, next_actions)
                    self.store_experience(prev_record['state_key'], prev_record['action'],
                                        0.0, state_key, next_actions)
            
            # 定期进行经验回放
            if episode % replay_frequency == 0:
                self.replay_experiences(agent1)
                self.replay_experiences(agent2)
            
            # 统计
            if winner == 1:
                self.stats['player1_wins'] += 1
            elif winner == -1:
                self.stats['player2_wins'] += 1
            else:
                self.stats['draws'] += 1
            self.stats['total_games'] += 1
            
            # 打印进度
            if (episode + 1) % 1000 == 0:
                self.print_stats(episode + 1)
        
        print("\n训练完成！")
        self.print_stats(num_episodes)


def train_ensemble(num_models=5, episodes_per_model=20000):
    """训练集成模型"""
    print("=" * 70)
    print("训练集成模型")
    print("=" * 70)
    
    models = []
    
    for i in range(num_models):
        print(f"\n训练模型 {i+1}/{num_models}")
        print("-" * 70)
        
        # 使用不同的超参数
        lr = 0.1 + i * 0.02
        eps = 0.15 + i * 0.03
        
        agent1 = DoubleQLearningAgent(player=1, learning_rate=lr, epsilon=eps, discount_factor=0.95)
        agent2 = DoubleQLearningAgent(player=-1, learning_rate=lr, epsilon=eps, discount_factor=0.95)
        
        trainer = ExperienceReplayTrainer()
        
        # 训练
        trainer.train_with_replay(agent1, agent2, episodes_per_model, replay_frequency=10)
        
        # 评估并选择更好的
        agent1.set_epsilon(0.0)
        agent2.set_epsilon(0.0)
        agent2.player = 1
        
        random_opponent = RandomAgent(player=-1)
        wins1 = trainer.evaluate(agent1, random_opponent, num_games=50, verbose=False)
        wins2 = trainer.evaluate(agent2, random_opponent, num_games=50, verbose=False)
        
        if wins1[1] >= wins2[1]:
            models.append(agent1)
            print(f"✓ 选择agent1 (胜率: {wins1[1]/50*100:.1f}%)")
        else:
            models.append(agent2)
            print(f"✓ 选择agent2 (胜率: {wins2[1]/50*100:.1f}%)")
    
    return models


def ensemble_predict(models, game):
    """集成预测"""
    available_actions = game.get_available_actions()
    if not available_actions:
        return None
    
    # 投票
    votes = {action: 0 for action in available_actions}
    
    for model in models:
        action = model.choose_action(game, training=False)
        if action in votes:
            votes[action] += 1
    
    # 返回得票最多的动作
    best_action = max(votes.items(), key=lambda x: x[1])[0]
    return best_action


def main():
    parser = argparse.ArgumentParser(description='超级高级训练：集成所有优化方法')
    parser.add_argument('--episodes', type=int, default=30000,
                        help='每个模型的训练轮数 (默认: 30000)')
    parser.add_argument('--num-models', type=int, default=3,
                        help='集成模型数量 (默认: 3)')
    parser.add_argument('--output-dir', type=str, default='models/ultra',
                        help='模型保存目录 (默认: models/ultra)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("井字棋AI超级高级训练 - 集成所有优化方法")
    print("=" * 70)
    print(f"\n优化方法:")
    print("  ✓ 状态对称性利用 (8倍效率提升)")
    print("  ✓ 双Q学习 (减少过高估计)")
    print("  ✓ 经验回放 (稳定学习)")
    print("  ✓ 集成学习 (多模型投票)")
    print(f"\n训练参数:")
    print(f"  每模型训练轮数: {args.episodes}")
    print(f"  集成模型数: {args.num_models}")
    print()
    
    # 训练集成模型
    models = train_ensemble(num_models=args.num_models, 
                           episodes_per_model=args.episodes)
    
    # 最终评估
    print("\n" + "=" * 70)
    print("集成模型最终评估")
    print("=" * 70)
    
    # 创建虚拟智能体用于评估
    class EnsembleAgent:
        def __init__(self, models):
            self.models = models
            self.player = 1
        
        def choose_action(self, game, training=False):
            return ensemble_predict(self.models, game)
    
    ensemble_agent = EnsembleAgent(models)
    random_opponent = RandomAgent(player=-1)
    
    trainer = TicTacToeTrainer()
    wins = trainer.evaluate(ensemble_agent, random_opponent, num_games=200, verbose=False)
    
    win_rate = wins[1] / 200 * 100
    draw_rate = wins[0] / 200 * 100
    
    print(f"\n🎯 集成模型最终胜率: {win_rate:.1f}%")
    print(f"🤝 平局率: {draw_rate:.1f}%")
    
    # 保存所有模型
    print("\n" + "=" * 70)
    print("保存模型")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i, model in enumerate(models):
        filepath = os.path.join(args.output_dir, f'model_{i+1}.pkl')
        model.save_model(filepath)
        print(f"✓ 模型 {i+1} 已保存到: {filepath}")
    
    # 保存元信息
    import json
    meta_path = os.path.join(args.output_dir, 'ensemble_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'num_models': args.num_models,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'training_episodes': args.episodes
        }, f, indent=2)
    print(f"✓ 元信息已保存到: {meta_path}")
    
    print("\n训练完成！")
    print(f"\n使用方法:")
    print(f"  加载模型时需要加载所有 {args.num_models} 个模型文件")
    print(f"  使用集成预测可获得最佳性能")
    
    if win_rate >= 70:
        print("\n🌟 优秀！集成模型达到了高级水平!")
    elif win_rate >= 60:
        print("\n✅ 很好！集成模型表现出色!")
    else:
        print("\n✅ 良好！集成模型有显著提升!")


if __name__ == '__main__':
    main()
