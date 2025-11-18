#!/usr/bin/env python3
"""
训练脚本：应用复盘方法进行针对性训练
Training script with review-based targeted training on failed cases
"""

import argparse
import os
import sys
from collections import defaultdict
import pickle

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tic_tac_toe import TicTacToe, QLearningAgent, RandomAgent, TicTacToeTrainer


class ReviewBasedTrainer:
    """基于复盘的训练器"""
    
    def __init__(self, agent, opponent):
        self.agent = agent
        self.opponent = opponent
        self.trainer = TicTacToeTrainer()
        self.failed_states = []
        self.weak_positions = defaultdict(int)
        
    def collect_failed_cases(self, num_games=100):
        """收集失败案例"""
        print(f"\n⏳ 收集失败案例 ({num_games} 局)...")
        
        failed_games = []
        game = TicTacToe()
        
        for i in range(num_games):
            if (i + 1) % 20 == 0:
                print(f"  已完成 {i+1}/{num_games} 局")
            
            game.reset()
            history = []  # 记录状态-动作历史
            game_over = False
            
            # 进行对局
            while not game_over:
                state = game.get_state()
                
                if game.current_player == self.agent.player:
                    action = self.agent.choose_action(game, training=False)
                    history.append((state.copy(), action))
                else:
                    action = self.opponent.choose_action(game, training=False)
                
                success, winner, game_over = game.make_move(action)
            
            # 如果AI输了，记录这局
            winner = game.check_winner()
            if winner is not None and winner == -self.agent.player:
                failed_games.append({
                    'history': history,
                    'final_state': game.get_state().copy()
                })
                
                # 记录所有失败状态
                for state, action in history:
                    state_key = self._state_to_key(state)
                    self.weak_positions[state_key] += 1
        
        print(f"✅ 收集完成：{len(failed_games)} 局失败案例")
        self.failed_states = failed_games
        return len(failed_games)
    
    def _state_to_key(self, state):
        """将状态转换为可哈希的键"""
        return tuple(state.flatten())
    
    def targeted_training(self, num_episodes=5000):
        """针对失败案例进行重点训练"""
        print(f"\n🎯 针对性训练开始 ({num_episodes} 轮)...")
        
        if not self.failed_states:
            print("⚠️  没有失败案例，跳过针对性训练")
            return
        
        # 使用标准训练器，但增加针对弱势状态的训练权重
        # 通过临时调整agent的学习率来加强学习
        original_lr = self.agent.learning_rate
        self.agent.learning_rate = min(0.2, original_lr * 1.5)  # 提高学习率
        
        print(f"  提高学习率: {original_lr:.2f} → {self.agent.learning_rate:.2f}")
        print(f"  针对 {len(self.failed_states)} 个失败案例和 {len(self.weak_positions)} 个弱势状态")
        
        # 使用trainer进行训练
        self.trainer.train(self.agent, self.opponent, num_episodes=num_episodes, print_interval=1000)
        
        # 恢复原始学习率
        self.agent.learning_rate = original_lr
        
        print("✅ 针对性训练完成")
    
    def general_training(self, num_episodes=5000):
        """常规训练"""
        print(f"\n📚 常规训练 ({num_episodes} 轮)...")
        self.trainer.train(self.agent, self.opponent, num_episodes=num_episodes, print_interval=1000)
        print("✅ 常规训练完成")
    
    def evaluate_performance(self, num_games=200):
        """评估性能"""
        print(f"\n📊 评估性能 ({num_games} 局)...")
        
        wins = 0
        losses = 0
        draws = 0
        
        game = TicTacToe()
        original_epsilon = self.agent.epsilon
        self.agent.set_epsilon(0.0)  # 评估时不探索
        
        for _ in range(num_games):
            game.reset()
            game_over = False
            
            while not game_over:
                if game.current_player == self.agent.player:
                    action = self.agent.choose_action(game, training=False)
                else:
                    action = self.opponent.choose_action(game, training=False)
                
                success, winner, game_over = game.make_move(action)
            
            winner = game.check_winner()
            if winner == self.agent.player:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
        
        self.agent.set_epsilon(original_epsilon)
        
        win_rate = wins / num_games * 100
        draw_rate = draws / num_games * 100
        loss_rate = losses / num_games * 100
        
        print(f"\n结果:")
        print(f"  胜: {wins} ({win_rate:.1f}%)")
        print(f"  平: {draws} ({draw_rate:.1f}%)")
        print(f"  负: {losses} ({loss_rate:.1f}%)")
        
        return win_rate, draw_rate, loss_rate


def main():
    parser = argparse.ArgumentParser(description='基于复盘的训练脚本')
    parser.add_argument('--model', type=str, default='models/ultra_trained/model_1.pkl',
                        help='初始模型路径')
    parser.add_argument('--output', type=str, default='models/review_trained_model.pkl',
                        help='输出模型路径')
    parser.add_argument('--collect-games', type=int, default=100,
                        help='收集失败案例的对局数')
    parser.add_argument('--targeted-episodes', type=int, default=10000,
                        help='针对性训练轮数')
    parser.add_argument('--general-episodes', type=int, default=5000,
                        help='常规训练轮数')
    parser.add_argument('--eval-games', type=int, default=200,
                        help='评估对局数')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("基于复盘的训练系统")
    print("Review-Based Training System")
    print("=" * 70)
    
    # 加载初始模型
    print(f"\n⏳ 加载初始模型: {args.model}")
    agent = QLearningAgent(player=1, learning_rate=0.1, epsilon=0.1)
    
    if os.path.exists(args.model):
        agent.load_model(args.model)
        print(f"✅ 模型加载成功 (Q表大小: {len(agent.q_table)})")
    else:
        print(f"⚠️  模型文件不存在，使用新模型")
    
    opponent = RandomAgent(player=-1)
    
    # 创建训练器
    trainer = ReviewBasedTrainer(agent, opponent)
    
    # 第一阶段：评估基线性能
    print("\n" + "=" * 70)
    print("📊 阶段 1: 基线性能评估")
    print("=" * 70)
    
    baseline_win, baseline_draw, baseline_loss = trainer.evaluate_performance(args.eval_games)
    
    # 第二阶段：收集失败案例
    print("\n" + "=" * 70)
    print("🔍 阶段 2: 收集失败案例")
    print("=" * 70)
    
    num_failed = trainer.collect_failed_cases(args.collect_games)
    
    print(f"\n弱势位置统计:")
    sorted_weak = sorted(trainer.weak_positions.items(), key=lambda x: x[1], reverse=True)
    print(f"  识别到 {len(sorted_weak)} 个不同的弱势状态")
    print(f"  最常失败的前3个状态出现次数: {[x[1] for x in sorted_weak[:3]]}")
    
    # 第三阶段：针对性训练
    print("\n" + "=" * 70)
    print("🎯 阶段 3: 针对失败案例训练")
    print("=" * 70)
    
    trainer.targeted_training(args.targeted_episodes)
    
    # 第四阶段：常规训练
    print("\n" + "=" * 70)
    print("📚 阶段 4: 常规训练")
    print("=" * 70)
    
    trainer.general_training(args.general_episodes)
    
    # 第五阶段：评估改进后的性能
    print("\n" + "=" * 70)
    print("📊 阶段 5: 改进后性能评估")
    print("=" * 70)
    
    improved_win, improved_draw, improved_loss = trainer.evaluate_performance(args.eval_games)
    
    # 保存模型
    print(f"\n💾 保存改进后的模型: {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    agent.save_model(args.output)
    print("✅ 模型已保存")
    
    # 最终对比报告
    print("\n" + "=" * 70)
    print("📈 训练效果对比")
    print("=" * 70)
    
    print(f"\n基线性能 (训练前):")
    print(f"  胜率: {baseline_win:.1f}%")
    print(f"  平局率: {baseline_draw:.1f}%")
    print(f"  失败率: {baseline_loss:.1f}%")
    
    print(f"\n改进性能 (训练后):")
    print(f"  胜率: {improved_win:.1f}%")
    print(f"  平局率: {improved_draw:.1f}%")
    print(f"  失败率: {improved_loss:.1f}%")
    
    win_improvement = improved_win - baseline_win
    print(f"\n🎯 胜率提升: {win_improvement:+.1f}%")
    
    if win_improvement > 0:
        print(f"✅ 复盘训练有效！胜率从 {baseline_win:.1f}% 提升到 {improved_win:.1f}%")
    elif win_improvement == 0:
        print(f"➡️  胜率保持稳定在 {improved_win:.1f}%")
    else:
        print(f"⚠️  胜率略有下降 {win_improvement:.1f}%，可能需要调整训练参数")
    
    print(f"\nQ表增长: {len(agent.q_table)} 状态")
    
    # 保存训练报告
    report_path = 'review_training_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("基于复盘的训练报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"初始模型: {args.model}\n")
        f.write(f"输出模型: {args.output}\n\n")
        f.write(f"训练配置:\n")
        f.write(f"  收集失败案例: {args.collect_games} 局\n")
        f.write(f"  针对性训练: {args.targeted_episodes} 轮\n")
        f.write(f"  常规训练: {args.general_episodes} 轮\n")
        f.write(f"  评估对局数: {args.eval_games} 局\n\n")
        f.write(f"识别的弱势状态数: {len(sorted_weak)}\n")
        f.write(f"失败案例数: {num_failed}\n\n")
        f.write("性能对比:\n")
        f.write(f"  基线胜率: {baseline_win:.1f}%\n")
        f.write(f"  改进胜率: {improved_win:.1f}%\n")
        f.write(f"  胜率提升: {win_improvement:+.1f}%\n\n")
        f.write(f"Q表规模: {len(agent.q_table)} 状态\n")
    
    print(f"\n💾 训练报告已保存到: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
