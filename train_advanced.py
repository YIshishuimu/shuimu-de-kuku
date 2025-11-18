#!/usr/bin/env python3
"""
高级井字棋模型训练脚本 - 提升胜率
Advanced Tic-Tac-Toe Model Training Script - Improved Win Rate

使用自我对弈和渐进式训练策略来提高AI胜率
Uses self-play and progressive training strategy to improve AI win rate

用法:
    python train_advanced.py                    # 使用默认参数
    python train_advanced.py --episodes 50000   # 自定义训练轮数
"""

import argparse
import os
from tic_tac_toe import QLearningAgent, RandomAgent, TicTacToeTrainer


def main():
    parser = argparse.ArgumentParser(description='高级训练：使用自我对弈提升AI胜率')
    parser.add_argument('--episodes', type=int, default=30000,
                        help='总训练轮数 (默认: 30000)')
    parser.add_argument('--learning-rate', type=float, default=0.15,
                        help='学习率 (默认: 0.15)')
    parser.add_argument('--discount-factor', type=float, default=0.95,
                        help='折扣因子 (默认: 0.95)')
    parser.add_argument('--output', type=str, default='models/advanced_model.pkl',
                        help='模型保存路径 (默认: models/advanced_model.pkl)')
    parser.add_argument('--eval-games', type=int, default=200,
                        help='评估游戏数 (默认: 200)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("井字棋AI高级训练 - 提升胜率")
    print("Advanced Tic-Tac-Toe AI Training - Improved Win Rate")
    print("=" * 70)
    print(f"\n训练参数:")
    print(f"  总训练轮数: {args.episodes}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  折扣因子: {args.discount_factor}")
    print(f"  模型保存路径: {args.output}")
    print()
    
    # 创建两个智能体进行自我对弈
    print("🎯 策略: 使用自我对弈训练 (两个Q-learning智能体互相对弈)")
    print("-" * 70)
    agent1 = QLearningAgent(
        player=1,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=0.3  # 开始时高探索率
    )
    
    agent2 = QLearningAgent(
        player=-1,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=0.3
    )
    
    trainer = TicTacToeTrainer()
    
    # 阶段1: 高探索率 - 探索各种策略
    print("\n阶段 1: 高探索期 (epsilon=0.3)")
    print("-" * 70)
    print("让两个AI互相对弈，大胆探索各种策略...")
    phase1_episodes = int(args.episodes * 0.4)
    trainer.train(agent1, agent2, num_episodes=phase1_episodes, 
                  print_interval=phase1_episodes//5)
    
    # 阶段2: 中等探索率 - 平衡探索和利用
    print("\n阶段 2: 平衡期 (epsilon=0.15)")
    print("-" * 70)
    print("降低探索率，在探索新策略和利用已学知识之间平衡...")
    agent1.set_epsilon(0.15)
    agent2.set_epsilon(0.15)
    trainer.reset_stats()
    phase2_episodes = int(args.episodes * 0.4)
    trainer.train(agent1, agent2, num_episodes=phase2_episodes,
                  print_interval=phase2_episodes//5)
    
    # 阶段3: 低探索率 - 精细调优
    print("\n阶段 3: 精细调优期 (epsilon=0.05)")
    print("-" * 70)
    print("进一步降低探索率，专注于优化已学策略...")
    agent1.set_epsilon(0.05)
    agent2.set_epsilon(0.05)
    trainer.reset_stats()
    phase3_episodes = int(args.episodes * 0.2)
    trainer.train(agent1, agent2, num_episodes=phase3_episodes,
                  print_interval=phase3_episodes//2)
    
    # 评估两个智能体
    print("\n" + "=" * 70)
    print("评估训练后的智能体性能")
    print("=" * 70)
    
    # 对抗随机智能体
    print("\n1. 智能体1 vs 随机对手")
    print("-" * 70)
    agent1.set_epsilon(0.0)
    random_opponent = RandomAgent(player=-1)
    wins1 = trainer.evaluate(agent1, random_opponent, num_games=args.eval_games)
    
    print("\n2. 智能体2 vs 随机对手")
    print("-" * 70)
    agent2.set_epsilon(0.0)
    agent2.player = 1  # 切换到先手
    wins2 = trainer.evaluate(agent2, random_opponent, num_games=args.eval_games)
    
    # 选择表现更好的智能体保存
    agent1_win_rate = wins1[1] / args.eval_games * 100
    agent2_win_rate = wins2[1] / args.eval_games * 100
    
    print("\n" + "=" * 70)
    print("性能对比")
    print("=" * 70)
    print(f"智能体1胜率: {agent1_win_rate:.1f}%")
    print(f"智能体2胜率: {agent2_win_rate:.1f}%")
    
    # 选择更好的智能体
    if agent1_win_rate >= agent2_win_rate:
        best_agent = agent1
        best_agent.player = 1
        print(f"\n✅ 选择智能体1进行保存 (胜率: {agent1_win_rate:.1f}%)")
    else:
        best_agent = agent2
        best_agent.player = 1
        print(f"\n✅ 选择智能体2进行保存 (胜率: {agent2_win_rate:.1f}%)")
    
    # 保存模型
    print("\n" + "=" * 70)
    print("保存最佳模型")
    print("=" * 70)
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    best_agent.save_model(args.output)
    
    print("\n训练完成！")
    print(f"最佳模型已保存到: {args.output}")
    print(f"Q表大小: {len(best_agent.q_table):,} 个状态")
    
    total_actions = sum(len(actions) for actions in best_agent.q_table.values())
    print(f"学习的动作数: {total_actions:,}")
    
    # 最终性能报告
    best_win_rate = max(agent1_win_rate, agent2_win_rate)
    print(f"\n🎯 最终胜率: {best_win_rate:.1f}%")
    
    if best_win_rate >= 60:
        print("🌟 优秀! AI达到了高级水平!")
    elif best_win_rate >= 50:
        print("✅ 很好! AI表现出色!")
    elif best_win_rate >= 45:
        print("✅ 良好! AI掌握了基本策略!")
    else:
        print("⚠️ AI还需要更多训练")
    
    print("\n使用方法:")
    print(f"  python play_game.py --model {args.output}")


if __name__ == '__main__':
    main()
