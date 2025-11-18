#!/usr/bin/env python3
"""
井字棋模型训练脚本
Tic-Tac-Toe Model Training Script

用法:
    python train_model.py               # 使用默认参数训练
    python train_model.py --episodes 50000 --epsilon 0.2  # 自定义参数
"""

import argparse
import os
from tic_tac_toe import QLearningAgent, RandomAgent, TicTacToeTrainer


def main():
    parser = argparse.ArgumentParser(description='训练井字棋AI模型')
    parser.add_argument('--episodes', type=int, default=20000,
                        help='训练轮数 (默认: 20000)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='学习率 (默认: 0.1)')
    parser.add_argument('--discount-factor', type=float, default=0.9,
                        help='折扣因子 (默认: 0.9)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='探索率 (默认: 0.1)')
    parser.add_argument('--output', type=str, default='models/tic_tac_toe_model.pkl',
                        help='模型保存路径 (默认: models/tic_tac_toe_model.pkl)')
    parser.add_argument('--eval-games', type=int, default=100,
                        help='评估游戏数 (默认: 100)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("井字棋AI模型训练")
    print("=" * 60)
    print(f"\n训练参数:")
    print(f"  训练轮数: {args.episodes}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  折扣因子: {args.discount_factor}")
    print(f"  探索率: {args.epsilon}")
    print(f"  模型保存路径: {args.output}")
    print()
    
    # 创建智能体
    agent = QLearningAgent(
        player=1,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon
    )
    
    # 创建对手（随机智能体）
    opponent = RandomAgent(player=-1)
    
    # 创建训练器
    trainer = TicTacToeTrainer()
    
    # 训练
    print("阶段 1: 对抗随机智能体训练")
    print("-" * 60)
    trainer.train(agent, opponent, num_episodes=args.episodes, print_interval=args.episodes//10)
    
    # 降低探索率进行精细调优
    print("\n阶段 2: 降低探索率进行精细调优")
    print("-" * 60)
    agent.set_epsilon(0.01)
    trainer.reset_stats()
    trainer.train(agent, opponent, num_episodes=args.episodes//5, print_interval=args.episodes//20)
    
    # 评估
    print("\n" + "=" * 60)
    print("评估智能体性能")
    print("=" * 60)
    agent.set_epsilon(0.0)  # 评估时不探索
    trainer.evaluate(agent, opponent, num_games=args.eval_games)
    
    # 保存模型
    print("\n" + "=" * 60)
    print("保存模型")
    print("=" * 60)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    agent.save_model(args.output)
    
    print("\n训练完成！")
    print(f"模型已保存到: {args.output}")
    print(f"Q表大小: {len(agent.q_table)} 个状态")
    
    total_actions = sum(len(actions) for actions in agent.q_table.values())
    print(f"学习的动作数: {total_actions}")


if __name__ == '__main__':
    main()
