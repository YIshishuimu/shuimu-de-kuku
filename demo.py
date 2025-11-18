#!/usr/bin/env python3
"""
快速演示井字棋AI训练和对战
Quick demonstration of Tic-Tac-Toe AI training and gameplay
"""

from tic_tac_toe import TicTacToe, QLearningAgent, RandomAgent, TicTacToeTrainer


def main():
    print("=" * 70)
    print("井字棋AI训练和对战演示")
    print("Tic-Tac-Toe AI Training and Gameplay Demo")
    print("=" * 70)
    print()
    
    # 1. 创建并训练智能体
    print("步骤 1: 创建并训练Q-learning智能体")
    print("-" * 70)
    agent = QLearningAgent(player=1, learning_rate=0.1, epsilon=0.2)
    opponent = RandomAgent(player=-1)
    trainer = TicTacToeTrainer()
    
    print("训练中... (2000轮对抗随机智能体)")
    trainer.train(agent, opponent, num_episodes=2000, print_interval=500)
    
    # 2. 评估性能
    print("\n步骤 2: 评估训练后的智能体性能")
    print("-" * 70)
    agent.set_epsilon(0.0)  # 关闭探索
    wins = trainer.evaluate(agent, opponent, num_games=100, verbose=False)
    
    # 3. 演示游戏
    print("\n步骤 3: 演示智能体对战")
    print("-" * 70)
    print("观看AI (X) 与随机智能体 (O) 对战...\n")
    
    for game_num in range(3):
        print(f"第 {game_num + 1} 局:")
        game = TicTacToe()
        game.reset()
        
        while True:
            current_player = game.current_player
            
            if current_player == 1:
                # AI回合
                action = agent.choose_action(game, training=False)
            else:
                # 随机智能体回合
                action = opponent.choose_action(game)
            
            success, winner, done = game.make_move(action)
            
            if done:
                game.render()
                if winner == 0:
                    print("  结果: 平局\n")
                elif winner == 1:
                    print("  结果: AI (X) 获胜! 🎉\n")
                else:
                    print("  结果: 随机智能体 (O) 获胜\n")
                break
    
    # 4. 展示Q表信息
    print("步骤 4: Q表统计信息")
    print("-" * 70)
    print(f"学习的状态数: {len(agent.q_table)}")
    total_actions = sum(len(actions) for actions in agent.q_table.values())
    print(f"学习的状态-动作对数: {total_actions}")
    print(f"平均每个状态的动作数: {total_actions / len(agent.q_table):.2f}")
    
    # 5. 展示一些Q值
    print("\n步骤 5: 示例Q值 (前5个状态)")
    print("-" * 70)
    for i, (state_key, actions) in enumerate(list(agent.q_table.items())[:5]):
        print(f"\n状态 {i+1}:")
        for action, q_value in list(actions.items())[:3]:
            print(f"  动作 {action}: Q值 = {q_value:.4f}")
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)
    print("\n你可以:")
    print("1. 运行 'python train_model.py' 训练一个更强的模型")
    print("2. 运行 'python play_game.py' 与训练好的AI对战")
    print("3. 运行 'python test_tic_tac_toe.py' 运行测试套件")


if __name__ == '__main__':
    main()
