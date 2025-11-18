#!/usr/bin/env python3
"""
测试井字棋AI功能
Test Tic-Tac-Toe AI functionality
"""

from tic_tac_toe import TicTacToe, QLearningAgent, RandomAgent, TicTacToeTrainer


def test_game_logic():
    """测试游戏逻辑"""
    print("测试1: 游戏逻辑")
    print("-" * 40)
    
    game = TicTacToe()
    game.reset()
    
    # 测试初始状态
    assert game.current_player == 1
    assert len(game.get_available_actions()) == 9
    
    # 测试移动
    success, winner, done = game.make_move((0, 0))
    assert success
    assert not done
    assert game.current_player == -1
    
    print("✓ 游戏逻辑测试通过")
    print()


def test_winning_condition():
    """测试获胜条件"""
    print("测试2: 获胜条件")
    print("-" * 40)
    
    game = TicTacToe()
    game.reset()
    
    # 测试横向获胜
    game.make_move((0, 0))  # X
    game.make_move((1, 0))  # O
    game.make_move((0, 1))  # X
    game.make_move((1, 1))  # O
    success, winner, done = game.make_move((0, 2))  # X wins
    
    assert done
    assert winner == 1
    
    print("✓ 获胜条件测试通过")
    print()


def test_agent_training():
    """测试智能体训练"""
    print("测试3: 智能体训练")
    print("-" * 40)
    
    agent = QLearningAgent(player=1, learning_rate=0.1, epsilon=0.1)
    opponent = RandomAgent(player=-1)
    trainer = TicTacToeTrainer()
    
    # 短期训练
    trainer.train(agent, opponent, num_episodes=100, print_interval=100)
    
    # 检查Q表是否更新
    assert len(agent.q_table) > 0
    
    print("✓ 智能体训练测试通过")
    print()


def test_model_save_load():
    """测试模型保存和加载"""
    print("测试4: 模型保存和加载")
    print("-" * 40)
    
    import os
    import tempfile
    
    agent = QLearningAgent(player=1)
    agent.q_table = {'test_state': {(0, 0): 0.5}}
    
    # 保存模型
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    agent.save_model(temp_path)
    
    # 加载模型
    new_agent = QLearningAgent()
    new_agent.load_model(temp_path)
    
    assert new_agent.q_table == agent.q_table
    
    # 清理
    os.unlink(temp_path)
    
    print("✓ 模型保存和加载测试通过")
    print()


def test_agent_vs_random():
    """测试训练后的智能体对抗随机智能体"""
    print("测试5: 训练后的智能体对抗随机智能体")
    print("-" * 40)
    
    agent = QLearningAgent(player=1, learning_rate=0.1, epsilon=0.1)
    opponent = RandomAgent(player=-1)
    trainer = TicTacToeTrainer()
    
    # 训练
    trainer.train(agent, opponent, num_episodes=1000, print_interval=1000)
    
    # 评估
    agent.set_epsilon(0.0)  # 不探索
    wins = trainer.evaluate(agent, opponent, num_games=50, verbose=False)
    
    # 训练后的智能体应该表现不错
    print(f"智能体胜率: {wins[1]/50*100:.1f}%")
    print(f"随机智能体胜率: {wins[-1]/50*100:.1f}%")
    print(f"平局率: {wins[0]/50*100:.1f}%")
    
    print("✓ 对抗测试完成")
    print()


def demo_game():
    """演示一局游戏"""
    print("演示: 完整的一局游戏")
    print("-" * 40)
    
    game = TicTacToe()
    game.reset()
    
    print("初始棋盘:")
    game.render()
    
    moves = [(0, 0), (1, 1), (0, 1), (2, 2), (0, 2)]
    
    for i, move in enumerate(moves):
        player = 'X' if game.current_player == 1 else 'O'
        print(f"玩家 {player} 移动到 {move}")
        success, winner, done = game.make_move(move)
        game.render()
        
        if done:
            if winner == 0:
                print("平局！")
            else:
                winner_symbol = 'X' if winner == 1 else 'O'
                print(f"玩家 {winner_symbol} 获胜！")
            break
    
    print()


def main():
    """运行所有测试"""
    print("=" * 60)
    print("井字棋AI测试套件")
    print("=" * 60)
    print()
    
    try:
        test_game_logic()
        test_winning_condition()
        test_agent_training()
        test_model_save_load()
        test_agent_vs_random()
        demo_game()
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
