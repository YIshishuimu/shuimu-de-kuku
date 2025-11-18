#!/usr/bin/env python3
"""
井字棋对战演示
Play against the trained Tic-Tac-Toe AI

用法:
    python play_game.py                              # 加载默认模型对战
    python play_game.py --model models/my_model.pkl  # 加载指定模型
"""

import argparse
import os
from tic_tac_toe import TicTacToe, QLearningAgent


def get_human_move(game: TicTacToe) -> tuple:
    """获取人类玩家的移动"""
    while True:
        try:
            print("请输入你的移动 (行 列，例如: 0 0 表示左上角):")
            print("行和列的范围都是 0-2")
            move_input = input("> ").strip()
            
            if move_input.lower() == 'q':
                return None
            
            parts = move_input.split()
            if len(parts) != 2:
                print("❌ 输入格式错误！请输入两个数字，用空格分隔")
                continue
            
            row, col = int(parts[0]), int(parts[1])
            
            if not game.is_valid_action((row, col)):
                print("❌ 无效的移动！该位置已被占用或超出范围")
                continue
            
            return (row, col)
            
        except ValueError:
            print("❌ 请输入有效的数字！")
        except KeyboardInterrupt:
            print("\n游戏已退出")
            return None


def play_game(agent: QLearningAgent, human_first: bool = True):
    """人类对战AI"""
    game = TicTacToe()
    game.reset()
    
    print("\n" + "=" * 60)
    print("井字棋对战开始！")
    print("=" * 60)
    print(f"你是: {'X (先手)' if human_first else 'O (后手)'}")
    print(f"AI是: {'O (后手)' if human_first else 'X (先手)'}")
    print("输入 'q' 退出游戏")
    print()
    
    human_player = 1 if human_first else -1
    agent.player = -human_player
    agent.set_epsilon(0.0)  # 不探索，使用最佳策略
    
    game.render()
    
    while True:
        current_player = game.current_player
        
        if current_player == human_player:
            # 人类玩家回合
            print("轮到你了！")
            action = get_human_move(game)
            
            if action is None:
                print("游戏已退出")
                break
        else:
            # AI回合
            print("AI正在思考...")
            action = agent.choose_action(game, training=False)
            print(f"AI选择: {action}")
        
        # 执行移动
        success, winner, done = game.make_move(action)
        
        # 显示棋盘
        game.render()
        
        if done:
            if winner == 0:
                print("🤝 平局！")
            elif winner == human_player:
                print("🎉 恭喜你赢了！")
            else:
                print("😔 AI赢了！")
            break
    
    print("\n游戏结束")


def main():
    parser = argparse.ArgumentParser(description='与训练好的井字棋AI对战')
    parser.add_argument('--model', type=str, default='models/tic_tac_toe_model.pkl',
                        help='模型文件路径 (默认: models/tic_tac_toe_model.pkl)')
    parser.add_argument('--ai-first', action='store_true',
                        help='让AI先手')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        print("请先运行 train_model.py 训练模型")
        return
    
    # 加载模型
    print("加载模型...")
    agent = QLearningAgent()
    agent.load_model(args.model)
    print(f"✓ 模型加载成功！Q表包含 {len(agent.q_table)} 个状态")
    
    # 开始游戏
    while True:
        play_game(agent, human_first=not args.ai_first)
        
        # 询问是否再来一局
        print("\n再来一局? (y/n)")
        response = input("> ").strip().lower()
        if response != 'y' and response != 'yes':
            break
    
    print("\n感谢游玩！")


if __name__ == '__main__':
    main()
