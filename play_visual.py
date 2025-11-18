#!/usr/bin/env python3
"""
井字棋可视化人机对战
Visualized Tic-Tac-Toe Human vs AI

用法:
    python play_visual.py                              # 使用默认模型
    python play_visual.py --model models/my_model.pkl  # 指定模型
"""

import argparse
import os
import sys
from tic_tac_toe import TicTacToe, QLearningAgent


class VisualTicTacToe:
    """可视化井字棋游戏"""
    
    def __init__(self):
        self.colors = {
            'reset': '\033[0m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
        }
    
    def print_colored(self, text, color='reset', bold=False):
        """打印彩色文本"""
        prefix = self.colors['bold'] if bold else ''
        print(f"{prefix}{self.colors.get(color, '')}{text}{self.colors['reset']}")
    
    def print_board(self, game: TicTacToe):
        """打印美化的棋盘"""
        board = game.get_state()
        print("\n" + "=" * 50)
        print("           井 字 棋 游 戏 棋 盘")
        print("=" * 50)
        
        # 打印列号
        print("\n      0       1       2")
        print("  " + "╔═══════╦═══════╦═══════╗")
        
        for i in range(3):
            # 打印行内容
            row_content = f"{i} ║"
            for j in range(3):
                value = board[i, j]
                if value == 1:
                    # X (人类) - 红色
                    cell = f"{self.colors['red']}   X   {self.colors['reset']}"
                elif value == -1:
                    # O (AI) - 蓝色
                    cell = f"{self.colors['blue']}   O   {self.colors['reset']}"
                else:
                    # 空位 - 显示坐标
                    cell = f"  {i},{j}  "
                row_content += cell + "║"
            print(row_content)
            
            # 打印分隔线
            if i < 2:
                print("  ╠═══════╬═══════╬═══════╣")
        
        print("  ╚═══════╩═══════╩═══════╝")
        print()
    
    def print_game_stats(self, stats):
        """打印游戏统计"""
        print("\n" + "─" * 50)
        self.print_colored(f"📊 游戏统计", 'cyan', bold=True)
        print(f"  胜利次数: {stats['wins']}")
        print(f"  失败次数: {stats['losses']}")
        print(f"  平局次数: {stats['draws']}")
        total = stats['wins'] + stats['losses'] + stats['draws']
        if total > 0:
            win_rate = stats['wins'] / total * 100
            print(f"  胜率: {win_rate:.1f}%")
        print("─" * 50)
    
    def get_human_move(self, game: TicTacToe):
        """获取人类玩家的移动"""
        while True:
            try:
                self.print_colored("\n🎮 轮到你了！", 'green', bold=True)
                print("请输入你要落子的位置:")
                print("  - 输入 '行号,列号' (例如: 0,0 表示左上角)")
                print("  - 输入 'q' 退出游戏")
                print("  - 输入 'h' 查看帮助")
                
                move_input = input("\n>>> ").strip().lower()
                
                if move_input == 'q':
                    return None
                
                if move_input == 'h':
                    self.print_help()
                    continue
                
                # 解析输入
                parts = move_input.replace(' ', ',').split(',')
                if len(parts) != 2:
                    self.print_colored("❌ 格式错误！请输入 '行号,列号'", 'red')
                    continue
                
                row, col = int(parts[0]), int(parts[1])
                
                if not game.is_valid_action((row, col)):
                    self.print_colored("❌ 无效的移动！该位置已被占用或超出范围", 'red')
                    continue
                
                return (row, col)
                
            except ValueError:
                self.print_colored("❌ 请输入有效的数字！", 'red')
            except KeyboardInterrupt:
                print("\n")
                return None
    
    def print_help(self):
        """打印帮助信息"""
        print("\n" + "=" * 50)
        self.print_colored("📖 游戏帮助", 'yellow', bold=True)
        print("=" * 50)
        print("游戏规则:")
        print("  • 你是 X (红色), AI 是 O (蓝色)")
        print("  • 在3x3棋盘上，先连成一条线(横/竖/斜)者获胜")
        print("  • 行号和列号都是 0-2")
        print("\n输入格式:")
        print("  • 格式1: 0,0  (推荐)")
        print("  • 格式2: 0 0")
        print("\n坐标示例:")
        print("  左上角: 0,0  |  上中: 0,1  |  右上角: 0,2")
        print("  左中:   1,0  |  中心: 1,1  |  右中:   1,2")
        print("  左下角: 2,0  |  下中: 2,1  |  右下角: 2,2")
        print("\n特殊命令:")
        print("  q - 退出游戏")
        print("  h - 显示此帮助")
        print("=" * 50)
    
    def play_game(self, agent: QLearningAgent, human_first: bool = True):
        """开始游戏"""
        game = TicTacToe()
        game.reset()
        
        print("\n" + "╔" + "═" * 48 + "╗")
        self.print_colored("║     🎮  井字棋人机对战 - 可视化版本  🎮     ║", 'cyan', bold=True)
        print("╚" + "═" * 48 + "╝")
        
        if human_first:
            self.print_colored("\n✨ 你是 X (红色), AI 是 O (蓝色)", 'green')
            self.print_colored("你先手！", 'green', bold=True)
            human_player = 1
        else:
            self.print_colored("\n✨ 你是 O (蓝色), AI 是 X (红色)", 'green')
            self.print_colored("AI先手！", 'yellow', bold=True)
            human_player = -1
        
        agent.player = -human_player
        agent.set_epsilon(0.0)  # 使用最佳策略
        
        self.print_board(game)
        
        while True:
            current_player = game.current_player
            
            if current_player == human_player:
                # 人类玩家回合
                action = self.get_human_move(game)
                if action is None:
                    self.print_colored("\n👋 游戏已退出", 'yellow')
                    return None
            else:
                # AI回合
                self.print_colored("\n🤖 AI正在思考...", 'blue', bold=True)
                action = agent.choose_action(game, training=False)
                self.print_colored(f"AI选择: {action[0]},{action[1]}", 'blue')
                input("按 Enter 继续...")
            
            # 执行移动
            success, winner, done = game.make_move(action)
            
            # 显示棋盘
            self.print_board(game)
            
            if done:
                if winner == 0:
                    self.print_colored("🤝 平局！势均力敌！", 'yellow', bold=True)
                    return 'draw'
                elif winner == human_player:
                    self.print_colored("🎉🎉🎉 恭喜你赢了！🎉🎉🎉", 'green', bold=True)
                    return 'win'
                else:
                    self.print_colored("😔 AI赢了！继续加油！", 'red', bold=True)
                    return 'loss'


def main():
    parser = argparse.ArgumentParser(description='井字棋可视化人机对战')
    parser.add_argument('--model', type=str, default='models/ultra_trained/model_1.pkl',
                        help='模型文件路径 (默认: models/ultra_trained/model_1.pkl)')
    parser.add_argument('--ai-first', action='store_true',
                        help='让AI先手')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        print("\n可用的模型:")
        print("  • models/ultra_trained/model_1.pkl (胜率: 68%, 推荐)")
        print("  • models/ultra_trained/model_2.pkl (胜率: 68%)")
        print("  • models/ultra_trained/model_3.pkl (胜率: 64%)")
        print("\n请先运行训练脚本:")
        print("  python train_ultra.py")
        return
    
    # 加载模型
    print("⏳ 加载AI模型...")
    agent = QLearningAgent()
    agent.load_model(args.model)
    print(f"✅ 模型加载成功！")
    print(f"   Q表包含 {len(agent.q_table):,} 个学习状态")
    
    # 创建可视化游戏
    visual_game = VisualTicTacToe()
    
    # 游戏统计
    stats = {'wins': 0, 'losses': 0, 'draws': 0}
    
    # 游戏循环
    while True:
        result = visual_game.play_game(agent, human_first=not args.ai_first)
        
        if result is None:
            break
        
        # 更新统计
        if result == 'win':
            stats['wins'] += 1
        elif result == 'loss':
            stats['losses'] += 1
        else:
            stats['draws'] += 1
        
        visual_game.print_game_stats(stats)
        
        # 询问是否继续
        print("\n" + "─" * 50)
        response = input("再来一局？(y/n): ").strip().lower()
        if response not in ['y', 'yes', '是']:
            break
    
    print("\n" + "=" * 50)
    visual_game.print_colored("感谢游玩！", 'cyan', bold=True)
    visual_game.print_game_stats(stats)
    print("=" * 50)
    print()


if __name__ == '__main__':
    main()
