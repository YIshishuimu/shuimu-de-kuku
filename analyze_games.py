#!/usr/bin/env python3
"""
井字棋对局分析与复盘工具
Game Analysis and Review Tool for Tic-Tac-Toe

分析AI输掉的对局，找出弱点并提供改进建议
Analyzes games that AI loses to identify weaknesses and improvement opportunities
"""

import argparse
import os
import pickle
from collections import defaultdict
from tic_tac_toe import TicTacToe, QLearningAgent, RandomAgent, TicTacToeTrainer


class GameAnalyzer:
    """对局分析器"""
    
    def __init__(self):
        self.lost_games = []
        self.win_games = []
        self.draw_games = []
        self.weak_states = defaultdict(int)  # 失败状态统计
        self.critical_mistakes = []  # 关键失误
    
    def record_game(self, game_history, result, ai_player):
        """记录一局游戏"""
        game_record = {
            'history': game_history,
            'result': result,
            'ai_player': ai_player
        }
        
        if result == ai_player:
            self.win_games.append(game_record)
        elif result == 0:
            self.draw_games.append(game_record)
        else:
            self.lost_games.append(game_record)
            # 记录失败状态
            for state, action, player in game_history:
                if player == ai_player:
                    state_key = str(state.flatten().tolist())
                    self.weak_states[state_key] += 1
    
    def analyze_losses(self):
        """分析失败对局"""
        print("\n" + "=" * 70)
        print("📊 失败对局分析")
        print("=" * 70)
        
        if not self.lost_games:
            print("没有失败的对局！")
            return
        
        print(f"\n总失败局数: {len(self.lost_games)}")
        print(f"总胜利局数: {len(self.win_games)}")
        print(f"总平局局数: {len(self.draw_games)}")
        
        # 分析失败原因
        self.analyze_loss_patterns()
        self.find_critical_mistakes()
        self.identify_weak_positions()
    
    def analyze_loss_patterns(self):
        """分析失败模式"""
        print("\n" + "-" * 70)
        print("🔍 失败模式分析")
        print("-" * 70)
        
        # 统计失败时的棋局长度
        loss_lengths = [len(game['history']) for game in self.lost_games]
        avg_length = sum(loss_lengths) / len(loss_lengths) if loss_lengths else 0
        
        print(f"\n平均失败回合数: {avg_length:.1f}")
        
        # 统计失败时AI的棋子数
        for i, game in enumerate(self.lost_games[:3], 1):  # 显示前3局
            print(f"\n失败对局 #{i}:")
            self.visualize_game(game)
    
    def find_critical_mistakes(self):
        """找出关键失误"""
        print("\n" + "-" * 70)
        print("⚠️  关键失误分析")
        print("-" * 70)
        
        mistakes = []
        
        for game in self.lost_games:
            history = game['history']
            ai_player = game['ai_player']
            
            # 检查是否错过获胜机会
            for i, (state, action, player) in enumerate(history):
                if player == ai_player and i < len(history) - 1:
                    # 检查是否有更好的走法
                    game_temp = TicTacToe()
                    game_temp.board = state.copy()
                    game_temp.current_player = player
                    
                    # 检查所有可能的走法
                    available = game_temp.get_available_actions()
                    for alt_action in available:
                        game_test = TicTacToe()
                        game_test.board = state.copy()
                        game_test.current_player = player
                        success, winner, done = game_test.make_move(alt_action)
                        
                        if done and winner == ai_player:
                            mistakes.append({
                                'game_index': len(mistakes),
                                'move_index': i,
                                'state': state,
                                'chosen_action': action,
                                'winning_action': alt_action,
                                'type': '错过获胜机会'
                            })
                            break
        
        if mistakes:
            print(f"\n发现 {len(mistakes)} 个关键失误")
            for mistake in mistakes[:3]:  # 显示前3个
                print(f"\n失误类型: {mistake['type']}")
                print(f"实际选择: {mistake['chosen_action']}")
                print(f"最佳选择: {mistake['winning_action']}")
        else:
            print("\n未发现明显的关键失误")
    
    def identify_weak_positions(self):
        """识别弱势位置"""
        print("\n" + "-" * 70)
        print("🎯 弱势位置识别")
        print("-" * 70)
        
        if not self.weak_states:
            print("\n没有明显的弱势位置")
            return
        
        # 找出最常失败的状态
        sorted_weak = sorted(self.weak_states.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n发现 {len(sorted_weak)} 个不同的失败状态")
        print(f"最常失败的前5个状态:")
        
        for i, (state_key, count) in enumerate(sorted_weak[:5], 1):
            print(f"\n状态 #{i} (失败 {count} 次):")
            # 解析并显示状态
            try:
                import ast
                state_list = ast.literal_eval(state_key)
                state_array = [[state_list[j*3+k] for k in range(3)] for j in range(3)]
                self.print_board(state_array)
            except:
                print("  (无法显示)")
    
    def visualize_game(self, game_record):
        """可视化一局游戏"""
        history = game_record['history']
        ai_player = game_record['ai_player']
        
        print(f"  AI棋子: {'X' if ai_player == 1 else 'O'}")
        print(f"  总回合数: {len(history)}")
        
        # 显示最终状态
        if history:
            final_state = history[-1][0]
            print("  最终棋盘:")
            self.print_board(final_state)
    
    def print_board(self, board):
        """打印棋盘"""
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for i in range(3):
            row = "    "
            for j in range(3):
                if isinstance(board[i], list):
                    val = board[i][j]
                else:
                    val = board[i, j]
                row += symbols.get(val, symbols[0]) + " "
            print(row)
    
    def generate_improvement_report(self):
        """生成改进建议报告"""
        print("\n" + "=" * 70)
        print("💡 改进建议")
        print("=" * 70)
        
        suggestions = []
        
        # 基于分析结果生成建议
        if len(self.lost_games) > len(self.win_games) * 0.3:
            suggestions.append({
                'priority': '高',
                'issue': '失败率偏高',
                'suggestion': '增加训练轮数，特别是自我对弈训练'
            })
        
        if self.weak_states:
            suggestions.append({
                'priority': '高',
                'issue': f'发现 {len(self.weak_states)} 个弱势状态',
                'suggestion': '对这些状态进行针对性训练，增加经验回放权重'
            })
        
        if self.critical_mistakes:
            suggestions.append({
                'priority': '中',
                'issue': '存在关键失误',
                'suggestion': '降低探索率(epsilon)，更多利用已学习的策略'
            })
        
        # 通用建议
        suggestions.append({
            'priority': '中',
            'issue': '持续优化',
            'suggestion': '使用更高级的训练方法(双Q学习、优先经验回放)'
        })
        
        for i, sug in enumerate(suggestions, 1):
            print(f"\n{i}. [{sug['priority']}优先级] {sug['issue']}")
            print(f"   建议: {sug['suggestion']}")
        
        return suggestions


def collect_game_data(agent, opponent, num_games=100):
    """收集对局数据"""
    print(f"\n⏳ 正在收集 {num_games} 局对局数据...")
    
    analyzer = GameAnalyzer()
    trainer = TicTacToeTrainer()
    
    agent.set_epsilon(0.0)  # 使用最佳策略
    
    for i in range(num_games):
        game = TicTacToe()
        game.reset()
        game_history = []
        
        while True:
            current_player = game.current_player
            
            if current_player == agent.player:
                action = agent.choose_action(game, training=False)
            else:
                action = opponent.choose_action(game, training=False)
            
            if action is None:
                break
            
            # 记录状态和动作
            state = game.get_state().copy()
            game_history.append((state, action, current_player))
            
            success, winner, done = game.make_move(action)
            
            if done:
                analyzer.record_game(game_history, winner, agent.player)
                break
        
        if (i + 1) % 20 == 0:
            print(f"  已完成 {i + 1}/{num_games} 局")
    
    print("✅ 数据收集完成")
    return analyzer


def main():
    parser = argparse.ArgumentParser(description='井字棋对局分析与复盘工具')
    parser.add_argument('--model', type=str, default='models/ultra_trained/model_1.pkl',
                        help='要分析的模型文件路径')
    parser.add_argument('--games', type=int, default=100,
                        help='分析的对局数量 (默认: 100)')
    parser.add_argument('--output', type=str, default='game_analysis.txt',
                        help='分析报告输出文件')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("井字棋对局分析与复盘工具")
    print("=" * 70)
    
    # 加载模型
    if not os.path.exists(args.model):
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        return
    
    print(f"\n⏳ 加载模型: {args.model}")
    agent = QLearningAgent()
    agent.load_model(args.model)
    print(f"✅ 模型加载成功 (Q表大小: {len(agent.q_table):,})")
    
    # 创建对手
    opponent = RandomAgent(player=-agent.player)
    
    # 收集数据
    analyzer = collect_game_data(agent, opponent, args.games)
    
    # 分析失败对局
    analyzer.analyze_losses()
    
    # 生成改进建议
    suggestions = analyzer.generate_improvement_report()
    
    # 保存报告
    print(f"\n💾 保存分析报告到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("井字棋对局分析报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"分析局数: {args.games}\n")
        f.write(f"胜利: {len(analyzer.win_games)}\n")
        f.write(f"失败: {len(analyzer.lost_games)}\n")
        f.write(f"平局: {len(analyzer.draw_games)}\n")
        f.write(f"胜率: {len(analyzer.win_games)/args.games*100:.1f}%\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("改进建议\n")
        f.write("=" * 70 + "\n\n")
        for i, sug in enumerate(suggestions, 1):
            f.write(f"{i}. [{sug['priority']}] {sug['issue']}\n")
            f.write(f"   {sug['suggestion']}\n\n")
    
    print("✅ 分析完成！")
    
    # 输出总结
    print("\n" + "=" * 70)
    print("📈 分析总结")
    print("=" * 70)
    win_rate = len(analyzer.win_games) / args.games * 100
    print(f"胜率: {win_rate:.1f}%")
    print(f"失败率: {len(analyzer.lost_games)/args.games*100:.1f}%")
    print(f"平局率: {len(analyzer.draw_games)/args.games*100:.1f}%")
    
    if win_rate >= 70:
        print("\n🌟 模型表现优秀！")
    elif win_rate >= 60:
        print("\n✅ 模型表现良好，可以考虑进一步优化")
    else:
        print("\n⚠️  模型需要改进，建议增加训练")


if __name__ == '__main__':
    main()
