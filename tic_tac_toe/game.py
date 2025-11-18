"""
井字棋游戏逻辑
Tic-Tac-Toe game logic implementation
"""
import numpy as np
from typing import Optional, Tuple, List


class TicTacToe:
    """井字棋游戏类"""
    
    def __init__(self):
        """初始化游戏板"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        
    def reset(self):
        """重置游戏板"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """获取当前游戏状态"""
        return self.board.copy()
    
    def get_state_key(self) -> str:
        """获取游戏状态的唯一标识符"""
        return str(self.board.flatten().tolist())
    
    def get_available_actions(self) -> List[Tuple[int, int]]:
        """获取所有可用的动作（空位置）"""
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    actions.append((i, j))
        return actions
    
    def is_valid_action(self, action: Tuple[int, int]) -> bool:
        """检查动作是否有效"""
        i, j = action
        if i < 0 or i >= 3 or j < 0 or j >= 3:
            return False
        return self.board[i, j] == 0
    
    def make_move(self, action: Tuple[int, int]) -> Tuple[bool, Optional[int], bool]:
        """
        执行一个动作
        返回：(是否成功, 获胜者, 是否结束)
        """
        if not self.is_valid_action(action):
            return False, None, False
        
        i, j = action
        self.board[i, j] = self.current_player
        
        # 检查是否获胜
        winner = self.check_winner()
        if winner is not None:
            return True, winner, True
        
        # 检查是否平局
        if len(self.get_available_actions()) == 0:
            return True, 0, True  # 0 表示平局
        
        # 切换玩家
        self.current_player = -self.current_player
        return True, None, False
    
    def check_winner(self) -> Optional[int]:
        """
        检查是否有获胜者
        返回: 1 (X赢), -1 (O赢), 0 (平局), None (游戏继续)
        """
        # 检查行
        for i in range(3):
            if abs(self.board[i, :].sum()) == 3:
                return int(self.board[i, 0])
        
        # 检查列
        for j in range(3):
            if abs(self.board[:, j].sum()) == 3:
                return int(self.board[0, j])
        
        # 检查对角线
        if abs(self.board.diagonal().sum()) == 3:
            return int(self.board[0, 0])
        
        if abs(np.fliplr(self.board).diagonal().sum()) == 3:
            return int(self.board[0, 2])
        
        return None
    
    def render(self):
        """打印游戏板"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("\n")
        for i in range(3):
            row = ' '.join([symbols[int(self.board[i, j])] for j in range(3)])
            print(f"  {row}")
        print("\n")
    
    def get_reward(self, winner: Optional[int], player: int) -> float:
        """
        获取奖励
        player: 当前玩家 (1 或 -1)
        """
        if winner is None:
            return 0.0  # 游戏继续
        elif winner == 0:
            return 0.5  # 平局
        elif winner == player:
            return 1.0  # 获胜
        else:
            return -1.0  # 失败
