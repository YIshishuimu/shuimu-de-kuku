"""
Modified Tic-Tac-Toe Game Environment

Game Rules:
1. 3x3 board
2. Two players alternate turns
3. Win condition: Complete a COLUMN (not row)
4. Each player has 3-4 pieces
5. Special action: Remove opponent's piece (max 2 times per game)
6. Removed pieces return to player's reserve
"""

import numpy as np
from typing import Tuple, List, Optional


class ModifiedTicTacToe:
    """Modified Tic-Tac-Toe game environment"""
    
    def __init__(self, pieces_per_player: int = 4):
        """
        Initialize the game.
        
        Args:
            pieces_per_player: Number of pieces each player starts with (3-4)
        """
        assert 3 <= pieces_per_player <= 4, "Each player must have 3-4 pieces"
        
        self.pieces_per_player = pieces_per_player
        self.board = np.zeros((3, 3), dtype=int)  # 0: empty, 1: player 1, 2: player 2
        self.current_player = 1
        self.game_over = False
        self.winner = None
        
        # Track pieces in reserve (not yet placed on board)
        self.reserve = {1: pieces_per_player, 2: pieces_per_player}
        
        # Track removal opportunities
        self.removals_left = {1: 2, 2: 2}
        
        # Track pieces on board
        self.pieces_on_board = {1: 0, 2: 0}
        
    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.reserve = {1: self.pieces_per_player, 2: self.pieces_per_player}
        self.removals_left = {1: 2, 2: 2}
        self.pieces_on_board = {1: 0, 2: 0}
        return self.get_state()
        
    def get_state(self) -> Tuple:
        """
        Get current game state as a tuple.
        
        Returns:
            Tuple containing board state and game metadata
        """
        return (
            tuple(self.board.flatten()),
            self.current_player,
            tuple(self.reserve.values()),
            tuple(self.removals_left.values())
        )
    
    def get_valid_actions(self) -> List[Tuple[str, int, int]]:
        """
        Get list of valid actions for current player.
        
        Returns:
            List of tuples (action_type, row, col) where:
            - action_type: 'place' or 'remove'
            - row, col: board coordinates (0-2)
        """
        actions = []
        
        # Can place a piece if have pieces in reserve
        if self.reserve[self.current_player] > 0:
            for i in range(3):
                for j in range(3):
                    if self.board[i, j] == 0:
                        actions.append(('place', i, j))
        
        # Can remove opponent's piece if have removals left
        opponent = 3 - self.current_player
        if self.removals_left[self.current_player] > 0:
            for i in range(3):
                for j in range(3):
                    if self.board[i, j] == opponent:
                        actions.append(('remove', i, j))
        
        return actions
    
    def check_winner(self) -> Optional[int]:
        """
        Check if there's a winner (complete column).
        
        Returns:
            Player number (1 or 2) if winner exists, None otherwise
        """
        # Check columns only
        for col in range(3):
            if self.board[0, col] != 0 and \
               self.board[0, col] == self.board[1, col] == self.board[2, col]:
                return self.board[0, col]
        
        return None
    
    def is_board_full(self) -> bool:
        """Check if board is full"""
        return np.all(self.board != 0)
    
    def step(self, action: Tuple[str, int, int]) -> Tuple[Tuple, float, bool, dict]:
        """
        Execute an action and return the result.
        
        Args:
            action: Tuple (action_type, row, col)
            
        Returns:
            Tuple (state, reward, done, info)
        """
        if self.game_over:
            raise ValueError("Game is already over")
        
        action_type, row, col = action
        
        if action not in self.get_valid_actions():
            raise ValueError(f"Invalid action: {action}")
        
        reward = 0
        
        if action_type == 'place':
            # Place piece on board
            self.board[row, col] = self.current_player
            self.reserve[self.current_player] -= 1
            self.pieces_on_board[self.current_player] += 1
            
        elif action_type == 'remove':
            # Remove opponent's piece
            opponent = 3 - self.current_player
            self.board[row, col] = 0
            self.reserve[opponent] += 1  # Return to opponent's reserve
            self.removals_left[self.current_player] -= 1
            self.pieces_on_board[opponent] -= 1
        
        # Check for winner
        winner = self.check_winner()
        if winner:
            self.game_over = True
            self.winner = winner
            reward = 1 if winner == self.current_player else -1
        
        # Check for draw (no more valid moves and no winner)
        elif len(self.get_valid_actions()) == 0:
            self.game_over = True
            self.winner = 0  # Draw
            reward = 0
        
        # Switch player
        if not self.game_over:
            self.current_player = 3 - self.current_player
        
        info = {
            'winner': self.winner,
            'valid_actions': self.get_valid_actions()
        }
        
        return self.get_state(), reward, self.game_over, info
    
    def render(self):
        """Print the current board state"""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print("\n  0 1 2")
        for i in range(3):
            print(f"{i} ", end="")
            for j in range(3):
                print(f"{symbols[self.board[i, j]]} ", end="")
            print()
        
        print(f"\nCurrent player: {'X' if self.current_player == 1 else 'O'}")
        print(f"Reserve - X: {self.reserve[1]}, O: {self.reserve[2]}")
        print(f"Removals left - X: {self.removals_left[1]}, O: {self.removals_left[2]}")
        
        if self.game_over:
            if self.winner == 0:
                print("Game Over: Draw!")
            else:
                winner_symbol = 'X' if self.winner == 1 else 'O'
                print(f"Game Over: {winner_symbol} wins!")
    
    def get_board_copy(self) -> np.ndarray:
        """Get a copy of the current board"""
        return self.board.copy()
