"""
Demo script showing the capabilities of the RL Tic-Tac-Toe implementation
"""

from game_env import ModifiedTicTacToe
from agent import QLearningAgent, RandomAgent
import os


def demo_game_rules():
    """Demonstrate the game rules"""
    print("\n" + "="*60)
    print("DEMO: Game Rules Demonstration")
    print("="*60)
    
    env = ModifiedTicTacToe(pieces_per_player=4)
    
    print("\n1. Initial board (3x3):")
    env.render()
    
    print("\n2. Players take turns placing pieces...")
    env.step(('place', 0, 0))  # Player 1 at (0,0)
    env.render()
    
    env.step(('place', 0, 1))  # Player 2 at (0,1)
    env.render()
    
    env.step(('place', 1, 0))  # Player 1 at (1,0)
    env.render()
    
    print("\n3. Player 2 can REMOVE Player 1's piece:")
    env.step(('remove', 0, 0))  # Player 2 removes (0,0)
    env.render()
    print("Note: Player 1's piece returned to reserve!")
    
    print("\n4. Win condition: Complete a COLUMN (not row)")
    env.reset()
    env.step(('place', 0, 0))  # Player 1
    env.step(('place', 0, 1))  # Player 2
    env.step(('place', 1, 0))  # Player 1
    env.step(('place', 1, 1))  # Player 2
    env.step(('place', 2, 0))  # Player 1 - wins by completing column 0!
    env.render()
    

def demo_trained_agent():
    """Demonstrate a trained agent playing"""
    print("\n" + "="*60)
    print("DEMO: Trained Agent vs Random Opponent")
    print("="*60)
    
    if not os.path.exists('trained_agent_p1.pkl'):
        print("No trained agent found. Please run train.py first.")
        return
    
    env = ModifiedTicTacToe(pieces_per_player=4)
    agent = QLearningAgent(player_id=1)
    agent.load('trained_agent_p1.pkl')
    random_agent = RandomAgent(player_id=2)
    
    state = env.reset()
    print("\nInitial state:")
    env.render()
    
    move_count = 0
    while not env.game_over and move_count < 30:
        valid_actions = env.get_valid_actions()
        
        if env.current_player == 1:
            action = agent.choose_action(state, valid_actions, training=False)
            print(f"\nTrained Agent (X): {action[0].upper()} at ({action[1]}, {action[2]})")
        else:
            action = random_agent.choose_action(state, valid_actions)
            print(f"\nRandom Agent (O): {action[0].upper()} at ({action[1]}, {action[2]})")
        
        state, reward, done, info = env.step(action)
        env.render()
        move_count += 1
    
    print("\n" + "="*60)


def demo_performance_stats():
    """Show performance statistics of trained agent"""
    print("\n" + "="*60)
    print("DEMO: Performance Statistics")
    print("="*60)
    
    if not os.path.exists('trained_agent_p1.pkl'):
        print("No trained agent found. Please run train.py first.")
        return
    
    agent = QLearningAgent(player_id=1)
    agent.load('trained_agent_p1.pkl')
    random_agent = RandomAgent(player_id=2)
    
    print("\nTesting trained agent against random opponent (50 games)...")
    
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(50):
        env = ModifiedTicTacToe(pieces_per_player=4)
        state = env.reset()
        
        while not env.game_over:
            valid_actions = env.get_valid_actions()
            
            if env.current_player == 1:
                action = agent.choose_action(state, valid_actions, training=False)
            else:
                action = random_agent.choose_action(state, valid_actions)
            
            state, reward, done, info = env.step(action)
        
        if env.winner == 1:
            wins += 1
        elif env.winner == 0:
            draws += 1
        else:
            losses += 1
    
    print(f"\nResults over 50 games:")
    print(f"  Wins:   {wins:2d} ({wins/50*100:.1f}%)")
    print(f"  Draws:  {draws:2d} ({draws/50*100:.1f}%)")
    print(f"  Losses: {losses:2d} ({losses/50*100:.1f}%)")
    print(f"\nAgent Q-table contains {len(agent.q_table)} state-action pairs")
    print("="*60)


def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("RL TIC-TAC-TOE DEMONSTRATION")
    print("="*60)
    
    print("\nThis demo will show:")
    print("1. Game rules and mechanics")
    print("2. A trained agent playing")
    print("3. Performance statistics")
    
    input("\nPress Enter to start demo...")
    
    demo_game_rules()
    input("\nPress Enter to continue...")
    
    demo_trained_agent()
    input("\nPress Enter to continue...")
    
    demo_performance_stats()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nTo play the game yourself, run: python play.py")
    print("To train a new agent, run: python train.py")
    print("="*60)


if __name__ == "__main__":
    main()
