"""
Play against trained agent or watch AI vs AI gameplay
"""

from game_env import ModifiedTicTacToe
from agent import QLearningAgent, RandomAgent
import os


def get_human_action(env: ModifiedTicTacToe):
    """Get action from human player"""
    valid_actions = env.get_valid_actions()
    
    print("\nValid actions:")
    for idx, (action_type, row, col) in enumerate(valid_actions):
        print(f"{idx}: {action_type.upper()} at ({row}, {col})")
    
    while True:
        try:
            choice = int(input("\nSelect action number: "))
            if 0 <= choice < len(valid_actions):
                return valid_actions[choice]
            else:
                print(f"Please enter a number between 0 and {len(valid_actions)-1}")
        except (ValueError, EOFError):
            print("Invalid input. Please enter a number.")


def human_vs_ai(pieces_per_player: int = 4, human_first: bool = True):
    """Play against trained AI agent"""
    
    print("\n" + "="*50)
    print("HUMAN vs AI")
    print("="*50)
    
    env = ModifiedTicTacToe(pieces_per_player=pieces_per_player)
    
    # Load trained agent
    if human_first:
        ai_player = 2
        agent = QLearningAgent(player_id=2)
    else:
        ai_player = 1
        agent = QLearningAgent(player_id=1)
    
    agent_file = f'trained_agent_p{ai_player}.pkl'
    if not os.path.exists(agent_file):
        agent_file = 'trained_agent_p1.pkl'
    
    try:
        agent.load(agent_file)
    except:
        print("Warning: Could not load trained agent. AI will use random policy.")
    
    state = env.reset()
    env.render()
    
    while not env.game_over:
        if env.current_player == ai_player:
            # AI's turn
            print(f"\nAI (Player {ai_player}) is thinking...")
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, valid_actions, training=False)
            action_type, row, col = action
            print(f"AI chose: {action_type.upper()} at ({row}, {col})")
            state, reward, done, info = env.step(action)
        else:
            # Human's turn
            print(f"\nYour turn (Player {3-ai_player})!")
            action = get_human_action(env)
            action_type, row, col = action
            print(f"You chose: {action_type.upper()} at ({row}, {col})")
            state, reward, done, info = env.step(action)
        
        env.render()
    
    print("\n" + "="*50)
    if env.winner == 0:
        print("Game ended in a DRAW!")
    elif env.winner == ai_player:
        print("AI WINS!")
    else:
        print("HUMAN WINS! Congratulations!")
    print("="*50)


def ai_vs_ai(pieces_per_player: int = 4, num_games: int = 1):
    """Watch two AI agents play against each other"""
    
    print("\n" + "="*50)
    print("AI vs AI")
    print("="*50)
    
    # Load agents
    agent1 = QLearningAgent(player_id=1)
    agent2 = QLearningAgent(player_id=2)
    
    try:
        agent1.load('trained_agent_p1.pkl')
    except:
        print("Warning: Could not load agent 1. Using random policy.")
    
    try:
        agent2.load('trained_agent_p2.pkl')
    except:
        print("Warning: Could not load agent 2. Using agent 1 policy or random.")
        # If agent 2 doesn't exist, use agent 1's Q-table
        agent2 = QLearningAgent(player_id=2)
        try:
            agent2.load('trained_agent_p1.pkl')
        except:
            pass
    
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for game in range(num_games):
        if num_games > 1:
            print(f"\n--- Game {game + 1}/{num_games} ---")
        
        env = ModifiedTicTacToe(pieces_per_player=pieces_per_player)
        state = env.reset()
        
        if num_games == 1:
            env.render()
        
        while not env.game_over:
            valid_actions = env.get_valid_actions()
            
            if env.current_player == 1:
                action = agent1.choose_action(state, valid_actions, training=False)
                if num_games == 1:
                    action_type, row, col = action
                    print(f"\nAgent 1 (X): {action_type.upper()} at ({row}, {col})")
            else:
                action = agent2.choose_action(state, valid_actions, training=False)
                if num_games == 1:
                    action_type, row, col = action
                    print(f"\nAgent 2 (O): {action_type.upper()} at ({row}, {col})")
            
            state, reward, done, info = env.step(action)
            
            if num_games == 1:
                env.render()
        
        # Track results
        if env.winner == 1:
            agent1_wins += 1
        elif env.winner == 2:
            agent2_wins += 1
        else:
            draws += 1
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Agent 1 (X) wins: {agent1_wins}")
    print(f"Agent 2 (O) wins: {agent2_wins}")
    print(f"Draws: {draws}")
    print("="*50)


def ai_vs_random(pieces_per_player: int = 4, num_games: int = 100):
    """Test trained AI against random opponent"""
    
    print("\n" + "="*50)
    print("AI vs Random Opponent")
    print("="*50)
    
    agent = QLearningAgent(player_id=1)
    try:
        agent.load('trained_agent_p1.pkl')
    except:
        print("Error: Could not load trained agent.")
        return
    
    random_agent = RandomAgent(player_id=2)
    
    wins = 0
    draws = 0
    losses = 0
    
    for game in range(num_games):
        env = ModifiedTicTacToe(pieces_per_player=pieces_per_player)
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
        
        if (game + 1) % 10 == 0:
            print(f"Progress: {game + 1}/{num_games} games")
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"AI wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print(f"Random wins: {losses} ({losses/num_games*100:.1f}%)")
    print("="*50)


def main():
    """Main menu"""
    print("\n" + "="*60)
    print("Modified Tic-Tac-Toe - Play with Trained Agent")
    print("="*60)
    print("\nGame Rules:")
    print("- 3x3 board")
    print("- Win by completing a COLUMN (not row)")
    print("- Each player has 3-4 pieces")
    print("- Can remove opponent's piece (max 2 times)")
    print("- Removed pieces return to reserve")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Play against AI (Human first)")
        print("2. Play against AI (AI first)")
        print("3. Watch AI vs AI (single game)")
        print("4. Watch AI vs AI (multiple games)")
        print("5. Test AI vs Random opponent")
        print("6. Exit")
        
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "6":
                print("Thanks for playing!")
                break
            
            # Get pieces per player
            pieces_input = input("Pieces per player (3 or 4, default=4): ").strip()
            pieces = int(pieces_input) if pieces_input else 4
            pieces = max(3, min(4, pieces))
            
            if choice == "1":
                human_vs_ai(pieces_per_player=pieces, human_first=True)
            elif choice == "2":
                human_vs_ai(pieces_per_player=pieces, human_first=False)
            elif choice == "3":
                ai_vs_ai(pieces_per_player=pieces, num_games=1)
            elif choice == "4":
                num_games_input = input("Number of games (default=10): ").strip()
                num_games = int(num_games_input) if num_games_input else 10
                ai_vs_ai(pieces_per_player=pieces, num_games=num_games)
            elif choice == "5":
                num_games_input = input("Number of games (default=100): ").strip()
                num_games = int(num_games_input) if num_games_input else 100
                ai_vs_random(pieces_per_player=pieces, num_games=num_games)
            else:
                print("Invalid choice. Please select 1-6.")
        
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
