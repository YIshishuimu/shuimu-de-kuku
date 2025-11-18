"""
Quick start example for RL Tic-Tac-Toe

This script demonstrates the basic usage of the RL Tic-Tac-Toe package.
"""

from game_env import ModifiedTicTacToe
from agent import QLearningAgent, RandomAgent


def example_1_basic_game():
    """Example 1: Play a basic game between two random agents"""
    print("\n" + "="*60)
    print("Example 1: Two Random Agents Playing")
    print("="*60)
    
    env = ModifiedTicTacToe(pieces_per_player=4)
    agent1 = RandomAgent(player_id=1)
    agent2 = RandomAgent(player_id=2)
    
    state = env.reset()
    env.render()
    
    while not env.game_over:
        valid_actions = env.get_valid_actions()
        
        if env.current_player == 1:
            action = agent1.choose_action(state, valid_actions)
            print(f"\nAgent 1 (X): {action[0]} at ({action[1]}, {action[2]})")
        else:
            action = agent2.choose_action(state, valid_actions)
            print(f"\nAgent 2 (O): {action[0]} at ({action[1]}, {action[2]})")
        
        state, reward, done, info = env.step(action)
        env.render()


def example_2_train_agent():
    """Example 2: Train a Q-learning agent"""
    print("\n" + "="*60)
    print("Example 2: Training a Q-Learning Agent")
    print("="*60)
    
    print("\nTraining for 500 episodes (this will take a moment)...")
    
    env = ModifiedTicTacToe(pieces_per_player=4)
    agent = QLearningAgent(player_id=1, learning_rate=0.1, epsilon=1.0)
    random_opponent = RandomAgent(player_id=2)
    
    wins = 0
    
    for episode in range(500):
        state = env.reset()
        agent_transitions = []
        
        while not env.game_over:
            valid_actions = env.get_valid_actions()
            
            if env.current_player == 1:
                action = agent.choose_action(state, valid_actions, training=True)
                next_state, reward, done, info = env.step(action)
                agent_transitions.append((state, action, reward, next_state, done))
                state = next_state
            else:
                action = random_opponent.choose_action(state, valid_actions)
                state, reward, done, info = env.step(action)
        
        # Update Q-values
        for i, (s, a, r, ns, d) in enumerate(agent_transitions):
            step_reward = 0
            if i == len(agent_transitions) - 1 and d:
                if env.winner == 1:
                    step_reward = 1.0
                    wins += 1
                elif env.winner == 0:
                    step_reward = 0.0
            
            next_valid = [] if d else env.get_valid_actions()
            agent.update_q_value(s, a, step_reward, ns, next_valid, d)
        
        agent.decay_epsilon()
    
    print(f"\nTraining complete!")
    print(f"Win rate: {wins/500*100:.1f}%")
    print(f"Q-table size: {len(agent.q_table)} state-action pairs")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    
    return agent


def example_3_use_trained_agent(agent):
    """Example 3: Use a trained agent to play"""
    print("\n" + "="*60)
    print("Example 3: Trained Agent Playing")
    print("="*60)
    
    env = ModifiedTicTacToe(pieces_per_player=4)
    random_opponent = RandomAgent(player_id=2)
    
    print("\nWatching trained agent play one game...")
    
    state = env.reset()
    env.render()
    
    while not env.game_over:
        valid_actions = env.get_valid_actions()
        
        if env.current_player == 1:
            action = agent.choose_action(state, valid_actions, training=False)
            print(f"\nTrained Agent (X): {action[0]} at ({action[1]}, {action[2]})")
        else:
            action = random_opponent.choose_action(state, valid_actions)
            print(f"\nRandom Opponent (O): {action[0]} at ({action[1]}, {action[2]})")
        
        state, reward, done, info = env.step(action)
        env.render()


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("RL TIC-TAC-TOE - QUICK START EXAMPLES")
    print("="*60)
    
    # Example 1: Basic game
    example_1_basic_game()
    input("\nPress Enter to continue to Example 2...")
    
    # Example 2: Train agent
    trained_agent = example_2_train_agent()
    input("\nPress Enter to continue to Example 3...")
    
    # Example 3: Use trained agent
    example_3_use_trained_agent(trained_agent)
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("- Run 'python train.py' for full training")
    print("- Run 'python play.py' to play against trained agent")
    print("- Run 'python demo.py' for complete demonstration")
    print("="*60)


if __name__ == "__main__":
    main()
