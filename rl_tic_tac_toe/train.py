"""
Training script for Modified Tic-Tac-Toe Q-Learning agents
"""

import numpy as np
from game_env import ModifiedTicTacToe
from agent import QLearningAgent, RandomAgent
import matplotlib.pyplot as plt
from collections import deque


def train_against_random(num_episodes: int = 10000, pieces_per_player: int = 4):
    """
    Train a Q-learning agent by playing against a random opponent.
    
    Args:
        num_episodes: Number of training episodes
        pieces_per_player: Number of pieces per player
    """
    print(f"Training Q-Learning agent against random opponent for {num_episodes} episodes...")
    
    env = ModifiedTicTacToe(pieces_per_player=pieces_per_player)
    agent = QLearningAgent(player_id=1, learning_rate=0.1, discount_factor=0.95, 
                          epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01)
    random_agent = RandomAgent(player_id=2)
    
    wins = 0
    draws = 0
    losses = 0
    
    # Track recent performance
    recent_results = deque(maxlen=100)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Store transitions for agent's learning
        agent_transitions = []
        
        while not done:
            valid_actions = env.get_valid_actions()
            
            if env.current_player == agent.player_id:
                # Agent's turn
                action = agent.choose_action(state, valid_actions, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Store transition for later learning
                agent_transitions.append((state, action, reward, next_state, done))
                episode_reward += reward
                
                state = next_state
            else:
                # Random opponent's turn
                action = random_agent.choose_action(state, valid_actions)
                next_state, reward, done, info = env.step(action)
                
                # Opponent's reward is agent's penalty
                if done and reward != 0:
                    episode_reward -= reward
                
                state = next_state
        
        # Update Q-values for all agent's transitions
        for i, (s, a, r, ns, d) in enumerate(agent_transitions):
            # For intermediate steps, no immediate reward
            step_reward = r if d else 0
            
            # If game ended, assign final reward to last action
            if i == len(agent_transitions) - 1 and d:
                if env.winner == agent.player_id:
                    step_reward = 1.0
                elif env.winner == 0:
                    step_reward = 0.0
                else:
                    step_reward = -1.0
            
            next_valid = env.get_valid_actions() if not d else []
            agent.update_q_value(s, a, step_reward, ns, next_valid, d)
        
        # Track results
        if env.winner == agent.player_id:
            wins += 1
            recent_results.append(1)
        elif env.winner == 0:
            draws += 1
            recent_results.append(0)
        else:
            losses += 1
            recent_results.append(-1)
        
        agent.episode_rewards.append(episode_reward)
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            win_rate = wins / (episode + 1) * 100
            recent_win_rate = sum(1 for r in recent_results if r == 1) / len(recent_results) * 100 if recent_results else 0
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Overall: W={wins} D={draws} L={losses} (Win rate: {win_rate:.1f}%)")
            print(f"  Recent 100: Win rate = {recent_win_rate:.1f}%")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Q-table size: {len(agent.q_table)}")
    
    print("\nTraining complete!")
    print(f"Final stats: W={wins} D={draws} L={losses}")
    print(f"Overall win rate: {wins/num_episodes*100:.1f}%")
    
    # Save agent
    agent.save('trained_agent_p1.pkl')
    
    return agent


def train_self_play(num_episodes: int = 10000, pieces_per_player: int = 4):
    """
    Train two Q-learning agents through self-play.
    
    Args:
        num_episodes: Number of training episodes
        pieces_per_player: Number of pieces per player
    """
    print(f"Training two Q-Learning agents through self-play for {num_episodes} episodes...")
    
    env = ModifiedTicTacToe(pieces_per_player=pieces_per_player)
    agent1 = QLearningAgent(player_id=1, learning_rate=0.1, discount_factor=0.95,
                           epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01)
    agent2 = QLearningAgent(player_id=2, learning_rate=0.1, discount_factor=0.95,
                           epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01)
    
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    # Track recent performance
    recent_results = deque(maxlen=100)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        # Store transitions for both agents
        agent1_transitions = []
        agent2_transitions = []
        
        while not done:
            valid_actions = env.get_valid_actions()
            
            if env.current_player == 1:
                action = agent1.choose_action(state, valid_actions, training=True)
                next_state, reward, done, info = env.step(action)
                agent1_transitions.append((state, action, reward, next_state, done))
            else:
                action = agent2.choose_action(state, valid_actions, training=True)
                next_state, reward, done, info = env.step(action)
                agent2_transitions.append((state, action, reward, next_state, done))
            
            state = next_state
        
        # Update Q-values for both agents
        for i, (s, a, r, ns, d) in enumerate(agent1_transitions):
            step_reward = 0
            if i == len(agent1_transitions) - 1 and d:
                if env.winner == 1:
                    step_reward = 1.0
                elif env.winner == 0:
                    step_reward = 0.0
                else:
                    step_reward = -1.0
            
            next_valid = [] if d else env.get_valid_actions()
            agent1.update_q_value(s, a, step_reward, ns, next_valid, d)
        
        for i, (s, a, r, ns, d) in enumerate(agent2_transitions):
            step_reward = 0
            if i == len(agent2_transitions) - 1 and d:
                if env.winner == 2:
                    step_reward = 1.0
                elif env.winner == 0:
                    step_reward = 0.0
                else:
                    step_reward = -1.0
            
            next_valid = [] if d else env.get_valid_actions()
            agent2.update_q_value(s, a, step_reward, ns, next_valid, d)
        
        # Track results
        if env.winner == 1:
            agent1_wins += 1
            recent_results.append(1)
        elif env.winner == 2:
            agent2_wins += 1
            recent_results.append(2)
        else:
            draws += 1
            recent_results.append(0)
        
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            recent_p1_wins = sum(1 for r in recent_results if r == 1)
            recent_p2_wins = sum(1 for r in recent_results if r == 2)
            recent_draws = sum(1 for r in recent_results if r == 0)
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Overall: P1={agent1_wins} P2={agent2_wins} D={draws}")
            print(f"  Recent 100: P1={recent_p1_wins} P2={recent_p2_wins} D={recent_draws}")
            print(f"  Epsilon: {agent1.epsilon:.4f}")
            print(f"  Q-table sizes: P1={len(agent1.q_table)}, P2={len(agent2.q_table)}")
    
    print("\nTraining complete!")
    print(f"Final stats: P1={agent1_wins} P2={agent2_wins} D={draws}")
    
    # Save both agents
    agent1.save('trained_agent_p1.pkl')
    agent2.save('trained_agent_p2.pkl')
    
    return agent1, agent2


if __name__ == "__main__":
    import sys
    
    print("Modified Tic-Tac-Toe Q-Learning Training")
    print("=" * 50)
    print("\nTraining Options:")
    print("1. Train against random opponent (faster)")
    print("2. Train through self-play (more robust)")
    
    choice = input("\nSelect training mode (1 or 2, default=1): ").strip() or "1"
    
    if choice == "1":
        num_episodes = int(input("Number of episodes (default=10000): ").strip() or "10000")
        pieces = int(input("Pieces per player (3 or 4, default=4): ").strip() or "4")
        train_against_random(num_episodes, pieces)
    elif choice == "2":
        num_episodes = int(input("Number of episodes (default=10000): ").strip() or "10000")
        pieces = int(input("Pieces per player (3 or 4, default=4): ").strip() or "4")
        train_self_play(num_episodes, pieces)
    else:
        print("Invalid choice. Using default: training against random opponent.")
        train_against_random()
