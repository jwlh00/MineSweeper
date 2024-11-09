import numpy as np
import torch
from agent import DQNAgent
from minesweeper_env import MinesweeperEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

def evaluate_agent(agent, eval_env, episodes=5):
    total_reward = 0
    for _ in range(episodes):
        state = eval_env.reset()
        done = False
        while not done:
            valid_moves = eval_env.get_valid_moves()
            action = agent.act(state, valid_moves, epsilon=0)
            state, reward, done = eval_env.reveal(*action)
            total_reward += reward
    return total_reward / episodes

def train_agent(episodes=1000, board_size=(8, 8), num_mines=10, batch_size=64):
    rows, cols = board_size
    state_size = rows * cols
    action_size = rows * cols
    env = MinesweeperEnv(rows, cols, num_mines)
    eval_env = MinesweeperEnv(rows, cols, num_mines)
    agent = DQNAgent(state_size, action_size)

    best_eval_reward = -float("inf")
    no_improvement_count = 0

    total_rewards = []
    avg_q_values = []
    avg_losses = []

    for episode in tqdm(range(episodes), desc="Training"):
        print(f"Starting Episode {episode}")
        state = env.reset()
        done = False
        total_reward = 0
        episode_q_values = []
        episode_losses = []

        while not done:
            valid_moves = env.get_valid_moves()
            row, col = agent.act(state, valid_moves, epsilon=agent.epsilon)
            next_state, reward, done = env.reveal(row, col)
            agent.memorize(state, row * cols + col, reward, next_state, done)


            loss = agent.replay(batch_size)
            if loss is not None:
                episode_losses.append(loss.item()) 
                print(f"Episode {episode}, Loss: {loss.item()}")

            state = next_state
            total_reward += reward
            q_values = agent.model(torch.FloatTensor(state)).detach().numpy()
            episode_q_values.append(np.mean(q_values))

        print(f"Completed Episode {episode} with Total Reward: {total_reward}")

        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_losses.append(avg_loss)

        total_rewards.append(total_reward)
        avg_q_values.append(np.mean(episode_q_values))

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if episode % 50 == 0:
            agent.update_target_model()

    torch.save(agent.model.state_dict(), "final_model.pth")
    print("Training completed and final model saved!")
    save_training_plots(total_rewards, avg_q_values, avg_losses)

def save_training_plots(total_rewards, avg_q_values, avg_losses):
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Generate a timestamp
    episodes = range(1, len(total_rewards) + 1)

    plt.figure()
    plt.plot(episodes, total_rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards over Episodes")
    plt.legend()
    plt.savefig(f"total_rewards_{timestamp}.png")
    plt.close()

    plt.figure()
    plt.plot(episodes, avg_q_values, label="Average Q-value", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Average Q-value")
    plt.title("Average Q-values over Episodes")
    plt.legend()
    plt.savefig(f"average_q_values_{timestamp}.png")
    plt.close()

    plt.figure()
    plt.plot(episodes, avg_losses, label="Average Loss", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Average Loss over Episodes")
    plt.legend()
    plt.savefig(f"average_losses_{timestamp}.png")
    plt.close()

if __name__ == "__main__":
    train_agent(episodes=10000, board_size=(5, 5), num_mines=2, batch_size=64)