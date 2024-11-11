import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import minesweeper_env

# Custom callback to collect rewards and episode lengths for plotting
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # This function is called at each training step
        # Collect episode rewards and lengths at the end of each episode
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.locals["infos"][0]["episode"]["r"])
            self.episode_lengths.append(self.locals["infos"][0]["episode"]["l"])
        return True

# Set up the environment
env = gym.make("Minesweeper-v0", board_size=5, num_mines=3)

# Initialize the PPO model with the environment and desired policy
model = PPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=64, gae_lambda=0.95)

# Set up callback and progress bar for training
callback = RewardLoggerCallback()
timesteps = 2000000

with tqdm(total=timesteps) as pbar:
    # Custom callback to update the progress bar
    class ProgressCallback(BaseCallback):
        def __init__(self, pbar, verbose=0):
            super(ProgressCallback, self).__init__(verbose)
            self.pbar = pbar

        def _on_step(self) -> bool:
            # Increment progress bar by n_steps each time
            self.pbar.update(self.locals["n_steps"])
            return True

    # Train the model with the progress bar and reward logger callback
    model.learn(total_timesteps=timesteps, callback=[callback, ProgressCallback(pbar)])

# Save the trained model
model.save("ppo_minesweeper3")

print("Training complete. Model saved as 'ppo_minesweeper.zip'")

# Plot results after training
rolling_length = 100
rewards_moving_average = np.convolve(callback.episode_rewards, np.ones(rolling_length), mode="valid") / rolling_length
lengths_moving_average = np.convolve(callback.episode_lengths, np.ones(rolling_length), mode="valid") / rolling_length

fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

# Plot average rewards per episode
axs[0].set_title("Average Rewards per Episode")
axs[0].plot(rewards_moving_average)
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

# Plot average episode lengths
axs[1].set_title("Average Episode Lengths")
axs[1].plot(lengths_moving_average)
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Episode Length")

plt.tight_layout()
plt.savefig("results3.png")
plt.show()
