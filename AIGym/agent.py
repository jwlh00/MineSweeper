import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import minesweeper_env

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.locals["infos"][0]["episode"]["r"])
            self.episode_lengths.append(self.locals["infos"][0]["episode"]["l"])
        return True

env = gym.make("Minesweeper-v0", board_size=5, num_mines=3)

model = PPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=64, gae_lambda=0.95)

callback = RewardLoggerCallback()
timesteps = 2000000

with tqdm(total=timesteps) as pbar:
    class ProgressCallback(BaseCallback):
        def __init__(self, pbar, verbose=0):
            super(ProgressCallback, self).__init__(verbose)
            self.pbar = pbar

        def _on_step(self) -> bool:
            self.pbar.update(self.locals["n_steps"])
            return True

    model.learn(total_timesteps=timesteps, callback=[callback, ProgressCallback(pbar)])

model.save("ppo_minesweeper3")

print("Training complete. Model saved as 'ppo_minesweeper.zip'")

rolling_length = 100
rewards_moving_average = np.convolve(callback.episode_rewards, np.ones(rolling_length), mode="valid") / rolling_length
lengths_moving_average = np.convolve(callback.episode_lengths, np.ones(rolling_length), mode="valid") / rolling_length

fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

axs[0].set_title("Average Rewards per Episode")
axs[0].plot(rewards_moving_average)
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

axs[1].set_title("Average Episode Lengths")
axs[1].plot(lengths_moving_average)
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Episode Length")

plt.tight_layout()
plt.savefig("results3.png")
plt.show()
