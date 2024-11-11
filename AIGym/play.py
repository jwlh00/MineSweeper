import gymnasium as gym
from stable_baselines3 import PPO
from minesweeper_env import MinesweeperEnv

model = PPO.load("ppo_minesweeper")  

env = gym.make("Minesweeper-v0", board_size=5, num_mines=4)

def simulate_100_games(env, model):
    wins = 0
    wins = 0

    for game in range(100):
        obs, _ = env.reset()
        done = False
        step_counter = 0  
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            
            step_counter += 1
            if step_counter > 1000:  
                break
            
            if done:
                if reward > 0:
                    wins += 1
        

    win_rate = (wins / 100) * 100
    print(f"Simulated 100 games.")
    print(f"Win rate: {win_rate:.2f}%")

def show_single_game(env, model):
    obs, _ = env.reset()
    done = False
    total_moves = 0

    print("Starting a new game...\n")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        row, col = divmod(action, env.board_size) 
        obs, reward, done, _, _ = env.step(action)
        total_moves += 1

        print(f"Move {total_moves}: AI reveals cell ({row}, {col})")
        env.render()  

    if reward > 0:
        print("\nThe AI won the game!")
    else:
        print("\nThe AI lost the game.")

def main_menu():
    while True:
        print("\nMinesweeper AI Menu")
        print("1. Simulate 100 games to view win rate")
        print("2. Show a single game with AI moves")
        print("3. Exit")
        choice = input("Select an option (1, 2, or 3): ")
        
        if choice == "1":
            simulate_100_games(env, model)
        elif choice == "2":
            show_single_game(env, model)
        elif choice == "3":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

main_menu()
