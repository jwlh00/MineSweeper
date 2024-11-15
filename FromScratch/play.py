import torch
from agent import DQNAgent
from minesweeper_env import MinesweeperEnv
import numpy as np

def simulate_games(agent, env, num_games=100):
    wins = 0
    for game in range(num_games):
        state = env.reset()
        done = False
        move_count = 0  

        while not done:
            move_count += 1
            valid_moves = env.get_valid_moves()

            state_tensor = torch.FloatTensor(state)
            row, col = agent.act(state_tensor, valid_moves, epsilon=0)  
            state, reward, done = env.reveal(row, col)

            if done:
                if env.is_win():
                    wins += 1
                break 

    win_rate = wins / num_games * 100
    print(f"\nSimulated {num_games} games. Win rate: {win_rate:.2f}%")
    return win_rate

def play_with_suggestions(agent, env):
    state = env.reset()
    env.display()
    print("\nLet's play! Enter row and column to reveal a cell.")
    
    while not env.game_over:
        valid_moves = env.get_valid_moves()
        state_tensor = torch.FloatTensor(state)
        suggested_row, suggested_col = agent.act(state_tensor, valid_moves, epsilon=0)
        
        print(f"Model suggests: Row {suggested_row}, Column {suggested_col}")

        try:
            x = int(input("Enter Row: "))
            y = int(input("Enter Column: "))

            if x < 0 or x >= env.rows or y < 0 or y >= env.cols:
                print("Invalid move! Please enter coordinates within the board range.")
                continue

            state, _, done = env.reveal(x, y)
            env.display()

            if done:
                if env.is_win():
                    print("Congratulations! You won!")
                else:
                    print("You hit a mine! Game over.")
                break
        except ValueError:
            print("Invalid input! Please enter integers for row and column.")

def main():
    rows, cols, mines = 5, 5, 3
    env = MinesweeperEnv(rows, cols, mines)
    state_size = rows * cols
    action_size = rows * cols

    agent = DQNAgent(state_size, action_size)
    
    try:
        agent.model.load_state_dict(torch.load("Results/Model/150k_110424.pth"))
        agent.model.eval() 
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    while True:
        print("\nMinesweeper AI Menu")
        print("1. Simulate 100 games and show win rate")
        print("2. Play manually with model suggestions")
        print("3. Exit")
        choice = input("Select an option: ")

        if choice == "1":
            simulate_games(agent, env)
        elif choice == "2":
            play_with_suggestions(agent, env)
        elif choice == "3":
            print("Exiting the game.")
            break
        else:
            print("Invalid option. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()