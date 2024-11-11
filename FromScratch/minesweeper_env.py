import numpy as np
import random

class MinesweeperEnv:
    def __init__(self, rows, cols, mines):
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.first_click = True  
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.visible_board = np.full((self.rows, self.cols), -1)  
        self.game_over = False
        self.first_click = True 
        return self._get_state()

    def _place_mines(self, safe_r, safe_c):
        mines_placed = 0
        while mines_placed < self.mines:
            r, c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if self.board[r, c] == 0 and not (r == safe_r and c == safe_c):
                self.board[r, c] = -1
                mines_placed += 1
                for i in range(max(0, r-1), min(self.rows, r+2)):
                    for j in range(max(0, c-1), min(self.cols, c+2)):
                        if self.board[i, j] != -1:
                            self.board[i, j] += 1

    def reveal(self, r, c):
        if self.game_over:
            return self._get_state(), -1.0, True

        if self.visible_board[r, c] != -1:
            return self._get_state(), -0.5, self.game_over 

        if self.first_click:
            self._place_mines(r, c)
            self.first_click = False

        if self.board[r, c] == -1:
            self.game_over = True
            return self._get_state(), -1.0, True 

        self._reveal_recursive(r, c)
        done = self.is_win()
        reward = -0.1 if self.board[r, c] == 0 else 0.1
        return self._get_state(), reward, done

    def _reveal_recursive(self, r, c):
        queue = [(r, c)]
        while queue:
            r, c = queue.pop()
            if self.visible_board[r, c] == -1:
                self.visible_board[r, c] = self.board[r, c]
                if self.board[r, c] == 0:
                    for i in range(max(0, r-1), min(self.rows, r+2)):
                        for j in range(max(0, c-1), min(self.cols, c+2)):
                            if self.visible_board[i, j] == -1:
                                queue.append((i, j))

    def is_win(self):
        return np.count_nonzero(self.visible_board == -1) == self.mines

    def _get_state(self):
        return self.visible_board.flatten()  

    def display(self):
        for row in self.visible_board:
            print(' '.join(['*' if cell == -1 else str(cell) for cell in row]))
        print()

    def get_valid_moves(self):
        return (self.visible_board == -1).flatten() 

    def play_turn(self, x, y):
        if self.game_over:
            print("Game over! Please reset to start a new game.")
            return
        _, reward, done = self.reveal(x, y)
        self.display()
        if done:
            if reward < 0:
                print("You hit a mine! Game over!")
            else:
                print("Congratulations! You win!")
            self.game_over = True
        else:
            print(f"Move completed. Reward: {reward:.2f}")