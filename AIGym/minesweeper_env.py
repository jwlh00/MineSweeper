import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
import random

class MinesweeperEnv(gym.Env):
    def __init__(self, board_size=5, num_mines=2):
        super(MinesweeperEnv, self).__init__()
        self.board_size = board_size
        self.num_mines = num_mines
        self.first_move = True
        self.max_steps = board_size * board_size
        self.terminated = False

        self.action_space = spaces.Discrete(board_size * board_size)
        self.observation_space = spaces.Box(
            low=-2, 
            high=8,
            shape=(board_size * board_size,),
            dtype=np.float32
        )

        self.board = np.zeros((board_size, board_size), dtype=np.float32)
        self.mines = np.zeros((board_size, board_size), dtype=bool)
        self.revealed = np.zeros((board_size, board_size), dtype=bool)
        self.steps = 0
        self.reset()

    def _place_mines(self, safe_r, safe_c):
        mines_placed = 0
        while mines_placed < self.num_mines:
            r, c = random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1)
            if self.board[r, c] == 0 and not (r == safe_r and c == safe_c):
                self.board[r, c] = -2
                mines_placed += 1
                for i in range(max(0, r - 1), min(self.board_size, r + 2)):
                    for j in range(max(0, c - 1), min(self.board_size, c + 2)):
                        if self.board[i, j] != -2:
                            self.board[i, j] += 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.fill(0)
        self.revealed.fill(False)
        self.mines.fill(False)
        self.first_move = True
        self.steps = 0
        self.terminated = False
        return self._get_observation(), {}

    def step(self, action):
        row, col = divmod(action, self.board_size)
        self.steps += 1
        terminated = False
        truncated = False
        reward = 0.0

        if self.first_move:
            self._place_mines(row, col)
            self.first_move = False

        if self.board[row, col] == -2:
            self.terminated = True
            reward = -20
        elif self.revealed[row, col]:
            reward = -5.0
        else:
            reward = 2.5
            self._reveal_recursive(row, col)
            if self.is_win():
                self.terminated = True
                reward += 20

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_observation(), reward, self.terminated, truncated, {}

    def _reveal_recursive(self, r, c):
        queue = [(r, c)]
        while queue:
            r, c = queue.pop()
            if self.revealed[r, c]:
                continue
            self.revealed[r, c] = True
            if self.board[r, c] == 0:
                for i in range(max(0, r - 1), min(self.board_size, r + 2)):
                    for j in range(max(0, c - 1), min(self.board_size, c + 2)):
                        if not self.revealed[i, j]:
                            queue.append((i, j))

    def _get_observation(self):
        return np.where(self.revealed, self.board, -1).flatten().astype(np.float32)

    def is_win(self):
        non_mine_revealed = np.sum(self.revealed & (self.board != -2)) == (self.board_size * self.board_size - self.num_mines)
        return non_mine_revealed

    def render(self):
        """
        Render the board with closed cells shown as '▢', mines as '*', and numbers for revealed cells.
        Reveal all mines if the game has ended (loss).
        """
        display_board = []
        for r in range(self.board_size):
            row_display = []
            for c in range(self.board_size):
                if not self.revealed[r, c]: 
                    if self.terminated and self.board[r, c] == -2:  
                        row_display.append('*')
                    else:
                        row_display.append('▢')
                elif self.board[r, c] == -2:  
                    row_display.append('*')
                else: 
                    row_display.append(str(int(self.board[r, c])))
            display_board.append('\t'.join(row_display))
        print('\n' + '\n'.join(display_board) + '\n')


register(
    id="Minesweeper-v0",
    entry_point="minesweeper_env:MinesweeperEnv",
    kwargs={"board_size": 5, "num_mines": 3}
)
