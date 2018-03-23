import numpy as np
from enum import Enum

class GameStates(Enum):
    WIN = 1
    LOSE = 2
    IN_PROGRESS = 3

class Board():
    def __init__(self, n):
        self.n = n
        self.board = np.zeros((n, n), dtype=np.int)
        self.add_two()
        self.add_two()

    def __getitem__(self, item):
        return self.board[item]

    @property
    def game_state(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == 2048: return GameStates.WIN
                if self.board[i][j] == 0 \
                        or (i < self.n - 1 and self.board[i][j] == self.board[i + 1][j]) \
                        or (j < self.n - 1 and self.board[i][j] == self.board[i][j + 1]):
                    return GameStates.IN_PROGRESS
        return GameStates.LOSE

    def add_two(self):
        found = False
        while not found:
            i, j = np.ravel(np.random.choice(len(self.board), (1, 2)))
            found = (self.board[i][j] == 0)
        self.board[i][j] = 2

    def compress(self):
        change_flag = False
        for i in range(self.n):
            newindex = -1
            for j in range(self.n):
                if newindex == -1:
                    if self.board[i][j] == 0: newindex = j
                    continue
                if self.board[i][j] != 0:
                    self.board[i][newindex] = self.board[i][j]
                    self.board[i][j] = 0
                    newindex = j
                    change_flag = True
        return change_flag

    def merge(self):
        change_flag = False
        for i in range(self.n):
            for j in range(self.n - 1):
                if self.board[i][j] == 0 or self.board[i][j] != self.board[i][j + 1]: continue
                self.board[i][j] *= 2
                self.board[i][j + 1] = 0
                change_flag = True
        return change_flag

    def up(self):
        print("up")
        self.board = np.rot90(self.board, axes=(0, 1))
        change_flag = self.compress_and_merge()
        self.board = np.rot90(self.board, axes=(1, 0))
        return change_flag

    def down(self):
        print("down")
        self.board = np.rot90(self.board, axes=(1, 0))
        change_flag = self.compress_and_merge()
        self.board = np.rot90(self.board, axes=(0, 1))
        return change_flag

    def left(self):
        print("left")
        return self.compress_and_merge()

    def right(self):
        print("right")
        self.board = np.flip(self.board, axis=1)
        change_flag = self.compress_and_merge()
        self.board = np.flip(self.board, axis=1)
        return change_flag

    def compress_and_merge(self):
        compressdone = self.compress()
        mergedone = self.merge()
        _ = self.compress()
        return compressdone or mergedone