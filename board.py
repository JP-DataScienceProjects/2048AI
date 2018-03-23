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
        self.curr_state = GameStates.IN_PROGRESS
        self.action_set = set()
        self.add_two()
        self.add_two()
        self.update_action_set()

    def __getitem__(self, item):
        return self.board[item]

    @property
    def game_state(self):
        return self.curr_state

    @property
    def actions(self):
        return self.action_set


    def update_action_set(self):
        """
        Updates the set of available actions that can be taken on this board
        This function iterates over the matrix only once but checks both rows and columns
        for available actions simultaneously by interchanging the indices i,j (exploits the
        fact that the board is always square)
        """
        self.action_set.clear()

        for i in range(self.n):
            h_zeroSeen, v_zeroSeen, v_digitSeen, h_digitSeen = False, False, False, False

            for j in range(self.n):
                if self.board[i][j] == 2048:
                    self.curr_state = GameStates.WIN
                    self.action_set.clear()
                    return

                if self.board[i][j] == 0:
                    h_zeroSeen = True
                    if h_digitSeen: self.action_set.add(self.right)

                if self.board[i][j] != 0:
                    h_digitSeen = True
                    if h_zeroSeen: self.action_set.add(self.left)

                if self.board[j][i] == 0:
                    v_zeroSeen = True
                    if v_digitSeen: self.action_set.add(self.down)

                if self.board[j][i] != 0:
                    v_digitSeen = True
                    if v_zeroSeen: self.action_set.add(self.up)

        self.curr_state = GameStates.LOSE if len(self.action_set) <= 0 else GameStates.IN_PROGRESS

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
        if not self.up in self.action_set: return
        print("up")
        self.board = np.rot90(self.board, axes=(0, 1))
        self.make_move()
        self.board = np.rot90(self.board, axes=(1, 0))
        self.update_action_set()

    def down(self):
        if not self.down in self.action_set: return
        print("down")
        self.board = np.rot90(self.board, axes=(1, 0))
        self.make_move()
        self.board = np.rot90(self.board, axes=(0, 1))
        self.update_action_set()

    def left(self):
        if not self.left in self.action_set: return
        print("left")
        self.make_move()
        self.update_action_set()

    def right(self):
        if not self.right in self.action_set: return
        print("right")
        self.board = np.flip(self.board, axis=1)
        self.make_move()
        self.board = np.flip(self.board, axis=1)
        self.update_action_set()


    def make_move(self):
        self.compress_and_merge()
        self.add_two()

    def compress_and_merge(self):
        self.compress()
        self.merge()
        self.compress()