import numpy as np
from enum import Enum

class GameStates(Enum):
    WIN = 1
    LOSE = 2
    IN_PROGRESS = 3

class Board():
    def __init__(self, n, max_tile=2048):
        self.n = n
        self.max_tile = max_tile
        self.board = np.zeros((n, n), dtype=np.int)
        self.curr_state = GameStates.IN_PROGRESS
        self.action_set = set()

        self.remaining_tiles = n**2
        self.largest_tile = 2
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

    @property
    def score(self):
        return self.largest_tile

    @property
    def free_tiles(self):
        return self.remaining_tiles

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
                if self.board[i][j] >= self.max_tile:
                    self.curr_state = GameStates.WIN
                    self.action_set.clear()
                    return

                # User can move tiles to the right if first a digit then a zero are seen when moving left-right in a row
                if self.board[i][j] == 0:
                    h_zeroSeen = True
                    if h_digitSeen: self.action_set.add(self.right)

                # User can move tiles to the left if first a zero then a digit are seen when moving left-right in a row
                if self.board[i][j] != 0:
                    h_digitSeen = True
                    if h_zeroSeen: self.action_set.add(self.left)
                    # If two adjacent horizontal tiles have the same value, either a left or right action can be performed
                    if (j < self.n - 1 and self.board[i][j] == self.board[i][j+1]): self.action_set.update([self.left, self.right])

                # User can move tiles down if first a digit then a zero are seen when moving top-bottom in a column
                if self.board[j][i] == 0:
                    v_zeroSeen = True
                    if v_digitSeen: self.action_set.add(self.down)

                # User can move tiles up if first a zero then a digit are seen when moving top-bottom in a column
                if self.board[j][i] != 0:
                    v_digitSeen = True
                    if v_zeroSeen: self.action_set.add(self.up)
                    # If two adjacent vertical tiles have the same value, either an up or down action can be performed
                    if (j < self.n - 1 and self.board[j][i] == self.board[j+1][i]): self.action_set.update([self.up, self.down])

        self.curr_state = GameStates.LOSE if len(self.action_set) <= 0 else GameStates.IN_PROGRESS

    def add_two(self):
        found = False
        while not found:
            i, j = np.ravel(np.random.choice(len(self.board), (1, 2)))
            found = (self.board[i][j] == 0)
        self.board[i][j] = 2
        self.remaining_tiles -= 1

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
        for i in range(self.n):
            for j in range(self.n - 1):
                if self.board[i][j] == 0 or self.board[i][j] != self.board[i][j + 1]: continue
                self.board[i][j] *= 2
                self.board[i][j + 1] = 0
                self.remaining_tiles += 1
                self.largest_tile = max(self.board[i][j], self.largest_tile)

    def up(self):
        if not self.up in self.action_set: return
        self.board = np.rot90(self.board, axes=(0, 1))
        self.make_move()
        self.board = np.rot90(self.board, axes=(1, 0))
        self.update_action_set()

    def down(self):
        if not self.down in self.action_set: return
        self.board = np.rot90(self.board, axes=(1, 0))
        self.make_move()
        self.board = np.rot90(self.board, axes=(0, 1))
        self.update_action_set()

    def left(self):
        if not self.left in self.action_set: return
        self.make_move()
        self.update_action_set()

    def right(self):
        if not self.right in self.action_set: return
        self.board = np.flip(self.board, axis=1)
        self.make_move()
        self.board = np.flip(self.board, axis=1)
        self.update_action_set()


    def make_move(self):
        self.compress_and_merge()
        self.add_two()
        print('Score: {0}, Remaining tiles: {1}'.format(self.score, self.remaining_tiles))

    def compress_and_merge(self):
        self.compress()
        self.merge()
        self.compress()