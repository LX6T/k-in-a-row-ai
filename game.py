"""
game.py
author: CSE 415 course staff

This file provides a data type for the game state.
You should not modify this file.
"""

from dataclasses import dataclass
from functools import cached_property
import re

"""
Use these globals below for good programming practices instead of hard coding 'X' or 'O' into your code.
"""
X_PIECE = 'X'
O_PIECE = 'O'
BLOCK_PIECE = '-'
EMPTY_PIECE = ' '


@dataclass
class GameState:
    """
    Data type for the game state. Contains the board, the next player to move, and k (pieces in a row to win).
    Because this is a dataclass, functions such as the constructor are automatically created despite not being shown.
    """
    board: list[list[str]]  # 2d array of board pieces
    next_player: str
    k: int

    @cached_property
    def h(self):
        """
        The height of the board. The @cached_property decorator on this method means that you reference this as if it
        were a property, such as through state.h instead of state.h()
        """
        return len(self.board[0])

    @cached_property
    def w(self):
        """
        The width of the board. The @cached_property decorator on this method means that you reference this as if it
        were a property, such as through state.w instead of state.w()
        """
        return len(self.board)

    def is_valid_move(self, move: (int, int)) -> bool:
        """
        Test for if a move is allowed or not.
        :param move: Tuple of (x,y) coords of the desired move
        :return: True if valid, False if not
        """
        return move[0] < self.w and move[1] < self.h and self.board[move[0]][move[1]] is EMPTY_PIECE

    def make_move(self, move: (int, int)) -> "GameState":
        """
        Applies a move to the game board and returns the new state
        :param move: Tuple of (x,y) coords of the desired move
        :return: new state with the move applied
        """
        assert self.is_valid_move(move)
        nboard = [list(row) for row in self.board]
        nboard[move[0]][move[1]] = self.next_player
        nplayer = X_PIECE if self.next_player is O_PIECE else O_PIECE
        nstate = GameState(nboard, nplayer, self.k)
        return nstate

    def winner(self) -> [str, None]:
        """
        Determines if any agent has won the game.
        :return: token of the winning player, 'draw', or None
        """
        rows = [''.join(row) for row in self.board]
        cols = [''.join(col) for col in list(zip(*self.board))]

        diag_coords = [(i, i) for i in range(min(self.w, self.h))]
        left_diag_coords = [[(c[0], c[1] + i) for c in diag_coords if c[1] + i < self.h] for i in range(1, self.h)]
        right_diag_coords = [[(c[0] + i, c[1]) for c in diag_coords if c[0] + i < self.w] for i in range(1, self.w)]

        diags = [''.join(self.board[c[0]][c[1]] for c in diag_coords)]
        diags.extend([''.join(self.board[c[0]][self.h - 1 - c[1]] for c in diag_coords)])
        diags.extend([''.join(self.board[c[0]][c[1]] for c in coords) for coords in left_diag_coords])
        diags.extend([''.join(self.board[c[0]][self.h - 1 - c[1]] for c in coords) for coords in left_diag_coords])
        diags.extend([''.join(self.board[c[0]][c[1]] for c in coords) for coords in right_diag_coords])
        diags.extend([''.join(self.board[c[0]][self.h - 1 - c[1]] for c in coords) for coords in right_diag_coords])

        rows.extend(cols)
        rows.extend(diags)

        for r in rows:
            if re.search(f'[{X_PIECE}]{{{self.k}}}', r):
                return X_PIECE
            if re.search(f'[{O_PIECE}]{{{self.k}}}', r):
                return O_PIECE

        if not sum(row.count(EMPTY_PIECE) for row in self.board):
            return "draw"

        return None

    @classmethod
    def empty(cls, size: (int, int), k: int, first: str = X_PIECE):
        """
        Creates a new empty board. Because this is a classmethod, call this function by referring to the class instead
        of an instance of the class, such as GameState.empty() instead of state.empty()
        :param size: tuple of dimensions of the board
        :param k: pieces in a row needed to win
        :param first: whose turn it is to start. defaults to X
        :return: new board
        """
        assert k <= max(size)
        nboard = [[EMPTY_PIECE for _ in range(size[1])] for _ in range(size[0])]
        return GameState(nboard, first, k)

    @classmethod
    def tic_tac_toe(cls):
        return cls.empty((3, 3), 3)

    @classmethod
    def no_corners(cls):
        nboard = [['-', ' ', ' ', ' ', ' ', ' ', '-'],
                  [' ', ' ', ' ', ' ', ' ', ' ', ' '],
                  [' ', ' ', ' ', ' ', ' ', ' ', ' '],
                  [' ', ' ', ' ', ' ', ' ', ' ', ' '],
                  [' ', ' ', ' ', ' ', ' ', ' ', ' '],
                  [' ', ' ', ' ', ' ', ' ', ' ', ' '],
                  ['-', ' ', ' ', ' ', ' ', ' ', '-']]
        return GameState(nboard, X_PIECE, 5)

    @classmethod
    def no_corners_small(cls):
        nboard = [['-', ' ', ' ', ' ', '-'],
                  [' ', ' ', ' ', ' ', ' '],
                  [' ', ' ', ' ', ' ', ' '],
                  [' ', ' ', ' ', ' ', ' '],
                  ['-', ' ', ' ', ' ', '-']]
        return GameState(nboard, X_PIECE, 4)

    def __str__(self):
        s = '+--' + 4 * (self.w - 1) * '-' + '-+\n'
        for row in self.board:
            s = s + '| ' + ' | '.join(row) + ' |\n'
        s = s + '+--' + 4 * (self.w - 1) * '-' + '-+\n'
        s = s + self.next_player + " to play next"
        return s

    def __repr__(self):
        return str(self)

    def copy(self):
        return GameState([list(row) for row in self.board], self.next_player, self.k)
