"""
agent.py
author: CSE 415 course staff

Any modifications you make to this file will not be considered during grading, so you probably shouldn't change it.
"""
import game
import random
from typing import final
from threading import Thread


class Agent:
    """
    This is a sample agent that plays completely randomly. Although the minimax and static_eval functions are present,
    they are just stubs that are never called. Use this agent as a sample agent to play against (which you should easily
    beat most of the time, especially on larger boards) and as a rough idea of what your agent might look like.
    """
    piece: str

    def __init__(self, initial_state: game.GameState, piece: str):
        """
        Sets up any state for the agent to keep track of. Note that this is called by any agents that subclass this one
        such as your agent, meaning your agent will have access to the piece property that is set here.
        self._move is a hidden property that is used for the timeout code and which you should not modify.
        :param initial_state: starting state of the board
        :param piece: which piece this agent is playing
        """
        self.piece = piece
        self._move = None

    def introduce(self):
        """
        returns a multi-line introduction string
        :return: intro string
        """
        return ("My name is Random Agent.\n" +
                "I was created by course staff (cse415).\n" +
                "I'm ready to play K-in-a-Row!")

    def nickname(self):
        """
        returns a short nickname for the agent
        :return: nickname
        """
        return "human_agent"

    @final
    def get_move(self, state: game.GameState, time_limit: float = None) -> (int, int):
        """
        Called by the game runner to get your agent's move. This is a final method, meaning it cannot be overriden.
        Handles the time limit, stopping the agent's play if it takes too long. Calls your choose_move method.
        :param state: game state
        :param time_limit: time (in seconds) before you'll be cutoff and forfeit the game
        :return: move to make
        """
        self._move = None
        if time_limit:
            def choose():
                self._move = self.choose_move(state, time_limit)
            t = Thread(target=choose)
            t.start()
            t.join(time_limit)
            if t.is_alive():
                raise TimeoutError
        else:
            self._move = self.choose_move(state, time_limit)
        return self._move

    def choose_move(self, state: game.GameState, time_limit: float) -> (int, int):
        """
        Selects a move to make on the given game board
        Your agent will override this function.
        :param state: current game state
        :param time_limit: time (in seconds) before you'll be cutoff and forfeit the game
        :return: move (x,y), remark
        """

        while True:
            player_input = [int(i) for i in input("Enter your move: ").split()]
            move = (player_input[0], player_input[1])
            if state.is_valid_move(move):
                break
            else:
                print("Invalid move")

        self.print_board(state, move)

        return move

    def minimax(self, state: game.GameState, depth_remaining: int, time_limit: float,
                alpha: float = None, beta: float = None, z_hashing=None) -> ((int, int), float):
        """
        Uses minimax to evaluate the given state and choose the best action from this state. Uses the next_player of the
        given state to decide between min and max. Recursively calls itself to reach depth_remaining layers. Optionally
        uses alpha, beta for pruning, and/or z_hashing for zobrist hashing.
        Your agent will override this function.
        :param state: State to evaluate
        :param depth_remaining: number of layers left to evaluate
        :param time_limit: argument for your use to make sure you return before the time limit. None means no time limit
        :param alpha: alpha value for pruning
        :param beta: beta value for pruning
        :param z_hashing: zobrist hashing data
        :return: move (x,y), state evaluation
        """
        return None, 0.0

    def static_eval(self, state: game.GameState) -> float:
        """
        Evaluates the given state. States good for X should be larger that states good for O.
        Your agent will override this function.
        :param state: state to evaluate
        :return: evaluation of the state
        """
        return 0.0

    def print_board(self, state, best_move):
        """
        Prints the game board and indicates the last move played.
        """
        h = state.h
        w = state.w
        print("   " + " " * (4 * best_move[1]) + " v " + " " * (4 * (w - best_move[1] - 1)) + " ")
        print("  +" + "-" * (4 * best_move[1]) + " ! " + "-" * (4 * (w - best_move[1] - 1)) + "+")
        for i in range(w):
            row_string = "  "
            if i == best_move[0]:
                row_string = ">--"
            for j in range(h):
                if not (j == 0 and i == best_move[0]):
                    row_string += "|"
                if (i, j) == best_move:
                    row_string += "[" + self.piece + "]"
                else:
                    centre_piece = state.board[i][j]
                    row_string += " " + centre_piece + " "
            if i == best_move[0]:
                row_string += "--<"
            else:
                row_string += "|"
            print(row_string)
        print("  +" + "-" * (4 * best_move[1]) + " ! " + "-" * (4 * (w - best_move[1] - 1)) + "+")
        print("   " + " " * (4 * best_move[1]) + " ^ " + " " * (4 * (w - best_move[1] - 1)) + " ")