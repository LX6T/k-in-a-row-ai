"""
autograder.py
author: CSE 415 course staff

This is the student autograder. You can run this file to verify that your agent works correctly. Note that we may have
additional tests on gradescope that are not present here, so passing all tests here does not guarantee full marks on the
assignment.

You may wish to add more tests to this file to make the testing more complete. You probably shouldn't modify existing
tests, though. Any modifications you make to this file will not be graded.

You can also run individual tests from this file. On PyCharm, you can click the green arrow next to a specific test. On
other platforms, refer to https://docs.python.org/3/library/unittest.html#command-line-interface for info on that.
"""

import math
import numbers

import agent
import game
import minimax_agent

import unittest

import runner


class TestAgent(minimax_agent.MinimaxAgent):
    """
    Subclass of your agent, allowing us to test how many times your static eval function is called.
    """
    sef_calls: int

    def __init__(self, initial_state: game.GameState = game.GameState.tic_tac_toe(), piece: str = 'X'):
        super().__init__(initial_state, piece)
        self.sef_calls = 0

    def static_eval(self, state: game.GameState) -> float:
        self.sef_calls += 1
        return super().static_eval(state)


class ConstructorTest(unittest.TestCase):
    def test_constructor(self):
        s = game.GameState.tic_tac_toe()
        a = minimax_agent.MinimaxAgent(s, game.X_PIECE)
        self.assertIsInstance(a, agent.Agent, "MinimaxAgent must inherit from Agent")


class IntroduceTest(unittest.TestCase):
    def test_simple(self):
        a = TestAgent()
        i = a.introduce()
        self.assertIsInstance(i, str, "Introduce must return a string")
        self.assertNotEqual(i, "", "Introduce must return a non-empty string")
        self.assertTrue('\n' in i, "Introduce must return a multi-line string")


class NicknameTest(unittest.TestCase):
    def test_simple(self):
        a = TestAgent()
        n = a.nickname()
        self.assertIsInstance(n, str, "Nickname must return a string")
        self.assertNotEqual(n, "", "Nickname must return a non-empty string")


class ChooseMoveTest(unittest.TestCase):
    def setUp(self):
        self.s = game.GameState.tic_tac_toe()
        self.a = TestAgent(initial_state=self.s, piece=game.X_PIECE)

    def test_simple(self):
        move = self.a.get_move(self.s)
        self.assertTrue(self.s.is_valid_move(move))


class MinimaxTest(unittest.TestCase):
    def setUp(self):
        self.s = game.GameState.tic_tac_toe()
        self.a = TestAgent(initial_state=self.s, piece=game.X_PIECE)

    def test_simple(self):
        for d in range(1, 5):  # change test depth on tic tac toe board here
            try:
                move, val = self.a.minimax(self.s, depth_remaining=d)
            except:
                self.fail(f"Minimax failed at depth {d}")
            self.assertIsInstance(move, tuple, "Minimax must return a move as a tuple")
            self.assertIsInstance(move[0], int, "Minimax must return a move as a tuple of ints")
            self.assertIsInstance(move[1], int, "Minimax must return a move as a tuple of ints")
            self.assertIsInstance(val, numbers.Number, "Minimax returned an evaluation that is not a number")
            self.assertTrue(self.s.is_valid_move(move), "Minimax returned an invalid move")

    def test_sef_count(self):
        for d in range(2, 5):  # change test depth on tic tac toe board here
            self.a.sef_calls = 0
            move, val = self.a.minimax(self.s, depth_remaining=d)
            self.assertTrue(self.s.is_valid_move(move), "Minimax returned an invalid move")
            self.assertLessEqual(self.a.sef_calls, math.factorial(9) // math.factorial(9-d))


class StaticEvalTest(unittest.TestCase):
    def setUp(self):
        self.s = game.GameState.tic_tac_toe()
        self.a = TestAgent(initial_state=self.s, piece=game.X_PIECE)

    def test_simple(self):
        val = self.a.static_eval(self.s)
        self.assertIsInstance(val, numbers.Number)

    def test_comparison(self):
        all_x = [[game.X_PIECE for _ in range(3)] for _ in range(3)]
        x_val = self.a.static_eval(game.GameState(all_x, game.X_PIECE, 3))
        e_val = self.a.static_eval(self.s)
        self.assertGreater(x_val, e_val, "Board of all Xs should have greater value than empty board")


class FullGameTest(unittest.TestCase):
    def test_7x7(self):
        wins = 0
        for _ in range(5):
            s = game.GameState.empty((7, 7), 5)
            a1 = TestAgent(s, game.X_PIECE)
            a2 = agent.Agent(s, game.O_PIECE)
            r = runner.GameRunner(a1, a2)

            if r.run_game(s, silent=True) == game.X_PIECE:
                wins += 1
        self.assertGreaterEqual(wins, 3)


if __name__ == '__main__':
    unittest.main()
