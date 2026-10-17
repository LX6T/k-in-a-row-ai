#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

import game
import minimax_agent

CPP_DRIVER = Path(__file__).resolve().parents[1] / "cpp_port" / "minimax_driver"


def get_python_move(board_rows, next_player, k):
    state = game.GameState(
        [[cell if cell != "." else game.EMPTY_PIECE for cell in row] for row in board_rows],
        next_player,
        k,
    )
    agent = minimax_agent.MinimaxAgent(state, next_player, 0)
    agent.silent = True
    move = agent.choose_move(state, None)
    return move


def get_cpp_move(board_rows, next_player, k):
    board = ";".join("".join(row) for row in board_rows)
    cmd = [
        str(CPP_DRIVER),
        "--rows", str(len(board_rows)),
        "--cols", str(len(board_rows[0])),
        "--k", str(k),
        "--next", next_player,
        "--board", board,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    row, col = [int(x.strip()) for x in result.stdout.strip().split(",")[:2]]
    return (row, col)


def legal_moves(state):
    return [
        (r, c)
        for r in range(state.w)
        for c in range(state.h)
        if state.is_valid_move((r, c))
    ]


def immediate_win_moves(state, player):
    wins = []
    for move in legal_moves(state):
        next_state = state.make_move(move)
        if next_state.winner() == player:
            wins.append(move)
    return wins


CASES = [
    {
        "name": "empty board legal move",
        "board": [[".", ".", "."], [".", ".", "."], [".", ".", "."]],
        "next": "X",
        "k": 3,
    },
    {
        "name": "immediate win for X",
        "board": [["X", "X", "."], ["O", ".", "O"], [".", ".", "."]],
        "next": "X",
        "k": 3,
    },
    {
        "name": "immediate block for X",
        "board": [["O", "O", "."], ["X", ".", "."], [".", ".", "."]],
        "next": "X",
        "k": 3,
    },
    {
        "name": "center prefers legal move",
        "board": [[".", ".", "."], [".", ".", "."], [".", ".", "."]],
        "next": "O",
        "k": 3,
    },
]


def main():
    for case in CASES:
        rows = [[cell if cell != "." else game.EMPTY_PIECE for cell in row] for row in case["board"]]
        state = game.GameState(rows, case["next"], case["k"])
        python_move = get_python_move(case["board"], case["next"], case["k"])
        cpp_move = get_cpp_move(case["board"], case["next"], case["k"])

        assert state.is_valid_move(python_move), f"Python move invalid in {case['name']}: {python_move}"
        assert state.is_valid_move(cpp_move), f"C++ move invalid in {case['name']}: {cpp_move}"

        winning_moves = immediate_win_moves(state, case["next"])
        if winning_moves:
            assert cpp_move in winning_moves or python_move in winning_moves, (
                f"Neither side chose an immediate win in {case['name']}: python={python_move}, cpp={cpp_move}, winning={winning_moves}"
            )

        print(f"PASS: {case['name']} -> python={python_move}, cpp={cpp_move}")

    print("All parity tests passed.")


if __name__ == "__main__":
    if not CPP_DRIVER.exists():
        raise SystemExit(f"Missing C++ parity driver at {CPP_DRIVER}. Run 'make' in the cpp_port folder first.")
    main()
