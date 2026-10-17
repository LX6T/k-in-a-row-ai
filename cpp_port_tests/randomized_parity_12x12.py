import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import game
import minimax_agent

# ROOT = Path(__file__).resolve().parent.parent
CPP_PORT = ROOT / "cpp_port"
CPP_TESTS = ROOT / "cpp_port_tests"

BOARD_ROWS = 9
BOARD_COLS = 9
K = 5
N_BOARDS = 5


def build_cpp_helper() -> Path:
    binary = CPP_TESTS / "choose_move_cli"
    src = CPP_TESTS / "choose_move_cli.cpp"
    if not binary.exists() or binary.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-pedantic",
                str(src),
                str(CPP_PORT / "MinimaxAgent.cpp"),
                str(CPP_PORT / "GameState.cpp"),
                "-I",
                str(CPP_PORT),
                "-o",
                str(binary),
            ],
            check=True,
            cwd=str(ROOT),
        )
    return binary


def random_legal_board(num_moves: int):
    board = [[game.EMPTY_PIECE for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
    next_player = game.X_PIECE
    positions = [(r, c) for r in range(BOARD_ROWS) for c in range(BOARD_COLS)]
    random.shuffle(positions)

    used = 0
    for r, c in positions:
        if used >= num_moves:
            break
        board[r][c] = next_player
        next_player = game.O_PIECE if next_player == game.X_PIECE else game.X_PIECE
        used += 1

    state = game.GameState(board, next_player, K)
    if state.winner() is not None:
        return random_legal_board(num_moves)
    return state


def python_move(state):
    agent = minimax_agent.MinimaxAgent(state, state.next_player, 15)
    agent.silent = True
    return agent.choose_move(state, None)


def python_static_eval(state):
    agent = minimax_agent.MinimaxAgent(state, state.next_player, 15)
    agent.silent = True
    return agent.static_eval(state)[0]


def cpp_move(state):
    binary = build_cpp_helper()
    rows = ["".join("X" if cell == game.X_PIECE else "O" if cell == game.O_PIECE else "." for cell in row) for row in state.board]
    proc = subprocess.run(
        [str(binary), "--rows", str(BOARD_ROWS), "--cols", str(BOARD_COLS), "--next", state.next_player],
        input="\n".join(rows) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"C++ helper produced no output for board: {rows}")
    x, y = [int(v.strip()) for v in out.split(",")]
    return (x, y)


def cpp_static_eval(state):
    binary = build_cpp_helper()
    rows = ["".join("X" if cell == game.X_PIECE else "O" if cell == game.O_PIECE else "." for cell in row) for row in state.board]
    proc = subprocess.run(
        [str(binary), "--rows", str(BOARD_ROWS), "--cols", str(BOARD_COLS), "--next", state.next_player, "--mode", "static_eval"],
        input="\n".join(rows) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"C++ helper produced no static_eval output for board: {rows}")
    return float(out)


def main():
    matches = 0
    mismatches = 0
    mismatch_examples = []
    for i in range(1, N_BOARDS + 1):
        num_moves = random.randint(0, 50)
        state = random_legal_board(num_moves)
        py_move = python_move(state)
        cpp_move_value = cpp_move(state)
        if py_move == cpp_move_value:
            matches += 1
        else:
            mismatches += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append((i, num_moves, state.next_player, py_move, cpp_move_value, state.board))

    print(f"Matched {matches} out of {N_BOARDS} boards.")
    print(f"Mismatched {mismatches} out of {N_BOARDS} boards.")
    for idx, num_moves, next_player, py_move, cpp_move_value, board in mismatch_examples:
        print(f"Example mismatch #{idx}: moves={num_moves}, next={next_player}, python={py_move}, cpp={cpp_move_value}")
        for row in board:
            print("".join("X" if cell == game.X_PIECE else "O" if cell == game.O_PIECE else "." for cell in row))
        print()
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
