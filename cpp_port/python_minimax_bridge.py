#!/usr/bin/env python3

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import game
import minimax_agent


def main() -> None:
    board = json.loads(os.environ.get("BRIDGE_BOARD", "[]"))
    next_player = os.environ.get("BRIDGE_NEXT_PLAYER", game.X_PIECE)
    k = int(os.environ.get("BRIDGE_K", "5"))
    init_ff = int(os.environ.get("BRIDGE_INIT_FF", "15"))

    if len(sys.argv) >= 2:
        candidate = sys.argv[1]
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                board = json.loads(f.read())
        else:
            board = json.loads(candidate)
    if len(sys.argv) >= 3:
        next_player = sys.argv[2]
    if len(sys.argv) >= 4:
        k = int(sys.argv[3])
    if len(sys.argv) >= 5:
        init_ff = int(sys.argv[4])

    state = game.GameState(board, next_player, k)
    agent = minimax_agent.MinimaxAgent(state, next_player, init_ff)
    agent.silent = True
    move = agent.get_move(state, None)
    print(f"{move[0]},{move[1]}")


if __name__ == "__main__":
    main()
