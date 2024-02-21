AI agent built to play k-in-a-row (generalised tic-tac-toe).

How to run:
* `python3 runner.py`: Default is Bot vs Bot, 7x7 5-in-a-row
* `python3 runner.py 3 3 3 1`: Human vs Bot, tic-tac-toe (3x3 3-in-a-row)
* `python3 runner.py 7 9 5 1`: Human vs Bot, 7x9 5-in-a-row
* `python3 runner.py 11 11 6 0 50 1`: Bot vs Bot, 11x11 6-in-a-row, 1.0s time limit, first 50 moves are random

Uses iterative deepening DFS, alpha-beta pruning and Zobrist hashing alongside a robust static evaluation function.

Originally created for as part of an assignment for CSE 415: Introduction to Artificial Intelligence (University of Washington).
