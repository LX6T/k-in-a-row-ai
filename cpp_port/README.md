# C++ port of the K-in-a-row game

This folder contains a lightweight C++ implementation of the main game logic from the Python project.

## Build

From this directory:

```bash
make
```

## Run

Default bot-vs-bot game on a 9x9 board with 5 in a row:

```bash
./k_in_row
```

Human-vs-bot on a 3x3 tic-tac-toe board:

```bash
./k_in_row 3 3 3 1
```

Human-vs-human on a 7x7 board with 5 in a row:

```bash
./k_in_row 7 7 5 2
```

The `players` argument is:

- `0` = bot-vs-bot
- `1` = human-vs-bot
- `2` = human-vs-human
