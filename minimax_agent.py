"""
minimax_agent.py
author: <YOUR NAME(s) HERE>
"""
from math import log10
import agent
import game
import time
import random


class MinimaxAgent(agent.Agent):
    def __init__(self, initial_state: game.GameState, piece: str):
        super().__init__(initial_state, piece)
        self.eval_calls = 0
        self.wrapup_time = 0.1
        self.silent = False

    def introduce(self):
        """
        returns a multi-line introduction string
        :return: intro string
        """
        return ("My name is Alpha Agent.\n" +
                "I was created by Alex Pullen and Ashley Fenton.\n" +
                "I'm ready to win K-in-a-Row.")

    def nickname(self):
        """
        returns a short nickname for the agent
        :return: nickname
        """
        return "alpha_agent"

    def choose_move(self, state: game.GameState, time_limit: float) -> (int, int):
        """
        Selects a move to make on the given game board. Returns a move
        :param state: current game state
        :param time_limit: time (in seconds) before you'll be cutoff and forfeit the game
        :return: move (x,y)
        """

        self.eval_calls = 0
        h = state.h
        w = state.w
        k = state.k

        """Default best move is first available empty space"""
        best_move = None
        for i in range(w):
            for j in range(h):
                if best_move is not None and state.board[i][j] == game.EMPTY_PIECE:
                    best_move = (i, j)
                    break

        """Number of empty spaces = max remaining moves = max search depth"""
        max_depth = 0
        for i in range(w):
            for j in range(h):
                if state.board[i][j] == game.EMPTY_PIECE:
                    max_depth += 1

        """Uncomment to limit the maximum search depth"""
        # max_depth = min(max_depth, 3)

        """Perform iterative deepening search until depth limit or time limit reached"""
        timeout = time.perf_counter() + time_limit if time_limit is not None else None
        depth = 1
        while depth <= max_depth:

            """Initialise Zobrist hash table"""
            z_table = [[random.getrandbits(32) for pieces in range(2)] for squares in range(h * w)]
            z_hashing = (z_table, dict(), 0)

            """Search for best value at current depth"""
            latest_time_limit = timeout - time.perf_counter() if timeout is not None else None
            move, value = self.minimax(state, depth, latest_time_limit, float("-inf"), float("inf"), z_hashing)

            if time_limit is None or time.perf_counter() < timeout - self.wrapup_time:

                """Full search complete, update best_move"""
                best_move = move
                best_value = value
                if not self.silent: print(f"depth={depth}, best_move={best_move}, best_value={best_value}")

                """Guaranteed to win, stop search"""
                if (self.piece == game.X_PIECE and best_value >= 10 ** k or
                        self.piece == game.O_PIECE and best_value <= -10 ** k):
                    if not self.silent: print(f"Win found in {round(k + 20 - log10(abs(best_value))) + depth} moves")
                    break
                elif (self.piece == game.X_PIECE and best_value <= -10 ** k or
                      self.piece == game.O_PIECE and best_value >= 10 ** k):
                    if not self.silent: print(f"Loss found in {round(k + 20 - log10(abs(best_value))) + depth} moves")
                    break

                """Search again, one layer deeper"""
                depth += 1

            else:
                """Time limit reached, exit search"""
                break

        """Report remaining time"""
        if timeout is not None:
            if not self.silent: print(
                f"Exited {round(timeout - time.perf_counter(), 4)} seconds remaining before timeout")

        """Report total number of static evaluations made"""
        if not self.silent: print(f"Called static_eval() {self.eval_calls} times")

        return best_move

    def minimax(self, state: game.GameState, depth_remaining: int, time_limit: float = None,
                alpha: float = None, beta: float = None, z_hashing=None) -> ((int, int), float):
        """
        Uses minimax to evaluate the given state and choose the best action from this state. Uses the next_player of the
        given state to decide between min and max. Recursively calls itself to reach depth_remaining layers. Optionally
        uses alpha, beta for pruning, and/or z_hashing for zobrist hashing.
        :param state: State to evaluate
        :param depth_remaining: number of layers left to evaluate
        :param time_limit: argument for your use to make sure you return before the time limit. None means no time limit
        :param alpha: alpha value for pruning
        :param beta: beta value for pruning
        :param z_hashing: zobrist hashing data
        :return: move (x,y) or None, state evaluation
        """

        h = state.h
        w = state.w
        A_PIECE = state.next_player

        """Generate Zobrist hash for the current board state"""
        if z_hashing is not None:
            (z_table, z_memory, z_key) = z_hashing

        if time_limit is not None and time_limit < self.wrapup_time:
            """Exit early if reached time limit"""
            return None, None
        elif depth_remaining == 0:
            """Return static evaluation if reached depth limit"""
            value = self.static_eval(state)
            if z_hashing is not None:
                z_memory[z_key] = (None, value)
            return None, value
        else:
            """Otherwise do minimax"""

            timeout = time.perf_counter() + time_limit if time_limit is not None else None
            best_move = None
            best_value = float("-inf") if A_PIECE == game.X_PIECE else float("inf")

            """Default best move is first available empty space"""
            for i in range(w):
                for j in range(h):
                    if state.board[i][j] == game.EMPTY_PIECE:
                        best_move = (i, j)
                        i = w - 1
                        break

            for i in range(w):
                for j in range(h):
                    """Iterate until all spaces have been tried, exit early if time limit is reached"""
                    if timeout is None or time.perf_counter() < timeout - self.wrapup_time:

                        if state.board[i][j] == game.EMPTY_PIECE:

                            """Play A in square (i,j), update Zobrist hash"""
                            new_state = state.make_move((i, j))
                            if z_hashing is not None:
                                z_index = 0 if A_PIECE == game.X_PIECE else 1
                                new_z_key = z_key ^ z_table[i * h + j][z_index]

                            if z_hashing is not None and new_z_key in z_memory:
                                """If already calculated for this state, no need to search further"""
                                (move, value) = z_memory[new_z_key]
                            elif new_state.winner() == A_PIECE:
                                """If A has won, no need to search further"""
                                value = self.static_eval(new_state) / 10 ** (depth_remaining - 1)
                            else:
                                """Run minimax on new state"""
                                new_time_limit = timeout - time.perf_counter() if time_limit is not None else None
                                new_z_hashing = (z_table, z_memory, new_z_key) if z_hashing is not None else None
                                move, value = self.minimax(new_state, depth_remaining - 1, new_time_limit, alpha, beta,
                                                           new_z_hashing)

                            """Exit early if reached time limit"""
                            if value is None:
                                break

                            """Update best move, alpha and beta"""
                            if A_PIECE == game.X_PIECE:
                                if value > best_value:
                                    best_move = (i, j)
                                    best_value = value
                                if beta is not None and best_value > beta:
                                    return best_move, best_value
                                elif alpha is not None:
                                    alpha = max(alpha, best_value)
                            elif A_PIECE == game.O_PIECE:
                                if value < best_value:
                                    best_move = (i, j)
                                    best_value = value
                                if alpha is not None and best_value < alpha:
                                    return best_move, best_value
                                elif beta is not None:
                                    beta = min(beta, best_value)

                    else:
                        """Exit early if reached time limit"""
                        break

            if z_hashing is not None:
                z_memory[z_key] = (best_move, best_value)

            return best_move, best_value

    def static_eval(self, state: game.GameState) -> float:
        """
        Evaluates the given state. States good for X should be larger that states good for O.
        :param state: state to evaluate
        :return: evaluation of the state
        """
        self.eval_calls += 1

        h = state.h
        w = state.w
        k = state.k

        value = 0
        A_PIECE = state.next_player
        B_PIECE = game.O_PIECE if A_PIECE == game.X_PIECE else game.X_PIECE
        win_value = 10.0 ** (k + 21)
        A_sign = 1 if A_PIECE == game.X_PIECE else -1
        B_sign = -A_sign
        A_value = 0
        B_value = 0
        A_wins = 0
        B_wins = 0
        A_wins_next_turn = set()
        B_wins_next_turn = set()
        A_wins_next_next_turn = set()
        B_wins_next_next_turn = set()
        A_wins_next_next_next_turn = set()
        B_wins_next_next_next_turn = set()
        A_k_minus_1_threats = set()
        B_k_minus_1_threats = set()
        A_k_minus_2_threats = set()
        B_k_minus_2_threats = set()
        A_overlapping_k_minus_2_threats = set()
        B_overlapping_k_minus_2_threats = set()

        """Directions to search: horizontal, vertical, diagonals"""
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]

        """
        For each of the four possible directions:
        """
        for (x, y) in directions:

            """
            Limit search starting locations depending on the search direction
            """
            min_row = 0
            min_col = k - 1 if (x == -1) else 0
            max_row = w - k if (y == 1) else w - 1
            max_col = h - k if (x == 1) else h - 1

            """
            For all possible k-in-a-row starting locations (i,j):
            """
            for i in range(min_row, max_row + 1):
                for j in range(min_col, max_col + 1):

                    row = i
                    col = j
                    A_count = 0
                    B_count = 0
                    A_blocked = False
                    B_blocked = False

                    """
                    Count A's and B's in k-in-a-row-squares starting from (i,j), stepping in direction (x,y)
                    If blocked by an opposing piece, no way to get k-in-a-row = no added value
                    """
                    last_free_square = None
                    last_last_free_square = None
                    last_last_last_free_square = None
                    for square in range(k):

                        try:
                            current_piece = state.board[row][col]
                            if current_piece == game.BLOCK_PIECE:
                                A_blocked = True
                                B_blocked = True
                                break
                            elif current_piece == A_PIECE:
                                A_count += 1
                                B_blocked = True
                            elif current_piece == B_PIECE:
                                B_count += 1
                                A_blocked = True
                            else:
                                last_last_last_free_square = last_last_free_square
                                last_last_free_square = last_free_square
                                last_free_square = (row, col)
                            row += y
                            col += x
                        except Exception as e:
                            A_blocked = True
                            B_blocked = True
                            break

                    """
                    Unblocked z-filled-in-a-row counts for 10^z value
                    Each k-filled-in-a-row = win
                    Each (k-1)-filled-in-a-k-row = potential win next turn on remaining square
                    Each (k-2)-filled-in-a-k-row can become a (k-1)-filled-in-a-k-row next turn
                    Each (k-2)-filled-in-a-(k-1)-row with space on both sides = potential win next next turn
                    Each (k-2)-filled-in-a-(k-1)-row with no space both sides = can overlap with another for potential win next next turn
                    Each (k-3)-filled-in-a-k-row can become a (k-2)-filled-in-a-k-row next turn
                    Each (k-3)-filled-in-a-(k-2)-row with space on both sides = potential win next next next turn
                    """
                    if A_count > 0 and not A_blocked:
                        A_value += 10.0 ** (A_count - 1)
                        if A_count == k:
                            A_wins += 1
                        elif A_count == k - 1 and k >= 2:
                            A_wins_next_turn.add(last_free_square)
                        elif A_count == k - 2 and k >= 3:
                            A_k_minus_1_threats.add((last_free_square, last_last_free_square))
                        elif A_count == k - 3 and k >= 4:
                            A_k_minus_2_threats.add(
                                (last_free_square, last_last_free_square, last_last_last_free_square))

                    if B_count > 0 and not B_blocked:
                        B_value += 10.0 ** (B_count - 1)
                        if B_count == k:
                            B_wins += 1
                        elif B_count == k - 1 and k >= 2:
                            B_wins_next_turn.add(last_free_square)
                        elif B_count == k - 2 and k >= 3:
                            B_k_minus_1_threats.add((last_free_square, last_last_free_square))
                        elif B_count == k - 3 and k >= 4:
                            B_k_minus_2_threats.add(
                                (last_free_square, last_last_free_square, last_last_last_free_square))

        completed = set()
        for t1 in A_k_minus_1_threats:
            completed.add(t1)
            for t2 in A_k_minus_1_threats.difference(completed):
                intersection = set(t1).intersection(set(t2))
                if len(intersection) == 1:
                    overlapping_square = intersection.pop()
                    wins_next_turn = tuple(set(t1).symmetric_difference(set(t2)))
                    A_wins_next_next_turn.add((overlapping_square, wins_next_turn))

        completed = set()
        for t1 in A_k_minus_2_threats:
            completed.add(t1)
            for t2 in A_k_minus_2_threats.difference(completed):
                intersection = set(t1).intersection(set(t2))
                if len(intersection) == 2:
                    overlapping_squares = tuple(intersection)
                    edge_squares = tuple(set(t1).symmetric_difference(set(t2)))
                    A_overlapping_k_minus_2_threats.add((overlapping_squares, edge_squares))
            completed_threats = set()
            for t2 in A_k_minus_1_threats:
                completed_threats.add(t2)
                for t3 in A_k_minus_1_threats.difference(completed_threats):
                    intersection_a = set(t1).intersection(t2)
                    intersection_b = set(t1).intersection(t3)
                    if len(intersection_a) == 1 and len(intersection_b) == 1 and intersection_a.isdisjoint(
                            intersection_b):
                        overlapping_square_a = intersection_a.pop()
                        overlapping_square_b = intersection_b.pop()
                        win_next_turn = tuple(set(t1).difference({overlapping_square_a, overlapping_square_b}))
                        wins_next_next_turn_a = tuple(set({(overlapping_square_b, win_next_turn)}))
                        wins_next_next_turn_b = tuple(set({(overlapping_square_a, win_next_turn)}))
                        A_wins_next_next_next_turn.add((overlapping_square_a, wins_next_next_turn_a))
                        A_wins_next_next_next_turn.add((overlapping_square_b, wins_next_next_turn_b))

        completed = set()
        for t1 in A_overlapping_k_minus_2_threats:
            completed.add(t1)
            for t2 in A_overlapping_k_minus_2_threats.difference(completed):
                intersection = set(t1[0]).intersection(t2[0])
                if len(intersection) == 1 and len(set(t1[0]).intersection(t2[1])) == 0:
                    overlapping_square = intersection.pop()
                    overlapping_square_a = set(t1[0]).difference({overlapping_square}).pop()
                    overlapping_square_b = set(t2[0]).difference({overlapping_square}).pop()
                    wins_next_turn_a = t1[1]
                    wins_next_turn_b = t2[1]
                    wins_next_next_turn = tuple(
                        set({(overlapping_square_a, wins_next_turn_a), (overlapping_square_b, wins_next_turn_b)}))
                    A_wins_next_next_next_turn.add((overlapping_square, wins_next_next_turn))
            for t2 in A_k_minus_1_threats:
                intersection = set(t1[0]).intersection(t2)
                if len(intersection) == 1:
                    overlapping_square = intersection.pop()
                    overlapping_square_a = set(t1[0]).difference({overlapping_square}).pop()
                    wins_next_turn_a = t1[1]
                    wins_next_next_turn = tuple(set({(overlapping_square_a, wins_next_turn_a, t2)}))
                    forced_B_square = t2[1] if overlapping_square == t2[0] else t2[0]
                    if forced_B_square != wins_next_next_turn[0][0] and forced_B_square not in wins_next_next_turn[0][
                        1]:
                        A_wins_next_next_next_turn.add((overlapping_square, wins_next_next_turn))

        completed = set()
        for t1 in B_k_minus_1_threats:
            completed.add(t1)
            for t2 in B_k_minus_1_threats.difference(completed):
                intersection = set(t1).intersection(set(t2))
                if len(intersection) == 1:
                    overlapping_square = intersection.pop()
                    wins_next_turn = tuple(set(t1).symmetric_difference(set(t2)))
                    B_wins_next_next_turn.add((overlapping_square, wins_next_turn))
            completed_threats = set()
            for t2 in B_k_minus_1_threats:
                completed_threats.add(t2)
                for t3 in B_k_minus_1_threats.difference(completed_threats):
                    intersection_a = set(t1).intersection(t2)
                    intersection_b = set(t1).intersection(t3)
                    if len(intersection_a) == 1 and len(intersection_b) == 1 and intersection_a.isdisjoint(
                            intersection_b):
                        overlapping_square_a = intersection_a.pop()
                        overlapping_square_b = intersection_b.pop()
                        win_next_turn = tuple(set(t1).difference({overlapping_square_a, overlapping_square_b}))
                        wins_next_next_turn_a = tuple(set({(overlapping_square_b, win_next_turn)}))
                        wins_next_next_turn_b = tuple(set({(overlapping_square_a, win_next_turn)}))
                        B_wins_next_next_next_turn.add((overlapping_square_a, wins_next_next_turn_a))
                        B_wins_next_next_next_turn.add((overlapping_square_b, wins_next_next_turn_b))

        completed = set()
        for t1 in B_k_minus_2_threats:
            completed.add(t1)
            for t2 in B_k_minus_2_threats.difference(completed):
                intersection = set(t1).intersection(set(t2))
                if len(intersection) == 2:
                    overlapping_squares = tuple(intersection)
                    edge_squares = tuple(set(t1).symmetric_difference(set(t2)))
                    B_overlapping_k_minus_2_threats.add((overlapping_squares, edge_squares))

        completed = set()
        for t1 in B_overlapping_k_minus_2_threats:
            completed.add(t1)
            for t2 in B_overlapping_k_minus_2_threats.difference(completed):
                intersection = set(t1[0]).intersection(t2[0])
                if len(intersection) == 1 and len(set(t1[0]).intersection(t2[1])) == 0:
                    overlapping_square = intersection.pop()
                    overlapping_square_a = set(t1[0]).difference({overlapping_square}).pop()
                    overlapping_square_b = set(t2[0]).difference({overlapping_square}).pop()
                    wins_next_turn_a = t1[1]
                    wins_next_turn_b = t2[1]
                    wins_next_next_turn = tuple(
                        set({(overlapping_square_a, wins_next_turn_a), (overlapping_square_b, wins_next_turn_b)}))
                    B_wins_next_next_next_turn.add((overlapping_square, wins_next_next_turn))
            for t2 in B_k_minus_1_threats:
                intersection = set(t1[0]).intersection(t2)
                if len(intersection) == 1:
                    overlapping_square = intersection.pop()
                    overlapping_square_a = set(t1[0]).difference({overlapping_square}).pop()
                    wins_next_turn_a = t1[1]
                    wins_next_next_turn = tuple(set({(overlapping_square_a, wins_next_turn_a, t2)}))
                    forced_A_square = t2[1] if overlapping_square == t2[0] else t2[0]
                    if forced_A_square != wins_next_next_turn[0][0] and forced_A_square not in wins_next_next_turn[0][
                        1]:
                        B_wins_next_next_next_turn.add((overlapping_square, wins_next_next_turn))

        A_win_value = A_sign * win_value * (1 + A_value / 10 ** k)
        B_win_value = B_sign * win_value * (1 + B_value / 10 ** k)

        """
        ===============================================
        ====== A TO PLAY NEXT, B HAS JUST PLAYED ======
        ===============================================
                    Predict any forced wins             
        """
        if A_wins >= 1:
            """
            A won on the previous turn
            A wins in 0 moves
            """
            value = A_win_value / 10 ** 0
        elif B_wins >= 1:
            """
            B has just won on this turn
            B wins in 1 move (B)
            """
            value = B_win_value / 10 ** 1
        elif len(A_wins_next_turn) >= 1:
            """
            A will win on the next turn
            A wins in 2 moves (BA)
            """
            value = A_win_value / 10 ** 2
        elif len(B_wins_next_turn) >= 2:
            """
            Multiple ways for B to win, A can only block 1 on their turn, B wins next turn
            B wins in 3 moves (BAB)
            """
            value = B_win_value / 10 ** 3
        else:
            """
            At this point:
                len(A_wins) = 0
                len(B_wins) = 0
                len(A_wins_next_turn) = 0
                len(B_wins_next_turn) <= 1
            """

            """Fast forward through any series of forced moves"""
            fast_forward_counter = 0
            while True:
                if len(B_wins_next_turn) == 1:
                    blocking_square = B_wins_next_turn.pop()
                    state = state.make_move(blocking_square)
                    # print(f"Play A at {blocking_square} to block B's win")
                    # print(state)
                    fast_forward_counter += 1

                    A_k_minus_1_threats_revised = A_k_minus_1_threats.copy()
                    for t1 in A_k_minus_1_threats:
                        if blocking_square in t1:
                            A_wins_next_turn.update(set(t1).difference({blocking_square}))
                            A_k_minus_1_threats_revised.remove(t1)
                    A_k_minus_1_threats = A_k_minus_1_threats_revised

                    A_wins_next_next_turn_revised = A_wins_next_next_turn.copy()
                    for w1 in A_wins_next_next_turn:
                        if blocking_square == w1[0]:
                            A_wins_next_turn.update(w1[1])
                            A_wins_next_next_turn_revised.remove(w1)
                        else:
                            for w2 in w1[1]:
                                if blocking_square == w2:
                                    A_wins_next_turn.add(w1[0])
                                    A_wins_next_next_turn_revised.remove(w1)
                    A_wins_next_next_turn = A_wins_next_next_turn_revised

                    A_k_minus_2_threats_revised = A_k_minus_2_threats.copy()
                    for t1 in A_k_minus_2_threats:
                        if blocking_square in t1:
                            A_k_minus_1_threats.add(tuple(set(t1).difference({blocking_square})))
                            A_k_minus_2_threats_revised.remove(t1)
                    A_k_minus_2_threats = A_k_minus_2_threats_revised

                    A_wins_next_next_next_turn_revised = A_wins_next_next_next_turn.copy()
                    for w1 in A_wins_next_next_next_turn:
                        if blocking_square == w1[0]:
                            A_wins_next_next_turn.update(w1[1])
                            A_wins_next_next_next_turn_revised.remove(w1)
                        else:
                            for w2 in w1[1]:
                                if blocking_square == w2[0]:
                                    A_wins_next_next_turn.add(w2)
                                    A_wins_next_next_next_turn_revised.remove(w1)
                    A_wins_next_next_next_turn = A_wins_next_next_next_turn_revised

                    completed = set()
                    for t1 in A_k_minus_1_threats:
                        completed.add(t1)
                        for t2 in A_k_minus_1_threats.difference(completed):
                            intersection = set(t1).intersection(set(t2))
                            if len(intersection) == 1:
                                overlapping_square = intersection.pop()
                                wins_next_turn = tuple(set(t1).symmetric_difference(set(t2)))
                                A_wins_next_next_turn.add((overlapping_square, wins_next_turn))

                    completed = set()
                    for t1 in A_k_minus_2_threats:
                        completed.add(t1)
                        for t2 in A_k_minus_2_threats.difference(completed):
                            intersection = set(t1).intersection(set(t2))
                            if len(intersection) == 2:
                                overlapping_squares = tuple(intersection)
                                edge_squares = tuple(set(t1).symmetric_difference(set(t2)))
                                A_overlapping_k_minus_2_threats.add((overlapping_squares, edge_squares))
                        completed_threats = set()
                        for t2 in A_k_minus_1_threats:
                            completed_threats.add(t2)
                            for t3 in A_k_minus_1_threats.difference(completed_threats):
                                intersection_a = set(t1).intersection(t2)
                                intersection_b = set(t1).intersection(t3)
                                if len(intersection_a) == 1 and len(intersection_b) == 1 and intersection_a.isdisjoint(
                                        intersection_b):
                                    overlapping_square_a = intersection_a.pop()
                                    overlapping_square_b = intersection_b.pop()
                                    win_next_turn = tuple(
                                        set(t1).difference({overlapping_square_a, overlapping_square_b}))
                                    wins_next_next_turn_a = tuple(set({(overlapping_square_b, win_next_turn)}))
                                    wins_next_next_turn_b = tuple(set({(overlapping_square_a, win_next_turn)}))
                                    A_wins_next_next_next_turn.add((overlapping_square_a, wins_next_next_turn_a))
                                    A_wins_next_next_next_turn.add((overlapping_square_b, wins_next_next_turn_b))

                    completed = set()
                    for t1 in A_overlapping_k_minus_2_threats:
                        completed.add(t1)
                        for t2 in A_overlapping_k_minus_2_threats.difference(completed):
                            intersection = set(t1[0]).intersection(t2[0])
                            if len(intersection) == 1 and len(set(t1[0]).intersection(t2[1])) == 0:
                                overlapping_square = intersection.pop()
                                overlapping_square_a = set(t1[0]).difference({overlapping_square}).pop()
                                overlapping_square_b = set(t2[0]).difference({overlapping_square}).pop()
                                wins_next_turn_a = t1[1]
                                wins_next_turn_b = t2[1]
                                wins_next_next_turn = tuple(set({(overlapping_square_a, wins_next_turn_a),
                                                                 (overlapping_square_b, wins_next_turn_b)}))
                                A_wins_next_next_next_turn.add((overlapping_square, wins_next_next_turn))
                        for t2 in A_k_minus_1_threats:
                            intersection = set(t1[0]).intersection(t2)
                            if len(intersection) == 1:
                                overlapping_square = intersection.pop()
                                overlapping_square_a = set(t1[0]).difference({overlapping_square}).pop()
                                wins_next_turn_a = t1[1]
                                wins_next_next_turn = tuple(set({(overlapping_square_a, wins_next_turn_a, t2)}))
                                forced_B_square = t2[1] if overlapping_square == t2[0] else t2[0]
                                if forced_B_square != wins_next_next_turn[0][0] and forced_B_square not in \
                                        wins_next_next_turn[0][1]:
                                    A_wins_next_next_next_turn.add((overlapping_square, wins_next_next_turn))

                    for (x, y) in directions:
                        for scan in range(1 - k, 1):
                            i = blocking_square[0] + scan * y
                            j = blocking_square[1] + scan * x
                            row = i
                            col = j
                            A_count = 0
                            B_count = 0
                            A_blocked = False
                            B_blocked = False
                            last_last_last_free_square = None
                            last_last_free_square = None
                            last_free_square = None
                            for square in range(k):
                                if row < 0 or row > h - 1 or col < 0 or col > w - 1:
                                    A_blocked = True
                                    B_blocked = True
                                    break

                                try:
                                    current_piece = state.board[row][col]
                                    if current_piece == game.BLOCK_PIECE:
                                        A_blocked = True
                                        B_blocked = True
                                        break
                                    elif current_piece == B_PIECE:
                                        B_count += 1
                                        A_blocked = True
                                    elif current_piece == A_PIECE:
                                        A_count += 1
                                        if (row, col) != blocking_square:
                                            B_blocked = True
                                    else:
                                        last_last_last_free_square = last_last_free_square
                                        last_last_free_square = last_free_square
                                        last_free_square = (row, col)
                                    row += y
                                    col += x
                                except Exception as e:
                                    A_blocked = True
                                    B_blocked = True
                                    break

                            if B_count > 0 and not B_blocked:
                                B_value -= 10.0 ** (B_count - 1)
                            if A_count > 0 and not A_blocked:
                                A_value += 10.0 ** (A_count - 1)
                                if A_count > 1:
                                    A_value -= 10.0 ** (A_count - 2)
                                if A_count == k - 2 and k >= 3:
                                    A_k_minus_1_threats.add((last_free_square, last_last_free_square))
                                elif A_count == k - 3 and k >= 4:
                                    A_k_minus_2_threats.add(
                                        (last_free_square, last_last_free_square, last_last_last_free_square))

                    B_k_minus_1_threats_revised = B_k_minus_1_threats.copy()
                    for t1 in B_k_minus_1_threats:
                        if blocking_square in t1:
                            B_k_minus_1_threats_revised.remove(t1)
                    B_k_minus_1_threats = B_k_minus_1_threats_revised

                    B_wins_next_next_turn_revised = B_wins_next_next_turn.copy()
                    for w1 in B_wins_next_next_turn:
                        if blocking_square == w1[0] or blocking_square in w1[1]:
                            B_wins_next_next_turn_revised.remove(w1)
                    B_wins_next_next_turn = B_wins_next_next_turn_revised

                    B_k_minus_2_threats_revised = B_k_minus_2_threats.copy()
                    for t1 in B_k_minus_2_threats:
                        if blocking_square in t1:
                            B_k_minus_2_threats_revised.remove(t1)
                    B_k_minus_2_threats = B_k_minus_2_threats_revised

                    B_k_minus_2_threats_revised = B_k_minus_2_threats.copy()
                    for t1 in B_k_minus_2_threats:
                        if blocking_square in t1[0] or blocking_square in t1[1]:
                            B_k_minus_2_threats_revised.remove(t1)
                    B_k_minus_2_threats = B_k_minus_2_threats_revised

                    B_wins_next_next_next_turn_revised = B_wins_next_next_next_turn.copy()
                    for w1 in B_wins_next_next_next_turn:
                        if blocking_square == w1[0]:
                            B_wins_next_next_next_turn_revised.remove(w1)
                        else:
                            for w2 in w1[1]:
                                if blocking_square == w2[0] or blocking_square in w2[1] or (
                                        len(w2) == 3 and blocking_square in w2[2]):
                                    B_wins_next_next_next_turn_revised.remove(w1)
                                    break
                    B_wins_next_next_next_turn = B_wins_next_next_next_turn_revised

                    A_win_value = A_sign * win_value * (1 + A_value / 10 ** k)
                    B_win_value = B_sign * win_value * (1 + B_value / 10 ** k)

                    if len(A_wins_next_turn) == 1:
                        blocking_square = A_wins_next_turn.pop()
                        state = state.make_move(blocking_square)
                        # print(f"Play B at {blocking_square} to block A's win")
                        # print(state)
                        fast_forward_counter += 1

                        B_k_minus_1_threats_revised = B_k_minus_1_threats.copy()
                        for t1 in B_k_minus_1_threats:
                            if blocking_square in t1:
                                B_wins_next_turn.update(set(t1).difference({blocking_square}))
                                B_k_minus_1_threats_revised.remove(t1)
                        B_k_minus_1_threats = B_k_minus_1_threats_revised

                        B_wins_next_next_turn_revised = B_wins_next_next_turn.copy()
                        for w1 in B_wins_next_next_turn:
                            if blocking_square == w1[0]:
                                B_wins_next_turn.update(w1[1])
                                B_wins_next_next_turn_revised.remove(w1)
                            else:
                                for w2 in w1[1]:
                                    if blocking_square == w2:
                                        B_wins_next_turn.add(w1[0])
                                        B_wins_next_next_turn_revised.remove(w1)
                        B_wins_next_next_turn = B_wins_next_next_turn_revised

                        B_k_minus_2_threats_revised = B_k_minus_2_threats.copy()
                        for t1 in B_k_minus_2_threats:
                            if blocking_square in t1:
                                B_k_minus_1_threats.add(tuple(set(t1).difference({blocking_square})))
                                B_k_minus_2_threats_revised.remove(t1)
                        B_k_minus_2_threats = B_k_minus_2_threats_revised

                        B_wins_next_next_next_turn_revised = B_wins_next_next_next_turn.copy()
                        for w1 in B_wins_next_next_next_turn:
                            if blocking_square == w1[0]:
                                B_wins_next_next_turn.update(w1[1])
                                B_wins_next_next_next_turn_revised.remove(w1)
                            else:
                                for w2 in w1[1]:
                                    if blocking_square == w2[0]:
                                        B_wins_next_next_turn.add(w2)
                                        B_wins_next_next_next_turn_revised.remove(w1)
                        B_wins_next_next_next_turn = B_wins_next_next_next_turn_revised

                        completed = set()
                        for t1 in B_k_minus_1_threats:
                            completed.add(t1)
                            for t2 in B_k_minus_1_threats.difference(completed):
                                intersection = set(t1).intersection(set(t2))
                                if len(intersection) == 1:
                                    overlapping_square = intersection.pop()
                                    wins_next_turn = tuple(set(t1).symmetric_difference(set(t2)))
                                    B_wins_next_next_turn.add((overlapping_square, wins_next_turn))

                        completed = set()
                        for t1 in B_k_minus_2_threats:
                            completed.add(t1)
                            for t2 in B_k_minus_2_threats.difference(completed):
                                intersection = set(t1).intersection(set(t2))
                                if len(intersection) == 2:
                                    overlapping_squares = tuple(intersection)
                                    edge_squares = tuple(set(t1).symmetric_difference(set(t2)))
                                    B_overlapping_k_minus_2_threats.add((overlapping_squares, edge_squares))
                        completed_threats = set()
                        for t2 in B_k_minus_1_threats:
                            completed_threats.add(t2)
                            for t3 in B_k_minus_1_threats.difference(completed_threats):
                                intersection_a = set(t1).intersection(t2)
                                intersection_b = set(t1).intersection(t3)
                                if len(intersection_a) == 1 and len(intersection_b) == 1 and intersection_a.isdisjoint(
                                        intersection_b):
                                    overlapping_square_a = intersection_a.pop()
                                    overlapping_square_b = intersection_b.pop()
                                    win_next_turn = tuple(
                                        set(t1).difference({overlapping_square_a, overlapping_square_b}))
                                    wins_next_next_turn_a = tuple(set({(overlapping_square_b, win_next_turn)}))
                                    wins_next_next_turn_b = tuple(set({(overlapping_square_a, win_next_turn)}))
                                    B_wins_next_next_next_turn.add((overlapping_square_a, wins_next_next_turn_a))
                                    B_wins_next_next_next_turn.add((overlapping_square_b, wins_next_next_turn_b))

                        completed = set()
                        for t1 in B_overlapping_k_minus_2_threats:
                            completed.add(t1)
                            for t2 in B_overlapping_k_minus_2_threats.difference(completed):
                                intersection = set(t1[0]).intersection(t2[0])
                                if len(intersection) == 1 and len(set(t1[0]).intersection(t2[1])) == 0:
                                    overlapping_square = intersection.pop()
                                    overlapping_square_a = set(t1[0]).difference({overlapping_square}).pop()
                                    overlapping_square_b = set(t2[0]).difference({overlapping_square}).pop()
                                    wins_next_turn_a = t1[1]
                                    wins_next_turn_b = t2[1]
                                    wins_next_next_turn = tuple(set({(overlapping_square_a, wins_next_turn_a),
                                                                     (overlapping_square_b, wins_next_turn_b)}))
                                    B_wins_next_next_next_turn.add((overlapping_square, wins_next_next_turn))
                            for t2 in B_k_minus_1_threats:
                                intersection = set(t1[0]).intersection(t2)
                                if len(intersection) == 1:
                                    overlapping_square = intersection.pop()
                                    overlapping_square_a = set(t1[0]).difference({overlapping_square}).pop()
                                    wins_next_turn_a = t1[1]
                                    wins_next_next_turn = tuple(set({(overlapping_square_a, wins_next_turn_a, t2)}))
                                    forced_A_square = t2[1] if overlapping_square == t2[0] else t2[0]
                                    if forced_A_square != wins_next_next_turn[0][0] and forced_A_square not in \
                                            wins_next_next_turn[0][1]:
                                        B_wins_next_next_next_turn.add((overlapping_square, wins_next_next_turn))

                        for (x, y) in directions:
                            for scan in range(1 - k, 1):
                                i = blocking_square[0] + scan * y
                                j = blocking_square[1] + scan * x
                                row = i
                                col = j
                                A_count = 0
                                B_count = 0
                                A_blocked = False
                                B_blocked = False
                                last_last_last_free_square = None
                                last_last_free_square = None
                                last_free_square = None
                                for square in range(k):
                                    if row < 0 or row > h - 1 or col < 0 or col > w - 1:
                                        A_blocked = True
                                        B_blocked = True
                                        break

                                    try:
                                        current_piece = state.board[row][col]
                                        if current_piece == game.BLOCK_PIECE:
                                            A_blocked = True
                                            B_blocked = True
                                            break
                                        elif current_piece == A_PIECE:
                                            A_count += 1
                                            B_blocked = True
                                        elif current_piece == B_PIECE:
                                            B_count += 1
                                            if (row, col) != blocking_square:
                                                A_blocked = True
                                        else:
                                            last_last_last_free_square = last_last_free_square
                                            last_last_free_square = last_free_square
                                            last_free_square = (row, col)
                                        row += y
                                        col += x
                                    except Exception as e:
                                        A_blocked = True
                                        B_blocked = True
                                        break

                                if A_count > 0 and not A_blocked:
                                    B_value -= 10.0 ** (B_count - 1)
                                if B_count > 0 and not B_blocked:
                                    B_value += 10.0 ** (B_count - 1)
                                    if B_count > 1:
                                        B_value -= 10.0 ** (B_count - 2)
                                    if B_count == k - 2 and k >= 3:
                                        B_k_minus_1_threats.add((last_free_square, last_last_free_square))
                                    elif B_count == k - 3 and k >= 4:
                                        B_k_minus_2_threats.add(
                                            (last_free_square, last_last_free_square, last_last_last_free_square))

                        A_k_minus_1_threats_revised = A_k_minus_1_threats.copy()
                        for t1 in A_k_minus_1_threats:
                            if blocking_square in t1:
                                A_k_minus_1_threats_revised.remove(t1)
                        A_k_minus_1_threats = A_k_minus_1_threats_revised

                        A_wins_next_next_turn_revised = A_wins_next_next_turn.copy()
                        for w1 in A_wins_next_next_turn:
                            if blocking_square == w1[0] or blocking_square in w1[1]:
                                A_wins_next_next_turn_revised.remove(w1)
                        A_wins_next_next_turn = A_wins_next_next_turn_revised

                        A_k_minus_2_threats_revised = A_k_minus_2_threats.copy()
                        for t1 in A_k_minus_2_threats:
                            if blocking_square in t1:
                                A_k_minus_2_threats_revised.remove(t1)
                        A_k_minus_2_threats = A_k_minus_2_threats_revised

                        A_k_minus_2_threats_revised = A_k_minus_2_threats.copy()
                        for t1 in A_k_minus_2_threats:
                            if blocking_square in t1[0] or blocking_square in t1[1]:
                                A_k_minus_2_threats_revised.remove(t1)
                        A_k_minus_2_threats = A_k_minus_2_threats_revised

                        A_wins_next_next_next_turn_revised = A_wins_next_next_next_turn.copy()
                        for w1 in A_wins_next_next_next_turn:
                            if blocking_square == w1[0]:
                                A_wins_next_next_next_turn_revised.remove(w1)
                            else:
                                for w2 in w1[1]:
                                    if blocking_square == w2[0] or blocking_square in w2[1] or (
                                            len(w2) == 3 and blocking_square in w2[2]):
                                        A_wins_next_next_next_turn_revised.remove(w1)
                                        break
                        A_wins_next_next_next_turn = A_wins_next_next_next_turn_revised

                        A_win_value = A_sign * win_value * (1 + A_value / 10 ** k)
                        B_win_value = B_sign * win_value * (1 + B_value / 10 ** k)

                    else:
                        if len(A_wins_next_turn) > 1:
                            """
                            A has multiple wins on their next turn,
                            B can only block one of them, B wins on the following turn
                            A wins in 3 moves after fast forward (ff-ABA)         
                            """
                            value = round(A_win_value / 10 ** (3 + fast_forward_counter))

                        """Fast forwarded out of sync, need to swap A and B"""
                        A_sign, B_sign = B_sign, A_sign
                        A_value, B_value = B_value, A_value
                        A_win_value, B_win_value = B_win_value, A_win_value
                        A_wins_next_turn, B_wins_next_turn = B_wins_next_turn, A_wins_next_turn
                        A_wins_next_next_turn, B_wins_next_next_turn = B_wins_next_next_turn, A_wins_next_next_turn
                        A_wins_next_next_next_turn, B_wins_next_next_next_turn = B_wins_next_next_next_turn, A_wins_next_next_next_turn
                        A_k_minus_1_threats, B_k_minus_1_threats = B_k_minus_1_threats, A_k_minus_1_threats
                        A_k_minus_2_threats, B_k_minus_2_threats = B_k_minus_2_threats, A_k_minus_2_threats

                        break

                else:
                    if len(B_wins_next_turn) > 1:
                        """
                        B has multiple wins on their next turn,
                        A can only block one of them, B wins on the following turn
                        B wins in 3 moves after fast forward (ff-BAB)         
                        """
                        value = round(B_win_value / 10 ** (3 + fast_forward_counter))

                    break

            """
            At this point:
                len(A_wins) = 0
                len(B_wins) = 0
                len(A_wins_next_turn) = 0
                len(B_wins_next_turn) = 0
            """

            if value == 0 and len(A_wins_next_next_turn) >= 1:
                """
                A can play a piece such that they have multiple wins on their next turn,
                B can only block one of them, A wins on the following turn
                A wins in 4 moves after fast forward (ff-BABA)
                """
                value = round(A_win_value / 10 ** (4 + fast_forward_counter))

            """
            At this point:
                len(A_wins) = 0
                len(B_wins) = 0
                len(A_wins_next_turn) = 0
                len(B_wins_next_turn) = 0
                len(A_wins_next_next_turn) = 0
            """

            if value == 0 and len(B_wins_next_next_turn) >= 1 and len(A_k_minus_1_threats) == 0:
                found_win = False
                for w1 in B_wins_next_next_turn:
                    if found_win: break
                    for w2 in B_wins_next_next_turn.difference({w1}):
                        if w1[0] != w2[0] and w1[0] not in w2[1] and w2[0] not in w1[1] and set(w1[1]).isdisjoint(
                                set(w2[1])):
                            found_win = True
                            break
                if found_win:
                    """
                    B can play a piece such that they have multiple wins on their next next turn,
                    A cannot block all of them, B wins in the end
                    B wins in 5 moves after fast forward (ff-BABAB)
                    """
                    value = round(B_win_value / 10 ** (5 + fast_forward_counter))

            if value == 0 and len(A_wins_next_next_next_turn) >= 1:
                found_win = False
                if len(B_k_minus_1_threats) == 0:
                    for w1 in A_wins_next_next_next_turn:
                        found_win = True
                elif len(A_k_minus_1_threats) >= 1:
                    B_k_minus_1_threat_squares = {}
                    for t1 in B_k_minus_1_threats:
                        B_k_minus_1_threat_squares[t1[0]] = t1[1]
                        B_k_minus_1_threat_squares[t1[1]] = t1[0]
                    for t1 in A_k_minus_1_threats:
                        if found_win: break
                        for w1 in A_wins_next_next_next_turn:
                            if found_win: break
                            if w1[0] in t1 or (t1[0] == w1[1][0][0] or (len(w1[1]) == 2 and t1[1] == w1[1][1][0])):
                                force_B_square = t1[1] if w1[0] == t1[0] else t1[0]
                                next_A_squares = set()
                                for w2 in w1[1]:
                                    next_A_squares.add(w2[0])
                                if force_B_square not in B_k_minus_1_threat_squares or B_k_minus_1_threat_squares[
                                    force_B_square] in next_A_squares:
                                    found_win = True

                if found_win:
                    """
                    A can play a piece such that they have multiple wins on their next next turn,
                    B cannot block all of them, A wins in the end
                    A wins in 6 moves after fast forward (ff-BABABA)
                    """
                    value = round(A_win_value / 10 ** (6 + fast_forward_counter))

            if value == 0 and len(B_wins_next_next_next_turn) >= 1 and len(A_k_minus_1_threats) == 0:

                found_win = False

                for w1 in B_wins_next_next_turn:
                    if found_win: break
                    for w2 in B_wins_next_next_next_turn:
                        if found_win: break
                        w2_squares = set({w2[1][0][0]}).union(w2[1][0][1])
                        if len(w2[1]) == 2:
                            w2_squares.update(set({w2[1][1][0]}).union(w2[1][1][1]))
                        if w1[0] != w2[0] and w1[0] not in w2_squares and w2[0] not in w1[1] and set(w1[1]).isdisjoint(
                                w2_squares):
                            found_win = True
                            for t1 in A_k_minus_2_threats:
                                if w1[0] in t1 or not set(w1[1]).isdisjoint(t1):
                                    found_win = False

                if not found_win and len(B_wins_next_next_next_turn) >= 2:
                    completed = set()
                    for w1 in B_wins_next_next_next_turn:
                        completed.add(w1)
                        if found_win: break
                        for w2 in B_wins_next_next_next_turn.difference(completed):
                            if found_win: break

                            w1_squares = set({w1[1][0][0]}).union(w1[1][0][1])
                            if len(w1[1][0]) == 3:
                                w1_squares.update(w1[1][0][2])
                            if len(w1[1]) == 2:
                                w1_squares.update(set({w1[1][1][0]}).union(w1[1][1][1]))

                            w2_squares = set({w2[1][0][0]}).union(w2[1][0][1])
                            if len(w2[1][0]) == 3:
                                w2_squares.update(w2[1][0][2])
                            if len(w2[1]) == 2:
                                w2_squares.update(set({w2[1][1][0]}).union(w2[1][1][1]))

                            if (len(w1[1]) == 2 or len(w2[1]) == 2) and len(A_k_minus_2_threats) != 0:
                                break
                            if w1[0] != w2[0] and w1[0] not in w2_squares and w2[
                                0] not in w1_squares and w1_squares.isdisjoint(w2_squares):
                                found_win = True

                if found_win:
                    """
                    B can play a piece such that they have at least four wins on their next next turn,
                    A can only block two of them, B wins in the end
                    B wins in 7 moves after fast forward (ff-BABABAB)
                    """
                    value = round(B_win_value / 10 ** (7 + fast_forward_counter))

        """No forced wins detected"""
        if value == 0:
            value = round(A_sign * A_value + B_sign * B_value)

        return value
