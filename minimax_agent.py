"""
minimax_agent.py
author: <YOUR NAME(s) HERE>
"""
from math import log10
import agent
import game
import time
import random


def process_count(k, last_free_squares, count, value, wins,
                  win_next_turn_threats, k_minus_1_threats, k_minus_2_threats):
    """
    Update the total value and record any k-rows close to being filled
    """

    """An unblocked k-row filled with x squares counts for 10^(x-1) value"""
    value += 10.0 ** (count - 1)
    if count == k:
        """k out of k squares are filled, game over"""
        wins += 1
    elif count == k - 1 and k >= 2:
        """k-1 out of k squares are filled, player can win next turn"""
        win_next_turn_threats.add(last_free_squares[0])
    elif count == k - 2 and k >= 3:
        """k-2 out of k squares are filled, player can fill to k-1 next turn"""
        k_minus_1_threats.add((last_free_squares[0], last_free_squares[1]))
    elif count == k - 3 and k >= 4:
        """k-3 out of k squares are filled, player can fill to k-2 next turn"""
        k_minus_2_threats.add((last_free_squares[0], last_free_squares[1], last_free_squares[2]))
    return value, wins


def detect_threats(k_minus_1_threats, k_minus_2_threats, overlapping_k_minus_2_threats,
                   win_next_next_turn_threats, win_next_next_next_turn_threats):
    """
    See if any existing threats overlap to form win threats
    """

    """
    Record places where two win threats can be created on the same turn
    """
    completed = set()
    for t1 in k_minus_1_threats:
        completed.add(t1)
        for t2 in k_minus_1_threats.difference(completed):
            intersection = set(t1).intersection(set(t2))
            if len(intersection) == 1:
                overlapping_square = intersection.pop()
                win_next_turn_threats = tuple(set(t1).symmetric_difference(set(t2)))
                win_next_next_turn_threats.add((overlapping_square, win_next_turn_threats))

    """
    Record places where two parallel k-2 threats can be created on the same turn
    """
    completed = set()
    for t1 in k_minus_2_threats:
        completed.add(t1)
        for t2 in k_minus_2_threats.difference(completed):
            intersection = set(t1).intersection(set(t2))
            if len(intersection) == 2:
                overlapping_squares = tuple(intersection)
                edge_squares = tuple(set(t1).symmetric_difference(set(t2)))
                overlapping_k_minus_2_threats.add((overlapping_squares, edge_squares))

    completed = set()
    for t1 in overlapping_k_minus_2_threats:

        """
        Record places where two sets of two parallel k-2 threats can be created on the same turn
        """
        completed.add(t1)
        for t2 in overlapping_k_minus_2_threats.difference(completed):
            intersection = set(t1[0]).intersection(t2[0])
            if len(intersection) == 1 and len(set(t1[0]).intersection(t2[1])) == 0:
                overlapping_square = intersection.pop()
                overlapping_square_a = set(t1[0]).difference({overlapping_square}).pop()
                overlapping_square_b = set(t2[0]).difference({overlapping_square}).pop()
                win_next_turn_threats_a = t1[1]
                win_next_turn_threats_b = t2[1]
                win_next_next_turn_threats = tuple(
                    {(overlapping_square_a, win_next_turn_threats_a), (overlapping_square_b, win_next_turn_threats_b)})
                win_next_next_next_turn_threats.add((overlapping_square, win_next_next_turn_threats))

        """
        Record places where two parallel k-2 threats and a k-1 threat can be created on the same turn
        """
        for t2 in k_minus_1_threats:
            intersection = set(t1[0]).intersection(t2)
            if len(intersection) == 1:
                overlapping_square = intersection.pop()
                overlapping_square_a = set(t1[0]).difference({overlapping_square}).pop()
                win_next_turn_threats_a = t1[1]
                win_next_next_turn_threats = tuple({(overlapping_square_a, win_next_turn_threats_a, t2)})
                forced_square = t2[1] if overlapping_square == t2[0] else t2[0]
                if (forced_square != win_next_next_turn_threats[0][0] and
                        forced_square not in win_next_next_turn_threats[0][1]):
                    win_next_next_next_turn_threats.add((overlapping_square, win_next_next_turn_threats))


def update_threats(blocking_square, k_minus_1_threats, k_minus_2_threats, win_next_turn_threats,
                   win_next_next_turn_threats, win_next_next_next_turn_threats):
    """
    Record any new threats resulting from blocking a win
    """

    tagged_for_removal = set()
    for t1 in k_minus_1_threats:
        if blocking_square in t1:
            win_next_turn_threats.update(set(t1).difference({blocking_square}))
            tagged_for_removal.add(t1)
    k_minus_1_threats.difference_update(tagged_for_removal)

    tagged_for_removal = set()
    for w1 in win_next_next_turn_threats:
        if blocking_square == w1[0]:
            win_next_turn_threats.update(w1[1])
            tagged_for_removal.add(w1)
        else:
            for w2 in w1[1]:
                if blocking_square == w2:
                    win_next_turn_threats.add(w1[0])
                    tagged_for_removal.add(w1)
    win_next_next_turn_threats.difference_update(tagged_for_removal)

    tagged_for_removal = set()
    for t1 in k_minus_2_threats:
        if blocking_square in t1:
            k_minus_1_threats.add(tuple(set(t1).difference({blocking_square})))
            tagged_for_removal.add(t1)
    k_minus_2_threats.difference_update(tagged_for_removal)

    tagged_for_removal = set()
    for w1 in win_next_next_next_turn_threats:
        if blocking_square == w1[0]:
            win_next_next_turn_threats.update(w1[1])
            tagged_for_removal.add(w1)
        else:
            for w2 in w1[1]:
                if blocking_square == w2[0]:
                    win_next_next_turn_threats.add(w2)
                    tagged_for_removal.add(w1)
    win_next_next_next_turn_threats.difference_update(tagged_for_removal)


def block_threats(blocking_square, k_minus_1_threats, k_minus_2_threats, win_next_turn_threats,
                  win_next_next_turn_threats, win_next_next_next_turn_threats):
    """
    If blocking a win also blocks any other threats, remove them from storage.
    """

    tagged_for_removal = set()
    for t1 in k_minus_1_threats:
        if blocking_square in t1:
            tagged_for_removal.add(t1)
    k_minus_1_threats.difference_update(tagged_for_removal)

    tagged_for_removal = set()
    for w1 in win_next_next_turn_threats:
        if blocking_square == w1[0] or blocking_square in w1[1]:
            tagged_for_removal.add(w1)
    win_next_next_turn_threats.difference_update(tagged_for_removal)

    tagged_for_removal = set()
    for t1 in k_minus_2_threats:
        if blocking_square in t1:
            tagged_for_removal.add(t1)
    k_minus_2_threats.difference_update(tagged_for_removal)

    tagged_for_removal = set()
    for t1 in k_minus_2_threats:
        if blocking_square in t1[0] or blocking_square in t1[1]:
            tagged_for_removal.add(t1)
    k_minus_2_threats.difference_update(tagged_for_removal)

    tagged_for_removal = set()
    for w1 in win_next_next_next_turn_threats:
        if blocking_square == w1[0]:
            tagged_for_removal.add(w1)
        else:
            for w2 in w1[1]:
                if blocking_square == w2[0] or blocking_square in w2[1] or (
                        len(w2) == 3 and blocking_square in w2[2]):
                    tagged_for_removal.add(w1)
                    break
    win_next_next_next_turn_threats.difference_update(tagged_for_removal)


def get_squares_from_threat(threat):
    """
    Helper method to deconstruct an element of win_next_next_next_turn_threats
    """
    squares = {threat[1][0][0]}.union(threat[1][0][1])
    if len(threat[1][0]) == 3:
        squares.update(threat[1][0][2])
    if len(threat[1]) == 2:
        squares.update({threat[1][1][0]}.union(threat[1][1][1]))
    return squares


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

        """Limit the maximum search depth to 3"""
        max_depth = min(max_depth, 3)

        """Perform iterative deepening search until depth limit or time limit reached"""
        timeout = time.perf_counter() + time_limit if time_limit is not None else None
        depth = 1
        while depth <= max_depth:

            """Initialise Zobrist hash table"""
            z_table = [[random.getrandbits(32) for _ in range(2)] for _ in range(h * w)]
            z_hashing = (z_table, dict(), 0)

            """Search for best value at current depth"""
            latest_time_limit = timeout - time.perf_counter() if timeout is not None else None
            move, value = self.minimax(state, depth, latest_time_limit, float("-inf"), float("inf"), z_hashing)

            if time_limit is None or time.perf_counter() < timeout - self.wrapup_time:

                """Full search complete, update best_move"""
                best_move = move
                best_value = value
                if not self.silent:
                    print(f"depth={depth}, best_move={best_move}, best_value={best_value}")

                """Guaranteed to win, stop search"""
                if (self.piece == game.X_PIECE and best_value >= 10 ** k or
                        self.piece == game.O_PIECE and best_value <= -10 ** k):
                    if not self.silent:
                        print(f"Win found in {round(k + 20 - log10(abs(best_value))) + depth} moves")
                    break
                elif (self.piece == game.X_PIECE and best_value <= -10 ** k or
                      self.piece == game.O_PIECE and best_value >= 10 ** k):
                    if not self.silent:
                        print(f"Loss found in {round(k + 20 - log10(abs(best_value))) + depth} moves")
                    break

                """Search again, one layer deeper"""
                depth += 1

            else:
                """Time limit reached, exit search"""
                break

        if not self.silent:

            if timeout is not None:
                """Report remaining time"""
                print(f"Exited {round(timeout - time.perf_counter(), 4)} seconds remaining before timeout")

            """Report total number of static evaluations made"""
            print(f"Called static_eval() {self.eval_calls} times")

            self.print_board(state, best_move)

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
        a_piece = state.next_player

        """Generate Zobrist hash for the current board state"""
        (z_table, z_memory, z_key) = (None, None, None)
        if z_hashing is not None:
            (z_table, z_memory, z_key) = z_hashing

        if time_limit is not None and time_limit < self.wrapup_time:
            """Exit early if reached time limit"""
            return None, None
        elif depth_remaining == 0:
            """Return static evaluation if reached depth limit"""
            value = self.static_eval(state)
            if z_memory is not None:
                z_memory[z_key] = (None, value)
            return None, value
        else:
            """Otherwise do minimax"""

            timeout = None
            if time_limit is not None:
                timeout = time.perf_counter() + time_limit

            best_move = None
            best_value = float("-inf") if a_piece == game.X_PIECE else float("inf")

            # """Default best move is first available empty space"""
            # for i in range(w):
            #     for j in range(h):
            #         if state.board[i][j] == game.EMPTY_PIECE:
            #             best_move = (i, j)
            #             i = w - 1
            #             break

            for i in range(w):
                for j in range(h):
                    """Iterate until all spaces have been tried, exit early if time limit is reached"""
                    if timeout is None or time.perf_counter() < timeout - self.wrapup_time:

                        if state.board[i][j] == game.EMPTY_PIECE:

                            """Play A in square (i,j), update Zobrist hash"""
                            new_state = state.make_move((i, j))
                            new_z_key = None
                            if z_hashing is not None:
                                z_index = 0 if a_piece == game.X_PIECE else 1
                                new_z_key = z_key ^ z_table[i * h + j][z_index]

                            if new_z_key is not None and new_z_key in z_memory:
                                """If already calculated for this state, no need to search further"""
                                (move, value) = z_memory[new_z_key]
                            elif new_state.winner() == a_piece:
                                """If A has won, no need to search further"""
                                value = self.static_eval(new_state) / 10 ** (depth_remaining - 1)
                            else:
                                """Run minimax on new state"""
                                new_time_limit = None
                                if timeout is not None:
                                    new_time_limit = timeout - time.perf_counter()
                                new_z_hashing = (z_table, z_memory, new_z_key)
                                move, value = self.minimax(new_state, depth_remaining - 1, new_time_limit, alpha, beta,
                                                           new_z_hashing)

                            """Exit early if reached time limit"""
                            if value is None:
                                break

                            """Update best move, alpha and beta"""
                            if a_piece == game.X_PIECE:
                                if value > best_value:
                                    best_move = (i, j)
                                    best_value = value
                                if beta is not None and best_value > beta:
                                    return best_move, best_value
                                elif alpha is not None:
                                    alpha = max(alpha, best_value)
                            elif a_piece == game.O_PIECE:
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
        win_value = 10.0 ** (k + 21)

        a_piece = state.next_player
        b_piece = game.O_PIECE if a_piece == game.X_PIECE else game.X_PIECE
        a_sign = 1 if a_piece == game.X_PIECE else -1
        b_sign = -a_sign
        a_value = 0
        b_value = 0
        a_wins = 0
        b_wins = 0
        a_win_next_turn_threats = set()
        b_win_next_turn_threats = set()
        a_win_next_next_turn_threats = set()
        b_win_next_next_turn_threats = set()
        a_win_next_next_next_turn_threats = set()
        b_win_next_next_next_turn_threats = set()
        a_k_minus_1_threats = set()
        b_k_minus_1_threats = set()
        a_k_minus_2_threats = set()
        b_k_minus_2_threats = set()
        a_overlapping_k_minus_2_threats = set()
        b_overlapping_k_minus_2_threats = set()

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
                    last_free_squares = [None, None, None]
                    a_count = 0
                    b_count = 0
                    a_blocked = False
                    b_blocked = False

                    """
                    Count A's and B's in k-in-a-row-squares starting from (i,j), stepping in direction (x,y)
                    If blocked by an opposing piece, no way to get k-in-a-row = no added value
                    """
                    for square in range(k):

                        try:
                            current_piece = state.board[row][col]
                            if current_piece == game.BLOCK_PIECE:
                                a_blocked = True
                                b_blocked = True
                                break
                            elif current_piece == a_piece:
                                a_count += 1
                                b_blocked = True
                            elif current_piece == b_piece:
                                b_count += 1
                                a_blocked = True
                            else:
                                last_free_squares[2] = last_free_squares[1]
                                last_free_squares[1] = last_free_squares[0]
                                last_free_squares[0] = (row, col)
                            row += y
                            col += x
                        except IndexError:
                            a_blocked = True
                            b_blocked = True
                            break

                    if a_count > 0 and not a_blocked:
                        a_value, a_wins = process_count(k, last_free_squares, a_count, a_value, a_wins,
                                                        a_win_next_turn_threats, a_k_minus_1_threats,
                                                        a_k_minus_2_threats)

                    elif b_count > 0 and not b_blocked:
                        b_value, b_wins = process_count(k, last_free_squares, b_count, b_value, b_wins,
                                                        b_win_next_turn_threats, b_k_minus_1_threats,
                                                        b_k_minus_2_threats)

        detect_threats(a_k_minus_1_threats, a_k_minus_2_threats, a_overlapping_k_minus_2_threats,
                       a_win_next_next_turn_threats, a_win_next_next_next_turn_threats)

        detect_threats(b_k_minus_1_threats, b_k_minus_2_threats, b_overlapping_k_minus_2_threats,
                       b_win_next_next_turn_threats, b_win_next_next_next_turn_threats)

        a_win_value = a_sign * win_value * (1 + a_value / 10 ** k)
        b_win_value = b_sign * win_value * (1 + b_value / 10 ** k)

        """
        ===============================================
        ====== A TO PLAY NEXT, B HAS JUST PLAYED ======
        ===============================================
                    Predict any forced wins             
        """
        if a_wins >= 1:
            """
            A won on the previous turn
            A wins in 0 moves
            """
            value = a_win_value / 10 ** 0
        elif b_wins >= 1:
            """
            B has just won on this turn
            B wins in 1 move (B)
            """
            value = b_win_value / 10 ** 1
        elif len(a_win_next_turn_threats) >= 1:
            """
            A will win on the next turn
            A wins in 2 moves (BA)
            """
            value = a_win_value / 10 ** 2
        elif len(b_win_next_turn_threats) >= 2:
            """
            Multiple ways for B to win, A can only block 1 on their turn, B wins next turn
            B wins in 3 moves (BAB)
            """
            value = b_win_value / 10 ** 3
        else:
            """
            At this point:
                len(a_wins) = 0
                len(b_wins) = 0
                len(a_win_next_turn_threats) = 0
                len(b_win_next_turn_threats) <= 1
            """

            """Fast forward through any series of forced moves (ff)"""
            fast_forward_counter = 0
            while True:
                if len(b_win_next_turn_threats) == 1:
                    blocking_square = b_win_next_turn_threats.pop()
                    state = state.make_move(blocking_square)
                    fast_forward_counter += 1

                    update_threats(blocking_square, a_k_minus_1_threats, a_k_minus_2_threats, a_win_next_turn_threats,
                                   a_win_next_next_turn_threats, a_win_next_next_next_turn_threats)

                    """Search in the vicinity of the blocking square for new threats and any updates to the value"""
                    for (x, y) in directions:
                        for scan in range(1 - k, 1):
                            i = blocking_square[0] + scan * y
                            j = blocking_square[1] + scan * x
                            row = i
                            col = j
                            last_free_squares = [None, None, None]
                            a_count = 0
                            b_count = 0
                            a_blocked = False
                            b_blocked = False
                            for square in range(k):
                                if row < 0 or row > h - 1 or col < 0 or col > w - 1:
                                    a_blocked = True
                                    b_blocked = True
                                    break

                                try:
                                    current_piece = state.board[row][col]
                                    if current_piece == game.BLOCK_PIECE:
                                        a_blocked = True
                                        b_blocked = True
                                        break
                                    elif current_piece == b_piece:
                                        b_count += 1
                                        a_blocked = True
                                    elif current_piece == a_piece:
                                        a_count += 1
                                        if (row, col) != blocking_square:
                                            b_blocked = True
                                    else:
                                        last_free_squares[2] = last_free_squares[1]
                                        last_free_squares[1] = last_free_squares[0]
                                        last_free_squares[0] = (row, col)
                                    row += y
                                    col += x
                                except IndexError:
                                    a_blocked = True
                                    b_blocked = True
                                    break

                            if b_count > 0 and not b_blocked:
                                b_value -= 10.0 ** (b_count - 1)
                            if a_count > 0 and not a_blocked:
                                a_value, a_wins = process_count(k, last_free_squares, a_count, a_value, a_wins,
                                                                a_win_next_turn_threats, a_k_minus_1_threats,
                                                                a_k_minus_2_threats)

                    detect_threats(a_k_minus_1_threats, a_k_minus_2_threats, a_overlapping_k_minus_2_threats,
                                   a_win_next_next_turn_threats, a_win_next_next_next_turn_threats)

                    block_threats(blocking_square, b_k_minus_1_threats, b_k_minus_2_threats, b_win_next_turn_threats,
                                  b_win_next_next_turn_threats, b_win_next_next_next_turn_threats)

                    a_win_value = a_sign * win_value * (1 + a_value / 10 ** k)
                    b_win_value = b_sign * win_value * (1 + b_value / 10 ** k)

                    """
                    Swap A and B; B has just played, A to play next
                    """
                    a_piece, b_piece = b_piece, a_piece
                    a_sign, b_sign = b_sign, a_sign
                    a_value, b_value = b_value, a_value
                    a_wins, b_wins = b_wins, a_wins
                    a_win_next_turn_threats, b_win_next_turn_threats = b_win_next_turn_threats, a_win_next_turn_threats
                    a_win_next_next_turn_threats, b_win_next_next_turn_threats = (
                        b_win_next_next_turn_threats, a_win_next_next_turn_threats)
                    a_win_next_next_next_turn_threats, b_win_next_next_next_turn_threats = (
                        b_win_next_next_next_turn_threats, a_win_next_next_next_turn_threats)
                    a_k_minus_1_threats, b_k_minus_1_threats = b_k_minus_1_threats, a_k_minus_1_threats
                    a_k_minus_2_threats, b_k_minus_2_threats = b_k_minus_2_threats, a_k_minus_2_threats
                    a_overlapping_k_minus_2_threats, b_overlapping_k_minus_2_threats = (
                        b_overlapping_k_minus_2_threats, a_overlapping_k_minus_2_threats)
                    a_win_value, b_win_value = b_win_value, a_win_value

                else:
                    if len(b_win_next_turn_threats) > 1:
                        """
                        B has multiple wins on their next turn,
                        A can only block one of them, B wins on the following turn
                        B wins in 3 moves after fast forward (ff-BAB)         
                        """
                        value = round(b_win_value / 10 ** (3 + fast_forward_counter))

                    break

            """
            At this point:
                len(a_wins) = 0
                len(b_wins) = 0
                len(a_win_next_turn_threats) = 0
                len(b_win_next_turn_threats) = 0
            """

            if value == 0 and len(a_win_next_next_turn_threats) >= 1:
                """
                A can play a piece such that they have multiple wins on their next turn,
                B can only block one of them, A wins on the following turn
                A wins in 4 moves after fast forward (ff-BABA)
                """
                value = round(a_win_value / 10 ** (4 + fast_forward_counter))

            """
            At this point:
                len(a_wins) = 0
                len(b_wins) = 0
                len(a_win_next_turn_threats) = 0
                len(b_win_next_turn_threats) = 0
                len(a_win_next_next_turn_threats) = 0
            """

            if value == 0 and len(b_win_next_next_turn_threats) >= 1 and len(a_k_minus_1_threats) == 0:
                found_win = False
                for w1 in b_win_next_next_turn_threats:
                    if found_win:
                        break
                    for w2 in b_win_next_next_turn_threats.difference({w1}):
                        if w1[0] != w2[0] and w1[0] not in w2[1] and w2[0] not in w1[1] and set(w1[1]).isdisjoint(
                                set(w2[1])):
                            found_win = True
                            break
                if found_win:
                    """
                    B can play a piece such that they have multiple wins on their next next turn,
                    A cannot block all of them, B wins in the end
                    B wins in 5 moves after fast forward (ff-BABAB)
                    
                    Example:
                    
                    |   |   |   |   |   |   |       |   |   |   |   |   |   |       |   |   |   |   |   |   |
                    |   | B | B |[B]|   |   |       |   | B | B | B |[X]|   |       |   | B | B | B | X |   |
                    |   |   |   | B |   |   |  -->  |   |   |   | B |   |   |  -->  |   |   |   | B |   |   |
                    |   |   |   | B |   |   |  -->  |   |   |   | B |   |   |  -->  |   |   |   | B |   |   |
                    |   |   |   |   |   |   |       |   |   |   |   |   |   |       |   |   |   |[B]|   |   |
                    |   |   |   |   |   |   |       |   |   |   |   |   |   |       |   |   |   |   |   |   |
                       (B has just played)
                                                    |   |   |   |   |   |   |       |   |   |   |[B]|   |   |
                                                    |   | B | B | B | X |   |       |   | B | B | B | X |   |
                                               -->  |   |   |   | B |   |   |  -->  |   |   |   | B |   |   |
                                               -->  |   |   |   | B |   |   |  -->  |   |   |   | B |   |   |
                                                    |   |   |   | B |   |   |       |   |   |   | B |   |   |
                                                    |   |   |   |[X]|   |   |       |   |   |   | X |   |   |
                    
                    """
                    value = round(b_win_value / 10 ** (5 + fast_forward_counter))

            if value == 0 and len(a_win_next_next_next_turn_threats) >= 1:
                found_win = False
                if len(b_k_minus_1_threats) == 0:
                    found_win = True
                elif len(a_k_minus_1_threats) >= 1:
                    b_k_minus_1_threat_squares = {}
                    for t1 in b_k_minus_1_threats:
                        b_k_minus_1_threat_squares[t1[0]] = t1[1]
                        b_k_minus_1_threat_squares[t1[1]] = t1[0]
                    for t1 in a_k_minus_1_threats:
                        if found_win:
                            break
                        for w1 in a_win_next_next_next_turn_threats:
                            if found_win:
                                break
                            if w1[0] in t1 or (t1[0] == w1[1][0][0] or (len(w1[1]) == 2 and t1[1] == w1[1][1][0])):
                                b_forced_square = t1[1] if w1[0] == t1[0] else t1[0]
                                next_a_squares = set()
                                for w2 in w1[1]:
                                    next_a_squares.add(w2[0])
                                if (b_forced_square not in b_k_minus_1_threat_squares or
                                        b_k_minus_1_threat_squares[b_forced_square] in next_a_squares):
                                    found_win = True

                if found_win:
                    """
                    A can play a piece such that they have multiple wins on their next next turn,
                    B cannot block all of them, A wins in the end
                    A wins in 6 moves after fast forward (ff-BABABA)
                    
                    Example:
                    
                    |   | A | A | A |[B]|   |       |   | A | A | A | B |   |       |   | A | A | A | B |   |
                    |   |   |   |   |   |   |       |   |   |   |   |   |   |       |   |   |   |   |[B]|   |
                    |   |   |   | A |   |   |  -->  |   |   |   | A |   |   |  -->  |   |   |   | A |   |   |
                    |   |   | A |   |   |   |  -->  |   |   | A |   |   |   |  -->  |   |   | A |   |   |   |
                    |   |   | A | A |   |   |       |   |[A]| A | A |   |   |       |   | A | A | A |   |   |
                    |   |   |   |   |   |   |       |   |   |   |   |   |   |       |   |   |   |   |   |   |
                       (B has just played)
                                                    |   | A | A | A | B |   |       |   | A | A | A | B |   |
                                                    |   |   |   |   | B |   |       |   |   |   |   | B |   |
                                               -->  |   |   |   | A |   |   |  -->  |   |   |   | A |   |   |
                                               -->  |   |   | A |   |   |   |  -->  |   |   | A |   |   |   |
                                                    |   | A | A | A |[A]|   |       |[B]| A | A | A | A |   |
                                                    |   |   |   |   |   |   |       |   |   |   |   |   |   |
                                                    
                                                    |   | A | A | A | B |   |
                                                    |   |   |   |   | B |   |
                                               -->  |   |   |   | A |   |   |
                                               -->  |   |   | A |   |   |   |
                                                    | B | A | A | A | A |[A]|
                                                    |   |   |   |   |   |   |
                    
                    """
                    value = round(a_win_value / 10 ** (6 + fast_forward_counter))

            if value == 0 and len(b_win_next_next_next_turn_threats) >= 1 and len(a_k_minus_1_threats) == 0:

                found_win = False

                for w1 in b_win_next_next_turn_threats:
                    if found_win:
                        break
                    for w2 in b_win_next_next_next_turn_threats:
                        if found_win:
                            break
                        w2_squares = get_squares_from_threat(w2)
                        if w1[0] != w2[0] and w1[0] not in w2_squares and w2[0] not in w1[1] and set(w1[1]).isdisjoint(
                                w2_squares):
                            found_win = True
                            for t1 in a_k_minus_2_threats:
                                if w1[0] in t1 or not set(w1[1]).isdisjoint(t1):
                                    found_win = False

                if not found_win and len(b_win_next_next_next_turn_threats) >= 2:
                    completed = set()
                    for w1 in b_win_next_next_next_turn_threats:
                        completed.add(w1)
                        if found_win:
                            break
                        for w2 in b_win_next_next_next_turn_threats.difference(completed):
                            if found_win:
                                break
                            w1_squares = get_squares_from_threat(w1)
                            w2_squares = get_squares_from_threat(w2)
                            if (len(w1[1]) == 2 or len(w2[1]) == 2) and len(a_k_minus_2_threats) != 0:
                                break
                            if (w1[0] != w2[0] and
                                    w1[0] not in w2_squares and
                                    w2[0] not in w1_squares and
                                    w1_squares.isdisjoint(w2_squares)):
                                found_win = True

                if found_win:
                    """
                    B can play a piece such that they have at least four wins on their next next turn,
                    A can only block two of them, B wins in the end
                    B wins in 7 moves after fast forward (ff-BABABAB)
                    
                    Example:
                    
                    |   |   |   |   |   |   |       |   |   |   |   |   |   |       |   |   |   |   |   |   |
                    |   |   | B |[B]|   |   |       |   |   | B | B |[A]|   |       |   |   | B | B | A |   |
                    |   | B |   |   | B |   |  -->  |   | B |   |   | B |   |  -->  |   | B |   |   | B |   |
                    |   | B |   |   | B |   |  -->  |   | B |   |   | B |   |  -->  |   | B |   |   | B |   |
                    |   |   | B | B |   |   |       |   |   | B | B |   |   |       |   |[B]| B | B |   |   |
                    |   |   |   |   |   |   |       |   |   |   |   |   |   |       |   |   |   |   |   |   |
                       (B has just played)
                                                    |   |   |   |   |   |   |       |   |   |   |   |   |   |
                                                    |   |   | B | B | A |   |       |   |[B]| B | B | A |   |
                                               -->  |   | B |   |   | B |   |  -->  |   | B |   |   | B |   |
                                               -->  |   | B |   |   | B |   |  -->  |   | B |   |   | B |   |
                                                    |   | B | B | B |[A]|   |       |   | B | B | B | A |   |
                                                    |   |   |   |   |   |   |       |   |   |   |   |   |   |
                    
                                                    |   |[A]|   |   |   |   |       |   | A |   |   |   |   |
                                                    |   | B | B | B | A |   |       |   | B | B | B | A |   |
                                               -->  |   | B |   |   | B |   |  -->  |   | B |   |   | B |   |
                                               -->  |   | B |   |   | B |   |  -->  |   | B |   |   | B |   |
                                                    |   | B | B | B | A |   |       |   | B | B | B | A |   |
                                                    |   |   |   |   |   |   |       |   |[B]|   |   |   |   |
                    """
                    value = round(b_win_value / 10 ** (7 + fast_forward_counter))

        """No forced wins detected"""
        if value == 0:
            value = round(a_sign * a_value + b_sign * b_value)

        return value
