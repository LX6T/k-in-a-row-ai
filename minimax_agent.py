"""
minimax_agent.py
author: <YOUR NAME(s) HERE>
"""
from math import log10
import agent
import game
import time
import random
import logging

POW10 = [10 ** x for x in range(0, 30)]

def update_wins_and_threats(k, last_free_squares, count, value, wins,
                  win_next_turn_threats, k_minus_1_threats, k_minus_2_threats):
    """
    Update the total value and record any k-rows close to being filled
    """

    """An unblocked k-row filled with x squares counts for 10^(x-1) value"""
    value += POW10[count - 1]
    if count == k:
        """k out of k squares are filled, game over"""
        return value, wins + 1

    if count == k - 1:
        """k-1 out of k squares are filled, player can win next turn"""
        win_next_turn_threats.add(last_free_squares[0])
        return value, wins
    
    if count == k - 2:
        """k-2 out of k squares are filled, player can fill to k-1 next turn"""
        k_minus_1_threats.add((last_free_squares[0], last_free_squares[1]))
        return value, wins
    
    if count == k - 3:
        """k-3 out of k squares are filled, player can fill to k-2 next turn"""
        k_minus_2_threats.add((last_free_squares[0], last_free_squares[1], last_free_squares[2]))
    
    return value, wins

def process_wins_and_threats(state, h, w, k, directions, a_piece, b_piece,
                            a_win_next_turn_threats, b_win_next_turn_threats,
                            a_k_minus_1_threats, b_k_minus_1_threats,
                            a_k_minus_2_threats, b_k_minus_2_threats):

    board = state.board
    BLOCK = game.BLOCK_PIECE

    a_value = b_value = 0
    a_wins = b_wins = 0

    for dx, dy in directions:

        min_row = 0
        min_col = k - 1 if dx == -1 else 0
        max_row = w - k if dy == 1 else w - 1
        max_col = h - k if dx == 1 else h - 1

        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):

                row, col = i, j
                a_count = b_count = 0
                a_blocked = b_blocked = False
                last0 = last1 = last2 = None

                for _ in range(k):
                    piece = board[row][col]

                    if piece == BLOCK:
                        a_blocked = b_blocked = True
                        break

                    if piece == a_piece:
                        a_count += 1
                        b_blocked = True
                    elif piece == b_piece:
                        b_count += 1
                        a_blocked = True
                    else:
                        last0, last1, last2 = (row, col), last0, last1

                    if a_blocked and b_blocked:
                        break

                    row += dy
                    col += dx

                if a_count and not a_blocked:
                    a_value, a_wins = update_wins_and_threats(
                        k, (last0, last1, last2), a_count,
                        a_value, a_wins,
                        a_win_next_turn_threats,
                        a_k_minus_1_threats,
                        a_k_minus_2_threats
                    )

                elif b_count and not b_blocked:
                    b_value, b_wins = update_wins_and_threats(
                        k, (last0, last1, last2), b_count,
                        b_value, b_wins,
                        b_win_next_turn_threats,
                        b_k_minus_1_threats,
                        b_k_minus_2_threats
                    )

    return a_value, b_value, a_wins, b_wins

def detect_threats(k_minus_1_threats, k_minus_2_threats, overlapping_k_minus_2_threats,
                   win_next_next_turn_threats, win_next_next_next_turn_threats):

    # Preconvert threats to (tuple, frozenset)
    k1 = [(t, frozenset(t)) for t in k_minus_1_threats]
    k2 = [(t, frozenset(t)) for t in k_minus_2_threats]
    ok2 = list(overlapping_k_minus_2_threats)

    """
    Record places where two win threats can be created on the same turn
    """
    n = len(k1)
    for i in range(n):
        t1, s1 = k1[i]
        for j in range(i + 1, n):
            t2, s2 = k1[j]

            inter = s1 & s2
            if len(inter) == 1:
                overlap = next(iter(inter))
                win_next_turn = tuple((s1 ^ s2))
                win_next_next_turn_threats.add((overlap, win_next_turn))

    """
    Record places where two parallel k-2 threats can be created on the same turn
    """
    n = len(k2)
    for i in range(n):
        t1, s1 = k2[i]
        for j in range(i + 1, n):
            t2, s2 = k2[j]

            inter = s1 & s2
            if len(inter) == 2:
                overlapping_squares = tuple(inter)
                edge_squares = tuple(s1 ^ s2)
                overlapping_k_minus_2_threats.add((overlapping_squares, edge_squares))

    """
    Higher order overlapping logic
    """
    ok2 = list(overlapping_k_minus_2_threats)
    m = len(ok2)

    for i in range(m):
        squares1, edges1 = ok2[i]
        s1 = frozenset(squares1)

        """
        Record places where two sets of two parallel k-2 threats can be created on the same turn
        """
        for j in range(i + 1, m):
            squares2, edges2 = ok2[j]
            s2 = frozenset(squares2)

            inter = s1 & s2
            if len(inter) == 1 and not (s1 & frozenset(edges2)):
                overlap = next(iter(inter))
                a = next(iter(s1 - {overlap}))
                b = next(iter(s2 - {overlap}))

                ref = ((a, edges1), (b, edges2))
                win_next_next_next_turn_threats.add((overlap, ref))

        """
        Record places where two parallel k-2 threats and a k-1 threat can be created on the same turn
        """
        for t2, s2 in k1:
            inter = s1 & s2
            if len(inter) == 1:
                overlap = next(iter(inter))
                a = next(iter(s1 - {overlap}))
                forced = t2[1] if overlap == t2[0] else t2[0]

                ref = ((a, edges1, t2),)
                if forced != a and forced not in edges1:
                    win_next_next_next_turn_threats.add((overlap, ref))

def update_threats(blocking_square, k_minus_1_threats, k_minus_2_threats, win_next_turn_threats,
                   win_next_next_turn_threats, win_next_next_next_turn_threats):
    """
    Record any new threats resulting from blocking a win
    """

    to_remove = []
    for t in k_minus_1_threats:
        if blocking_square in t:
            for sq in t:
                if sq != blocking_square:
                    win_next_turn_threats.add(sq)
            to_remove.append(t)
    if to_remove:
        k_minus_1_threats.difference_update(to_remove)

    to_remove = []
    for w in win_next_next_turn_threats:
        root = w[0]
        children = w[1] if len(w) < 3 else set(w[1]) | set(w[2])
        if blocking_square == root:
            win_next_turn_threats.update(children)
            to_remove.append(w)
            continue
        for sq in children:
            if blocking_square == sq:
                win_next_turn_threats.add(root)
                to_remove.append(w)
                break
    if to_remove:
        win_next_next_turn_threats.difference_update(to_remove)

    to_remove = []
    for t in k_minus_2_threats:
        if blocking_square in t:
            if t[0] == blocking_square:
                k_minus_1_threats.add((t[1], t[2]))
            elif t[1] == blocking_square:
                k_minus_1_threats.add((t[0], t[2]))
            else:
                k_minus_1_threats.add((t[0], t[1]))
            to_remove.append(t)
    if to_remove:
        k_minus_2_threats.difference_update(to_remove)

    to_remove = []
    for w in win_next_next_next_turn_threats:
        root, refs = w
        if blocking_square == root:
            win_next_next_turn_threats.update(refs)
            to_remove.append(w)
            continue
        for ref in refs:
            if blocking_square == ref[0]:
                win_next_next_turn_threats.add(ref)
                to_remove.append(w)
                break
    if to_remove:
        win_next_next_next_turn_threats.difference_update(to_remove)

def block_threats(blocking_square, k_minus_1_threats, k_minus_2_threats,
                  win_next_next_turn_threats, win_next_next_next_turn_threats):
    """
    If blocking a win also blocks any other threats, remove them from storage.
    """

    to_remove = []
    for t in k_minus_1_threats:
        if blocking_square in t:
            to_remove.append(t)
    if to_remove:
        k_minus_1_threats.difference_update(to_remove)

    to_remove = []
    for w in win_next_next_turn_threats:
        root = w[0]
        children = w[1] if len(w) < 3 else set(w[1]) | set(w[2])
        if blocking_square == root or blocking_square in children:
            to_remove.append(w)
    if to_remove:
        win_next_next_turn_threats.difference_update(to_remove)

    to_remove = []
    for t in k_minus_2_threats:
        if blocking_square in t:
            to_remove.append(t)
            continue
        if blocking_square in t[0] or blocking_square in t[1]:
            to_remove.append(t)
    if to_remove:
        k_minus_2_threats.difference_update(to_remove)

    to_remove = []
    for w in win_next_next_next_turn_threats:
        root, refs = w
        if blocking_square == root:
            to_remove.append(w)
            continue
        for ref in refs:
            root = ref[0]
            children = ref[1] if len(ref) < 3 else set(ref[1]) | set(ref[2])
            if blocking_square == root or blocking_square in children:
                to_remove.append(w)
                break
    if to_remove:
        win_next_next_next_turn_threats.difference_update(to_remove)

def get_squares_from_threat(threat):
    """
    Helper method to deconstruct an element of win_next_next_next_turn_threats
    """
    refs = threat[1]
    first = refs[0]

    squares = {first[0]}
    squares.update(first[1])

    if len(first) == 3:
        squares.update(first[2])

    if len(refs) == 2:
        second = refs[1]
        squares.add(second[0])
        squares.update(second[1])

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
        max_depth = 0
        for i in range(w):
            for j in range(h):
                if state.board[i][j] == game.EMPTY_PIECE:
                    max_depth += 1  # Count empty squares while finding default move
                    if best_move is None:
                        best_move = (i, j)

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
            latest_time_limit = (timeout - time.perf_counter()) if timeout is not None else None

            try:
                move, value, fff = self.minimax(state, depth, latest_time_limit, float("-inf"), float("inf"), z_hashing)
            except Exception as ex:
                logging.error(ex, exc_info=True)

            if time_limit is None or time.perf_counter() < (timeout - self.wrapup_time):

                """Full search complete, update best_move"""
                best_move = move
                best_value = value
                best_fff = fff
                if not self.silent:
                    print(f"depth={depth}, best_move={best_move}, best_value={best_value}, best_fff={best_fff}")

                """Guaranteed to win, stop search"""
                if (self.piece == game.X_PIECE and best_value >= POW10[k] or
                        self.piece == game.O_PIECE and best_value <= -POW10[k]):
                    if not self.silent:
                        print(f"Win found in {round(k + 20 - log10(abs(best_value))) + depth + best_fff} moves")
                    break
                elif (self.piece == game.X_PIECE and best_value <= -POW10[k] or
                    self.piece == game.O_PIECE and best_value >= POW10[k]):
                    if not self.silent:
                        print(f"Loss found in {round(k + 20 - log10(abs(best_value))) + depth + best_fff} moves")
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
            print("_" * 50)
            print()

        return best_move

    def get_moves_to_search(self, state, h, w, k, a_piece, b_piece):
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]

        a_win_next_turn_threats = set()
        b_win_next_turn_threats = set()
        a_k_minus_1_threats = set()
        b_k_minus_1_threats = set()
        a_k_minus_2_threats = set()
        b_k_minus_2_threats = set()

        a_win_next_next_turn_threats = set()
        b_win_next_next_turn_threats = set()
        a_win_next_next_next_turn_threats = set()
        b_win_next_next_next_turn_threats = set()
        a_overlapping_k_minus_2_threats = set()
        b_overlapping_k_minus_2_threats = set()

        process_wins_and_threats(state, h, w, k, directions, a_piece, b_piece, 
                                a_win_next_turn_threats, b_win_next_turn_threats,
                                a_k_minus_1_threats, b_k_minus_1_threats, 
                                a_k_minus_2_threats, b_k_minus_2_threats)

        detect_threats(a_k_minus_1_threats, a_k_minus_2_threats, a_overlapping_k_minus_2_threats,
                    a_win_next_next_turn_threats, a_win_next_next_next_turn_threats)

        detect_threats(b_k_minus_1_threats, b_k_minus_2_threats, b_overlapping_k_minus_2_threats,
                    b_win_next_next_turn_threats, b_win_next_next_next_turn_threats)

        move_set = set()
        moves_to_search = []
        win = None

        if a_win_next_turn_threats:
            win = next(iter(a_win_next_turn_threats))
        elif b_win_next_turn_threats:
            moves_to_search = list(b_win_next_turn_threats)
        elif a_win_next_next_turn_threats:
            win = next(iter(a_win_next_next_turn_threats))[0]
        elif b_win_next_next_turn_threats:
            for x in b_win_next_next_turn_threats:
                move_set.add(x[0])
                move_set.update(x[1])
            for t in a_k_minus_1_threats:
                move_set.update(t)
            moves_to_search = list(move_set)
        elif a_win_next_next_next_turn_threats:
            win = next(iter(a_win_next_next_next_turn_threats))[0]
        elif b_win_next_next_next_turn_threats:
            for b_win_nnnt_t in b_win_next_next_next_turn_threats:
                overlapping_square, b_win_nnt_t_ref = b_win_nnnt_t
                move_set.add(overlapping_square)
                for ref in b_win_nnt_t_ref:
                    move_set.add(ref[0])
                    move_set.update(ref[1])
                    if len(ref) == 3:
                        move_set.update(ref[2])
            moves_to_search = list(move_set)
        else:
            moves_to_search = [(i, j) for i in range(w) for j in range(h) if state.board[i][j] == game.EMPTY_PIECE]

        return moves_to_search, win

    def minimax(self, state: game.GameState, depth_remaining: int, time_limit: float = None,
                alpha: float = None, beta: float = None, z_hashing=None, 
                ff: int = 0, ff_branch_max: int = 10) -> ((int, int), float):
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

        """Unpack Zobrist hash table"""
        (z_table, z_memory, z_key) = z_hashing if z_hashing else (None, None, None)

        if time_limit is not None and time_limit < self.wrapup_time:
            """Exit early if reached time limit"""
            return None, None, None
        elif depth_remaining == 0:
            """Return static evaluation if reached depth limit or game over"""
            value = self.static_eval(state)
            if z_memory is not None:
                z_memory[z_key] = (None, value, ff)
            return None, value, ff
        else:
            """Otherwise do minimax"""
            timeout = None
            if time_limit is not None:
                timeout = time.perf_counter() + time_limit

            h = state.h
            w = state.w
            k = state.k
            a_piece = state.next_player
            b_piece = game.X_PIECE if a_piece == game.O_PIECE else game.O_PIECE
            depth_factor = POW10[depth_remaining]

            """Generate forced/preferred moves if they exist, or else all currently valid moves"""
            moves_to_search, win = self.get_moves_to_search(state, h, w, k, a_piece, b_piece)

            """If board is full or game is won, stop searching"""
            if len(moves_to_search) == 0:
                move = win if win else None
                value = self.static_eval(state) / depth_factor * POW10[2]
                if z_memory is not None:
                    z_memory[z_key] = (move, value, ff)
                return move, value, ff

            """Initialise best move state"""
            best_move = None
            best_value = float("-inf") if a_piece == game.X_PIECE else float("inf")
            best_fff = ff
            can_fast_forward = len(moves_to_search) <= ff_branch_max
            z_index = 0 if a_piece == game.X_PIECE else 1

            """Search"""
            for move in moves_to_search:
                i, j = move
                current_time = time.perf_counter()

                """Iterate until all spaces have been tried, exit early if time limit is reached"""
                if timeout and current_time >= timeout - self.wrapup_time:
                    break

                """Play A in square (i,j), update Zobrist hash"""
                new_state = state.make_move(move)
                new_z_key = z_key ^ z_table[i * h + j][z_index] if z_hashing else None

                """Find the value of the new state"""
                if new_z_key and new_z_key in z_memory:
                    _, value, fff = z_memory[new_z_key]
                else:
                    new_time_limit = timeout - current_time if timeout is not None else None
                    new_depth_remaining = depth_remaining if can_fast_forward else depth_remaining - 1
                    new_ff = ff + 1 if can_fast_forward else ff
                    _, value, fff = self.minimax(new_state, new_depth_remaining, new_time_limit,
                                                alpha, beta, (z_table, z_memory, new_z_key), new_ff)

                """Exit early if reached time limit"""
                if value is None:
                    break

                """Update best move, alpha and beta"""
                if a_piece == game.X_PIECE:
                    if value > best_value:
                        best_move, best_value, best_fff = move, value, fff
                    if beta is not None and best_value > beta:
                        return best_move, best_value, best_fff
                    if alpha is not None:
                        alpha = max(alpha, best_value)
                else:
                    if value < best_value:
                        best_move, best_value, best_fff = move, value, fff
                    if alpha is not None and best_value < alpha:
                        return best_move, best_value, best_fff
                    if beta is not None:
                        beta = min(beta, best_value)

            """Store result in Zobrist hash table"""
            if z_hashing is not None:
                z_memory[z_key] = (best_move, best_value, best_fff)

            return best_move, best_value, best_fff

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
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]

        win_value = POW10[k + 21]

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

        a_value, b_value, a_wins, b_wins = process_wins_and_threats(state, h, w, k, directions, a_piece, b_piece, 
                                                            a_win_next_turn_threats, b_win_next_turn_threats,
                                                            a_k_minus_1_threats, b_k_minus_1_threats, 
                                                            a_k_minus_2_threats, b_k_minus_2_threats)

        detect_threats(a_k_minus_1_threats, a_k_minus_2_threats, a_overlapping_k_minus_2_threats,
                       a_win_next_next_turn_threats, a_win_next_next_next_turn_threats)

        detect_threats(b_k_minus_1_threats, b_k_minus_2_threats, b_overlapping_k_minus_2_threats,
                       b_win_next_next_turn_threats, b_win_next_next_next_turn_threats)

        a_win_value = a_sign * win_value * (1 + a_value / POW10[k])
        b_win_value = b_sign * win_value * (1 + b_value / POW10[k])

        value = self.calculate_value(state, h, w, k, directions, win_value, 
                                     a_piece, b_piece, a_sign, b_sign, a_value, b_value, a_wins, b_wins, 
                                     a_win_next_turn_threats, b_win_next_turn_threats, 
                                     a_win_next_next_turn_threats, b_win_next_next_turn_threats, 
                                     a_win_next_next_next_turn_threats, b_win_next_next_next_turn_threats, 
                                     a_k_minus_1_threats, b_k_minus_1_threats, 
                                     a_k_minus_2_threats, b_k_minus_2_threats, 
                                     a_overlapping_k_minus_2_threats, b_overlapping_k_minus_2_threats,
                                     a_win_value, b_win_value)

        return value

    def calculate_value(self, state, h, w, k, directions, win_value, 
                            a_piece, b_piece, a_sign, b_sign, a_value, b_value, a_wins, b_wins, 
                            a_win_next_turn_threats, b_win_next_turn_threats, 
                            a_win_next_next_turn_threats, b_win_next_next_turn_threats, 
                            a_win_next_next_next_turn_threats, b_win_next_next_next_turn_threats, 
                            a_k_minus_1_threats, b_k_minus_1_threats, 
                            a_k_minus_2_threats, b_k_minus_2_threats, 
                            a_overlapping_k_minus_2_threats, b_overlapping_k_minus_2_threats,
                            a_win_value, b_win_value):
        """
        ===============================================
        ====== A TO PLAY NEXT, B HAS JUST PLAYED ======
        ===============================================
                    Predict any forced wins             
        """
        value = 0

        if a_wins >= 1:
            """
            A won on the previous turn
            A wins in 0 moves
            """
            value = a_win_value / POW10[0]
        elif b_wins >= 1:
            """
            B has just won on this turn
            B wins in 1 move (B)
            """
            value = b_win_value / POW10[1]
        elif len(a_win_next_turn_threats) >= 1:
            """
            A will win on the next turn
            A wins in 2 moves (BA)
            """
            value = a_win_value / POW10[2]
        elif len(b_win_next_turn_threats) >= 2:
            """
            Multiple ways for B to win, A can only block 1 on their turn, B wins next turn
            B wins in 3 moves (BAB)
            """
            value = b_win_value / POW10[3]
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
                    bx, by = blocking_square
                    for (dx, dy) in directions:
                        for scan in range(1 - k, 1):
                            i = bx + scan * dx
                            j = by + scan * dy
                            last_free_squares = [None, None, None]
                            a_count = 0
                            b_count = 0
                            a_blocked = False
                            b_blocked = False

                            for step in range(k):
                                if not (0 <= i < w and 0 <= j < h):
                                    a_blocked = b_blocked = True
                                    break

                                current_piece = state.board[i][j]
                                if current_piece == game.BLOCK_PIECE:
                                    a_blocked = b_blocked = True
                                    break
                                elif current_piece == b_piece:
                                    b_count += 1
                                    a_blocked = True
                                elif current_piece == a_piece:
                                    a_count += 1
                                    if (i, j) != blocking_square:
                                        b_blocked = True
                                else:
                                    last_free_squares = [(i, j)] + last_free_squares[:2]

                                i += dx
                                j += dy

                            if b_count > 0 and not b_blocked:
                                b_value -= POW10[b_count - 1]
                            if a_count > 0 and not a_blocked:
                                a_value, a_wins = update_wins_and_threats(
                                    k, last_free_squares, a_count, a_value, a_wins,
                                    a_win_next_turn_threats, a_k_minus_1_threats, a_k_minus_2_threats
                                )

                    detect_threats(a_k_minus_1_threats, a_k_minus_2_threats, a_overlapping_k_minus_2_threats,
                                a_win_next_next_turn_threats, a_win_next_next_next_turn_threats)

                    block_threats(blocking_square, b_k_minus_1_threats, b_k_minus_2_threats,
                                b_win_next_next_turn_threats, b_win_next_next_next_turn_threats)

                    a_win_value = a_sign * win_value * (1 + a_value / POW10[k])
                    b_win_value = b_sign * win_value * (1 + b_value / POW10[k])

                    """
                    Swap A and B; B has just played, A to play next
                    """
                    (
                        a_piece, b_piece,
                        a_sign, b_sign,
                        a_value, b_value,
                        a_wins, b_wins,
                        a_win_next_turn_threats, b_win_next_turn_threats,
                        a_win_next_next_turn_threats, b_win_next_next_turn_threats,
                        a_win_next_next_next_turn_threats, b_win_next_next_next_turn_threats,
                        a_k_minus_1_threats, b_k_minus_1_threats,
                        a_k_minus_2_threats, b_k_minus_2_threats,
                        a_overlapping_k_minus_2_threats, b_overlapping_k_minus_2_threats,
                        a_win_value, b_win_value
                    ) = (
                        b_piece, a_piece,
                        b_sign, a_sign,
                        b_value, a_value,
                        b_wins, a_wins,
                        b_win_next_turn_threats, a_win_next_turn_threats,
                        b_win_next_next_turn_threats, a_win_next_next_turn_threats,
                        b_win_next_next_next_turn_threats, a_win_next_next_next_turn_threats,
                        b_k_minus_1_threats, a_k_minus_1_threats,
                        b_k_minus_2_threats, a_k_minus_2_threats,
                        b_overlapping_k_minus_2_threats, a_overlapping_k_minus_2_threats,
                        b_win_value, a_win_value
                    )

                else:
                    if len(b_win_next_turn_threats) > 1:
                        """
                        B has multiple wins on their next turn,
                        A can only block one of them, B wins on the following turn
                        B wins in 3 moves after fast forward (ff-BAB)         
                        """
                        value = round(b_win_value / POW10[3 + fast_forward_counter])
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
                value = round(a_win_value / POW10[4 + fast_forward_counter])

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
                    value = round(b_win_value / POW10[5 + fast_forward_counter])

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
                    value = round(a_win_value / POW10[6 + fast_forward_counter])

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
                    value = round(b_win_value / POW10[7 + fast_forward_counter])

        """No forced wins detected"""
        if value == 0:
            value = round(a_sign * a_value + b_sign * b_value)

        return value
