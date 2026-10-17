# MinimaxAgent Python vs C++ function-by-function comparison

This document compares the Python implementation in [minimax_agent.py](../minimax_agent.py) with the C++ port in [cpp_port/MinimaxAgent.cpp](MinimaxAgent.cpp). It is organized one function at a time and is intended to highlight both behavioral parity and format/API differences.

## 1) update_wins_and_threats

Python:
- Standalone function.
- Returns `(value, wins)`.
- Uses a tuple `last` such as `(last0, last1, last2, last3)` and allows `None` placeholders.
- Adds `POW10[count - 1]` to the value.
- Uses early returns to classify threat states:
  - `count == k` => win
  - `count == k - 1` => `win_next_turn_threats.add(last[0])`
  - `count == k - 2` => `k_1_threats.add((last[0], last[1]))`
  - `count == k - 3` => `k_2_threats.add((last[0], last[1], last[2]))`
  - `count == k - 4` => `k_3_threats.add((last[0], last[1], last[2], last[3]))`

C++:
- Method on `MinimaxAgent`.
- Returns `void`; it mutates `value`, `wins`, and the threat sets via references.
- Uses `std::vector<Coord> last(4, {-1,-1})` instead of Python `None` placeholders.
- Same core logic, but the sentinel checks are expressed as `if (last[0].first != -1 && ...)` rather than `if last[0] is not None`.

Key differences:
- Python is functional and returns data; C++ is stateful and mutates references.
- Same semantics, but C++ is more explicit about invalid sentinel values.

## 2) process_wins_and_threats

Python:
- Standalone function.
- Reads `state.board`, scans every direction, and walks each potential line of length `k`.
- Tracks `a_count`, `b_count`, `a_blocked`, `b_blocked`, and a rolling `last` tuple.
- Calls `update_wins_and_threats(...)` for whichever side is active and not blocked.
- Returns `(a_value, b_value, a_wins, b_wins)`.

C++:
- Method with the same conceptual logic, but it receives all the side-specific threat sets and value/win accumulators as reference parameters.
- Uses `const auto& board = state.board();` and loops with explicit bounds and `std::vector<Coord> last(4, {-1,-1})`.
- Same scanning pattern and same branch logic for handling blocked vs unblocked lines.

Key differences:
- Python uses a tuple-based last-tracking pattern; C++ uses a vector with sentinel values.
- C++ is more heavily parameterized and less “returning tuple” oriented.
- Functionally the same: both accumulate values and threat information from every row/column/diagonal line.

## 3) detect_threats

Python:
- Standalone function.
- Converts each threat tuple set to a `set` form (`k_1_sets`, `k_2_sets`, `k_3_sets`).
- Builds higher-order threat collections like:
  - `win_next_next_turn_threats`
  - `win_next_next_next_turn_threats`
  - `win_next_next_next_next_turn_threats`
- Uses a lot of set intersection/difference logic and nested loops to detect “two threats overlapping in a forced-win pattern”.

C++:
- Method on `MinimaxAgent` with typed `std::set` containers.
- Uses helper lambdas: `set_intersection`, `set_difference`, and `coord_to_vector`.
- Uses explicit `std::tuple`/nested `std::tuple` structures for the higher-order threat sets.
- Same conceptual detection passes, but with a more C++-idiomatic representation.

Key differences:
- Python uses plain Python tuples and set math; C++ uses `std::set` and tuple aliases.
- C++ has more explicit type names and a more rigid nested tuple structure.
- The logic is the same in spirit, but the C++ version is more verbose and type-heavy.

## 4) update_threats

Python:
- Standalone function.
- When a new square is played, it updates threat state by:
  - removing resolved `k_1_threats` that include the square,
  - converting `k_2_threats` into new `k_1_threats` when one square is played,
  - promoting certain `win_next_next...` entries into `win_next_turn_threats`,
  - removing stale entries after a move blocks or resolves a threat.
- Uses `to_remove` lists and `difference_update`.

C++:
- Method form with the same flow, but implemented as erase-while-iterating loops.
- Uses set membership checks like `if (std::get<0>(t) == new_square || ...)` and then rewrites the relevant threat sets.
- The overall state updates are equivalent, but the precise iteration pattern differs.

Key differences:
- Python is more compact and declarative with `difference_update`.
- C++ is more imperative and explicit, using erase loops.
- The logic is effectively the same: after a move, old threat structures are retired or promoted as needed.

## 5) block_threats

Python:
- Standalone function.
- Removes any threat sets containing the blocking square.
- Also removes any higher-order threat structures whose root or child set contains the blocking square.
- Has explicit `to_remove` patterns and calls `difference_update`.

C++:
- Same idea, but exact loop structure differs.
- It removes entries from `k_1_threats`, `k_2_threats`, `k_3_threats`, and the nested higher-order threats by checking `blocking_square == root`, `blocking_square == ref0`, and set membership on child sets.

Key differences:
- Python groups set cleanup patterns in a compact “remove all threats involving this square” style.
- C++ does the same but in a methodical, branch-by-branch way.
- No major semantic difference beyond structure.

## 6) get_squares_from_wn3_threat

Python:
- Standalone helper for entries in `win_next_next_next_turn_threats`.
- Takes a tuple like `(root, refs)` where `refs` is a list/tuple of nested `(square, tuple_of_children)` entries.
- Builds a set of all relevant squares by collecting the root and children from each reference.
- Handles a second reference if present.

C++:
- Similar helper, but it accepts a typed tuple and returns `std::set<Coord>`.
- It inserts the root and children into a set and optionally processes a second nested reference.

Key differences:
- Python uses tuple-unpacking and raw lists/tuples.
- C++ uses typed tuples and `std::set<Coord>` for final output.
- Same intent: flatten a nested threat representation into a set of square coordinates.

## 7) get_squares_from_wn4_threat

Python:
- Standalone helper for the four-step threat structure.
- Expands the root and the nested `wn3` tuple into a single square set.

C++:
- Equivalent helper with a typed `std::pair<Coord, std::vector<std::pair<Coord, std::vector<Coord>>>>` input and a `std::set<Coord>` output.

Key differences:
- Python uses tuple unpacking and recursive flattening through `get_squares_from_wn3_threat(refs)`.
- C++ does the same thing but in a strongly typed format.
- Overall behavior is equivalent.

## 8) MinimaxAgent.__init__

Python:
- Constructor of `MinimaxAgent` adds runtime tracking fields:
  - `self.eval_calls`
  - `self.wrapup_time`
  - `self.silent`
  - `self.init_ff_branch_max`
- It calls `super().__init__(initial_state, piece)` and initializes the search configuration.

C++:
- Constructor is far smaller.
- Stores only:
  - `piece_`
  - `max_depth_`
  - `seed_`

Key differences:
- Python is built for full gameplay/search instrumentation and optimization behavior.
- C++ is a lean game-player object without the Python debugging/search metadata.
- The C++ constructor is much less feature-rich, but it still matches the minimal state needed for move selection.

## 9) MinimaxAgent.introduce

Python:
- Returns a multi-line introduction string.
- Uses literal strings describing the bot’s identity and creation.

C++:
- No equivalent method in the C++ port.

Key differences:
- Python includes personality/UI text; C++ omits this entirely.
- This is a non-essential gameplay function and is not part of the game logic.

## 10) MinimaxAgent.nickname

Python:
- Returns a short name string: `"alpha_agent"`.

C++:
- No equivalent method in the C++ port.

Key differences:
- This is purely metadata/identity functionality, not search logic.
- It is absent in the stripped-down C++ version.

## 11) MinimaxAgent.choose_move

Python:
- Main move-selection routine for the game.
- Accepts `state` and `time_limit`.
- Performs iterative deepening with time checks.
- Uses `self.eval_calls`, `self.depth_counter`, `zobrist` hashing, `self.minimax`, and `self.search_moves`.
- Tries immediate wins, then opponent wins, then deeper search with pruning and threat prioritization.
- Includes verbose printouts when not silent.

C++:
- Much simpler `choose_move(const GameState& state, int depth) const`.
- Checks legal moves, immediately returns a winning move for the current player, then checks for opponent winning moves, then performs a limited-depth minimax loop.
- No time limit, no debugging printouts, no iterative deepening in the Python sense.
- The port does not replicate the Python’s full search scheduler, hashing, or profiling structure.

Key differences:
- This is the biggest behavioral difference in the whole port.
- Python is a full search engine with time budgeting and expanded forced-move logic.
- C++ is a streamlined move chooser, but it is not a direct functional clone of the Python implementation in runtime behavior or API shape.

## 12) MinimaxAgent.minimax

Python:
- Full recursive minimax routine.
- Accepts `time_limit`, `alpha`, `beta`, `z_hashing`, `ff`, and `ff_branch_max`.
- Uses `state.is_full()`, `state.winner()`, and `self.static_eval(state)` as terminal conditions.
- Tracks `depth_counter` and handles Zobrist memoization.
- Returns a 5-tuple: `(move, value, deadline, result, fff)`.

C++:
- Much simpler version.
- Returns only a `double` evaluation for a state.
- No time limit, no Zobrist structure, no `ff`/`ff_branch_max`, no tuple return payload.
- Uses direct alpha-beta minimax and a simpler legal move iteration.

Key differences:
- Python is a full tactical search evaluator with any-time behavior and metadata.
- C++ is a compact alpha-beta evaluator that preserves only the core value search.
- This is not a one-to-one API match.

## 13) MinimaxAgent.sort_remaining_moves

Python:
- Instance method.
- Builds `empty_moves` from the board and excludes `searched_moves`.
- Uses `self.full_squares` to find adjacent squares around occupied cells.
- Returns three objects:
  - `adjacent_1_moves`
  - `adjacent_2_moves`
  - `remaining_moves`

C++
- Equivalent functionality but as a method that fills three reference outputs:
  - `adjacent_1_moves`
  - `adjacent_2_moves`
  - `remaining_moves`
- It does not depend on class state like `self.full_squares`; instead it reads the board directly.

Key differences:
- Python returns a tuple; C++ mutates out-parameters.
- Python uses class state to know “all occupied squares”; C++ recomputes from the board.
- Semantically similar, but the object model differs.

## 14) MinimaxAgent.eval_move

Python:
- Updates the best move candidate while considering value ordering and result/deadline logic.
- Returns a tuple: `(alpha, beta, best_move, best_value, best_deadline, best_result, best_fff, stop_search)`.
- Distinguishes between `X_WIN` and `O_WIN` and respects the current player’s maximizing/minimizing perspective.
- Uses `BLOCK` as a result class and includes deadline comparisons.

C++:
- Returns `bool stop_search` only.
- Mutates `alpha`, `beta`, `best_move`, `best_value`, `best_deadline`, `best_result`, `best_fff` by reference.
- Uses integer result constants and a simpler comparison structure.

Key differences:
- Python is a more full-featured search control API with tuple returns and richer status tracking.
- C++ is streamlined and less verbose, but equivalent in basic pruning decision-making.

## 15) MinimaxAgent.search_moves

Python:
- Iterates over candidate moves and evaluates each with a recursive call to `self.minimax`.
- Uses time-limit checks, Zobrist hashing, `can_fast_forward`, and `ff_branch_max`.
- Manages `self.full_squares` while exploring.
- Returns `(best_move, best_value, best_deadline, best_result, best_fff, alpha, beta)`.

C++:
- Non-equivalent simplified version.
- Ignores most of the Python-specific scheduling parameters (`z_index`, `ff`, `ff_branch_max`, `h`, `can_fast_forward`).
- Evaluates moves using a direct call to `minimax` and then `eval_move`.

Key differences:
- Python explicitly implements advanced move ordering and time-aware exploration.
- C++ strips this down to the core alpha-beta/branch search loop.
- This is a real behavior reduction vs. the Python original.

## 16) MinimaxAgent.static_eval

Python:
- Full evaluation routine.
- Increments `self.eval_calls`.
- Inspects board and threat structure.
- Returns a 3-tuple: `(value, deadline, result)`.
- Front-loads quick-return checks such as:
  - immediate win by A,
  - immediate win by B,
  - `A` can win next turn,
  - `B` has multiple winning next-turn threats,
  - otherwise delegates to `calculate_value(...)`.

C++:
- Returns a single `double` value only; no `(value, deadline, result)` tuple.
- Uses the same conceptual pipeline but in a more compact form.
- Performs the same board scans and calls `calculate_value(...)` in the fall-through path.

Key differences:
- Python exposes more search metadata and early-return evaluation states.
- C++ reduces the result to a scalar score and does not keep deadline/result metadata.
- Functionally, the evaluation goal is the same, but the API and the information returned are different.

## 17) MinimaxAgent.calculate_value

Python:
- Most complex function in the file.
- Performs forced-move fast-forwarding, threat updates, swap-of-A/B roles, and return conditions for tactical win detection.
- Contains lots of explanatory comments and specific “end-state” cases.
- Handles scenarios like:
  - B has a single blocking move,
  - B has multiple immediate wins,
  - A can create multiple wins in the next turn,
  - A can create a forced sequence of winning outcomes.

C++:
- Same conceptual tactical engine, but in a compressed C++ style.
- Uses typed sets and explicit swap logic instead of Python tuple assignments.
- It preserves most of the strategic logic but the formatting is much more compact and less explanatory.

Key differences:
- Python is much more readable and comment-heavy.
- C++ keeps the same high-level logic but is more imperative and less self-documenting.
- The C++ version attempts to replicate the same tactical patterns, but it is not an exact one-to-one rewrite of the Python tuple-heavy structure.

## Overall summary

The Python version is a more complete, metadata-rich minimax engine with:
- time limits,
- search accounting,
- Zobrist hashing,
- iterative deepening,
- tuple-heavy structured threat states,
- richer debug output,
- tuple return values.

The C++ version is a streamlined port that keeps the core game logic and tactical evaluation ideas but collapses several Python conveniences into a leaner C++ interface. The most significant differences are in:
- `choose_move`,
- `minimax`,
- `search_moves`,
- `eval_move`,
- `static_eval`.

The lower-level threat-manipulation functions remain much closer in concept, though their data structures and explicit formatting differ materially.

## Final status after the latest parity pass

The most recent C++ edits tightened the move ordering and branch firing around the search layer, especially in `sort_remaining_moves` and `search_moves`, so the C++ port now lines up more closely with the Python branch structure in the immediate tactical cases.

The remaining nontrivial functional mismatches are still concentrated in the six search/evaluation functions below:
- `choose_move`
- `minimax`
- `search_moves`
- `eval_move`
- `static_eval`
- `calculate_value`

These are not merely formatting differences:
- the Python versions return richer tuple payloads and metadata;
- the C++ versions collapse that state into scalar values / references;
- the Python move search is time-aware and iterative-deepening based;
- the C++ port is a simplified pruning engine with more compact state.

So, as of the final pass, the C++ port is behaviorally close and passes the current unit/parity checks, but it still is not a literal branch-by-branch translation of the Python source in the six search/evaluation functions above.

The tactical threat-building functions (`update_wins_and_threats`, `process_wins_and_threats`, `detect_threats`, `update_threats`, `block_threats`, `get_squares_from_wn3_threat`, and `get_squares_from_wn4_threat`) are closer to parity, but their structure and representation still differ materially from Python because of the C++ type system and mutable-reference style.
