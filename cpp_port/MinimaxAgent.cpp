#include "MinimaxAgent.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kWinScore = 1'000'000.0;

    bool in_bounds(int row, int col, int rows, int cols) {
        return row >= 0 && row < rows && col >= 0 && col < cols;
    }

}  // namespace

namespace {
    using Coord = std::pair<int, int>;

    template <typename T>
    struct is_std_vector : std::false_type {};

    template <typename T, typename Alloc>
    struct is_std_vector<std::vector<T, Alloc>> : std::true_type {};

    template <typename T>
    struct is_std_set : std::false_type {};

    template <typename T, typename Compare, typename Alloc>
    struct is_std_set<std::set<T, Compare, Alloc>> : std::true_type {};

    template <typename T>
    constexpr bool is_std_vector_v = is_std_vector<std::decay_t<T>>::value;

    template <typename T>
    constexpr bool is_std_set_v = is_std_set<std::decay_t<T>>::value;

    template <typename T, typename = void>
    struct is_tuple_like : std::false_type {};

    template <typename T>
    struct is_tuple_like<T, std::void_t<decltype(std::tuple_size<T>::value)>> : std::true_type {};

    // flatten_threat_squares --
    template <typename T>
    void flatten_threat_squares(const T& value, std::set<Coord>& out) {
        using D = std::decay_t<T>;
        if constexpr (std::is_same_v<D, Coord>) {
            out.insert(value);
            return;
        }

        if constexpr (is_std_vector_v<D> || is_std_set_v<D>) {
            for (const auto& item : value) {
                flatten_threat_squares(item, out);
            }
            return;
        }

        if constexpr (std::is_same_v<D, std::pair<int, int>>) {
            out.insert(value);
            return;
        }

        if constexpr (is_tuple_like<D>::value) {
            std::apply([&](const auto&... elems) {
                (flatten_threat_squares(elems, out), ...);
            }, value);
            return;
        }

        (void)value;
    }
}  // namespace

using Coord = std::pair<int, int>;
using CoordList = std::vector<Coord>;
using CoordSet = std::set<Coord>;

inline CoordSet coord_intersection(
    const CoordSet& a, const CoordSet& b)
{
    CoordSet result;
    std::set_intersection(
        a.begin(), a.end(), b.begin(), b.end(),
        std::inserter(result, result.end()));
    return result;
}

inline CoordSet coord_difference(
    const CoordSet& a, const CoordSet& b)
{
    CoordSet result;
    std::set_difference(
        a.begin(), a.end(), b.begin(), b.end(),
        std::inserter(result, result.end()));
    return result;
}

inline CoordSet coord_symmetric_difference(
    const CoordSet& a, const CoordSet& b)
{
    CoordSet result;
    std::set_symmetric_difference(
        a.begin(), a.end(), b.begin(), b.end(),
        std::inserter(result, result.end()));
    return result;
}

inline CoordList to_coord_list(const CoordSet& squares)
{
    return CoordList(squares.begin(), squares.end());
}

inline bool adjacent(const Coord& a, const Coord& b)
{
    return std::abs(a.first - b.first) <= 1
        && std::abs(a.second - b.second) <= 1;
}

void add_children_to_set(
    const MinimaxAgent::WinNextNextChildren& children,
    std::set<Coord>& moves
) {
    std::visit([&moves](const auto& value) {
        using T = std::decay_t<decltype(value)>;

        if constexpr (std::is_same_v<T, CoordList>) {
            moves.insert(value.begin(), value.end());
        } else {
            moves.insert(value.first.begin(), value.first.end());
            moves.insert(value.second.begin(), value.second.end());
        }
    }, children);
}

namespace {

    // Map a source coordinate to its destination under a symmetry.
    //
    // 0: identity
    // 1: left-right reflection
    // 2: top-bottom reflection
    // 3: 180-degree rotation
    // 4: 90-degree clockwise rotation (square boards only)
    // 5: 270-degree rotation (square boards only)
    // 6: main-diagonal reflection (square boards only)
    // 7: anti-diagonal reflection (square boards only)
    GameState::Move transform_coord(
        int r, int c, int rows, int cols, int symmetry
    ) {
        switch (symmetry) {
            case 0: return {r, c};
            case 1: return {r, cols - 1 - c};
            case 2: return {rows - 1 - r, c};
            case 3: return {rows - 1 - r, cols - 1 - c};

            case 4: return {c, rows - 1 - r};
            case 5: return {cols - 1 - c, r};
            case 6: return {c, r};
            case 7: return {rows - 1 - c, cols - 1 - r};

            default: return {r, c};
        }
    }

    int inverse_symmetry(int symmetry) {
        switch (symmetry) {
            case 4: return 5;
            case 5: return 4;
            default: return symmetry;
        }
    }

} // namespace

void MinimaxAgent::store_zobrist_entry(
    const GameState& state,
    const ZobristTable& z_table,
    ZobristMemory& z_memory,
    const GameState::Move& move,
    double value,
    int deadline,
    int result,
    int fff
) const {
    const CanonicalPosition canonical =
        canonical_zobrist_key(state, z_table);

    GameState::Move canonical_move{-1, -1};

    if (move.first >= 0 && move.second >= 0) {
        canonical_move = transform_coord(
            move.first, move.second,
            state.rows(), state.cols(),
            canonical.symmetry
        );
    }

    z_memory[canonical.key] = std::make_tuple(
        canonical_move,
        value,
        deadline,
        result,
        fff
    );
}

bool MinimaxAgent::lookup_zobrist_entry(
    const GameState& state,
    const ZobristTable& z_table,
    const ZobristMemory& z_memory,
    SearchResult& out
) const {
    const CanonicalPosition canonical =
        canonical_zobrist_key(state, z_table);

    const auto it = z_memory.find(canonical.key);

    if (it == z_memory.end()) {
        return false;
    }

    const auto& entry = it->second;

    GameState::Move move = std::get<0>(entry);

    if (move.first >= 0 && move.second >= 0) {
        move = transform_coord(
            move.first, move.second,
            state.rows(), state.cols(),
            inverse_symmetry(canonical.symmetry)
        );
    }

    // Reject a cached move that isn't legal in this orientation.
    if (move.first >= 0 && !state.is_valid_move(move)) {
        return false;
    }

    out = SearchResult{
        move,
        std::get<1>(entry),
        std::get<2>(entry),
        std::get<3>(entry),
        std::get<4>(entry)
    };

    return true;
}

MinimaxAgent::CanonicalPosition MinimaxAgent::canonical_zobrist_key(
    const GameState& state,
    const ZobristTable& z_table
) {
    const int rows = state.rows();
    const int cols = state.cols();

    // Ensure the Zobrist table corresponds to this board.
    if (z_table.size() !=
        static_cast<std::size_t>(rows) *
        static_cast<std::size_t>(cols)) {
        throw std::invalid_argument(
            "Zobrist table dimensions do not match GameState"
        );
    }

    const int symmetry_count = (rows == cols) ? 8 : 4;
    const auto& board = state.board();

    CanonicalPosition canonical{
        std::numeric_limits<std::uint64_t>::max(),
        0
    };

    for (int symmetry = 0;
         symmetry < symmetry_count;
         ++symmetry) {

        std::uint64_t key = 0;

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const char piece = board[r][c];

                if (piece == GameState::EMPTY_PIECE) {
                    continue;
                }

                int piece_index;

                if (piece == GameState::X_PIECE) {
                    piece_index = 0;
                } else if (piece == GameState::O_PIECE) {
                    piece_index = 1;
                } else {
                    throw std::invalid_argument(
                        "Unexpected piece in GameState board"
                    );
                }

                const auto [tr, tc] = transform_coord(
                    r, c, rows, cols, symmetry
                );

                const std::size_t offset =
                    static_cast<std::size_t>(tr) *
                    static_cast<std::size_t>(cols) +
                    static_cast<std::size_t>(tc);

                key ^= static_cast<std::uint64_t>(
                    z_table[offset][piece_index]
                );
            }
        }

        if (key < canonical.key) {
            canonical.key = key;
            canonical.symmetry = symmetry;
        }
    }

    return canonical;
}

// make_zobrist_hashing --
MinimaxAgent::ZobristHashing MinimaxAgent::make_zobrist_hashing(int rows, int cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::uint64_t> dist(0u, std::numeric_limits<std::uint64_t>::max());

    ZobristTable z_table(static_cast<std::size_t>(rows * cols), std::vector<std::uint64_t>(2, 0u));
    for (auto& entry : z_table) {
        entry[0] = dist(rng);
        entry[1] = dist(rng);
    }
    return std::make_tuple(z_table, ZobristMemory{}, std::uint64_t{0});
}

// step_zobrist_key --
std::uint64_t MinimaxAgent::step_zobrist_key(
    std::uint64_t current_key, const ZobristTable& z_table,
    int row, int col, int z_index, int board_cols
) {
    if (row < 0 || col < 0 || board_cols <= 0 ||
        col >= board_cols || z_index < 0 || z_index >= 2) {
        std::cerr << "Error: invalid step_zobrist_key arguments\n";
        std::exit(EXIT_FAILURE);
    }

    const std::size_t offset =
        static_cast<std::size_t>(row) * board_cols + col;

    if (offset >= z_table.size() || z_table[offset].size() < 2) {
        std::cerr << "Error: step_zobrist_key out of bounds\n";
        std::exit(EXIT_FAILURE);
    }

    return current_key ^ static_cast<std::uint64_t>(z_table[offset][z_index]);
}



// legal_moves --
std::vector<GameState::Move> MinimaxAgent::legal_moves(const GameState& state) const {
    std::vector<GameState::Move> moves;
    for (int r = 0; r < state.rows(); ++r) {
        for (int c = 0; c < state.cols(); ++c) {
            if (state.is_valid_move(r, c)) {
                moves.emplace_back(r, c);
            }
        }
    }
    return moves;
}

// opponent --
char MinimaxAgent::opponent(char piece) {
    return (piece == GameState::X_PIECE) ? GameState::O_PIECE : GameState::X_PIECE;
}

// is_winning_move --
bool MinimaxAgent::is_winning_move(const GameState& state, const GameState::Move& move, char player) {
    auto next_board = state.board();
    next_board[move.first][move.second] = player;

    GameState next(state.rows(), state.cols(), state.k_in_row(), opponent(player), next_board);
    for (int r = 0; r < next.rows(); ++r) {
        for (int c = 0; c < next.cols(); ++c) {
            if (next.board()[r][c] != player) {
                continue;
            }
            for (const auto& [dr, dc] : std::vector<std::pair<int, int>>{{0, 1}, {1, 0}, {1, 1}, {1, -1}}) {
                int count = 1;
                for (int step = 1; step < next.k_in_row(); ++step) {
                    const int rr = r + step * dr;
                    const int cc = c + step * dc;
                    if (!in_bounds(rr, cc, next.rows(), next.cols()) || next.board()[rr][cc] != player) {
                        break;
                    }
                    ++count;
                }
                if (count >= next.k_in_row()) {
                    return true;
                }
            }
        }
    }
    return false;
}

// score_window --
int MinimaxAgent::score_window(const std::vector<char>& window, int k, char player) {
    const char other = opponent(player);
    int player_count = 0;
    int other_count = 0;

    for (char ch : window) {
        if (ch == player) {
            ++player_count;
        } else if (ch == other) {
            ++other_count;
        }
    }

    if (player_count > 0 && other_count > 0) {
        return 0;
    }
    if (player_count == 0 && other_count == 0) {
        return 0;
    }
    if (player_count > 0) {
        if (player_count == k) {
            return 100000;
        }
        return static_cast<int>(std::pow(10, player_count - 1)) * (k - player_count + 1);
    }
    if (other_count == k) {
        return -100000;
    }
    return -static_cast<int>(std::pow(10, other_count - 1)) * (k - other_count + 1);
}




// update_wins_and_threats --
void MinimaxAgent::update_wins_and_threats(int k, const std::vector<Coord>& last, int count, double& value, int& wins,
    std::set<Coord>& win_next_turn_threats, std::set<Threat1>& k_1_threats,
    std::set<Threat2>& k_2_threats,
    std::set<Threat3>& k_3_threats) const {
    // Update the total value and record any k-rows close to being filled

    // An unblocked k-row filled with x squares counts for 10^(x-1) value
    value += std::pow(10.0, count - 1);
    if (count == k) {
        // k out of k squares are filled, game over
        ++wins;
        return;
    }
    if (count == k - 1) {
        // k-1 out of k squares are filled, player can win next turn
        win_next_turn_threats.insert(last[0]);
        return;
    }
    if (count == k - 2) {
        // k-2 out of k squares are filled, player can fill to k-1 next turn
        k_1_threats.insert(std::make_tuple(last[0], last[1]));
        return;
    }
    if (count == k - 3) {
        // k-3 out of k squares are filled, player can fill to k-2 next turn
        k_2_threats.insert(std::make_tuple(last[0], last[1], last[2]));
        return;
    }
    if (count == k - 4) {
        // k-4 out of k squares are filled, player can fill to k-3 next turn
        k_3_threats.insert(std::make_tuple(last[0], last[1], last[2], last[3]));
    }
}

// process_wins_and_threats --
void MinimaxAgent::process_wins_and_threats(const GameState& state, int h, int w, int k,
    const std::vector<std::pair<int, int>>& directions, char a_piece, char b_piece,
    std::set<Coord>& a_win_next_turn_threats,
    std::set<Coord>& b_win_next_turn_threats,
    std::set<Threat1>& a_k_1_threats,
    std::set<Threat1>& b_k_1_threats,
    std::set<Threat2>& a_k_2_threats,
    std::set<Threat2>& b_k_2_threats,
    std::set<Threat3>& a_k_3_threats,
    std::set<Threat3>& b_k_3_threats,
    double& a_value, double& b_value, int& a_wins, int& b_wins) const {

    const auto& board = state.board();
    const char block_value = '-';

    for (const auto& [dx, dy] : directions) {
        const int min_row = 0;
        const int min_col = (dx == -1) ? k - 1 : 0;
        const int max_row = (dy == 1) ? h - k : h - 1;
        const int max_col = (dx == 1) ? w - k : w - 1;

        for (int i = min_row; i <= max_row; ++i) {
            for (int j = min_col; j <= max_col; ++j) {

                int row = i;
                int col = j;
                int a_count = 0;
                int b_count = 0;
                bool a_blocked = false;
                bool b_blocked = false;
                std::vector<Coord> last(4, {-11, -11});

                for (int step = 0; step < k; ++step) {
                    const char piece = board[row][col];

                    if (piece == block_value) {
                        a_blocked = true;
                        b_blocked = true;
                        break;
                    }

                    if (piece == a_piece) {
                        ++a_count;
                        b_blocked = true;
                    } else if (piece == b_piece) {
                        ++b_count;
                        a_blocked = true;
                    } else {
                        last = {Coord(row, col), last[0], last[1], last[2]};
                    }

                    if (a_blocked && b_blocked) {
                        break;
                    }

                    row += dy;
                    col += dx;
                }

                if (a_count > 0 && !a_blocked) {
                    update_wins_and_threats(
                        k, last, a_count,
                        a_value, a_wins,
                        a_win_next_turn_threats,
                        a_k_1_threats,
                        a_k_2_threats,
                        a_k_3_threats
                    );
                } else if (b_count > 0 && !b_blocked) {
                    update_wins_and_threats(
                        k, last, b_count,
                        b_value, b_wins,
                        b_win_next_turn_threats,
                        b_k_1_threats,
                        b_k_2_threats,
                        b_k_3_threats
                    );
                }
            }
        }
    }
}

// detect_threats --
void MinimaxAgent::detect_threats(
    const std::set<Threat1>& k_1_threats,
    const std::set<Threat2>& k_2_threats,
    const std::set<Threat3>& k_3_threats,
    std::set<WinNextNextTurnThreat>& win_next_next_turn_threats,
    std::set<WinNextNextNextTurnThreat>& win_next_next_next_turn_threats,
    std::set<WinNextNextNextNextTurnThreat>& win_next_next_next_next_turn_threats
) const
{
    struct OverlappingK2 {
        CoordSet squares;
        CoordSet edges;
    };

    struct OverlappingK3 {
        CoordSet squares;
        CoordSet inner;
        CoordSet outer;
    };

    struct K1K2Threat {
        Coord square;
        Coord one;
        CoordSet two;
    };

    std::vector<OverlappingK2> overlapping_k_2_threats;
    std::vector<OverlappingK3> overlapping_k_3_threats;
    std::vector<K1K2Threat> k_1_k_2_threats;

    std::vector<CoordSet> k_1_sets;
    std::vector<CoordSet> k_2_sets;
    std::vector<CoordSet> k_3_sets;

    for (const auto& t : k_1_threats) {
        k_1_sets.emplace_back(
            CoordSet{std::get<0>(t), std::get<1>(t)});
    }

    for (const auto& t : k_2_threats) {
        k_2_sets.emplace_back(
            CoordSet{
                std::get<0>(t),
                std::get<1>(t),
                std::get<2>(t)
            });
    }

    for (const auto& t : k_3_threats) {
        k_3_sets.emplace_back(
            CoordSet{
                std::get<0>(t),
                std::get<1>(t),
                std::get<2>(t),
                std::get<3>(t)
            });
    }

    // Record places where two win threats can be created
    // on the same turn.
    for (const auto& s1 : k_1_sets) {
        for (const auto& s2 : k_1_sets) {
            CoordSet inter = coord_intersection(s1, s2);

            if (inter.size() == 1) {
                Coord overlap = *inter.begin();
                CoordSet remaining =
                    coord_symmetric_difference(s1, s2);

                win_next_next_turn_threats.insert({
                    overlap,
                    to_coord_list(remaining)
                });
            }
        }
    }

    // Record places where two parallel k-1 threats
    // can be created on the same turn.
    for (const auto& s1 : k_2_sets) {
        for (const auto& s2 : k_2_sets) {
            CoordSet inter = coord_intersection(s1, s2);

            if (inter.size() == 2) {
                CoordSet edge_squares =
                    coord_symmetric_difference(s1, s2);

                overlapping_k_2_threats.push_back({
                    inter,
                    edge_squares
                });
            }
        }
    }

    // Record places where two parallel k-2 threats
    // can be created on the same turn.
    for (const auto& s1 : k_3_sets) {
        for (const auto& s2 : k_3_sets) {
            CoordSet inter = coord_intersection(s1, s2);

            if (inter.size() == 2 && s1.size() != 2) {
                CoordList overlapping_squares =
                    to_coord_list(inter);

                const Coord& os1 = overlapping_squares[0];
                const Coord& os2 = overlapping_squares[1];

                if (adjacent(os1, os2)) {
                    continue;
                }

                CoordSet other_squares =
                    coord_symmetric_difference(s1, s2);

                CoordSet inner_squares;
                CoordSet outer_squares;

                for (const auto& square : other_squares) {
                    for (const auto& overlap : overlapping_squares) {
                        if (adjacent(square, overlap)) {
                            inner_squares.insert(square);
                        }
                    }
                }

                outer_squares =
                    coord_difference(other_squares, inner_squares);

                overlapping_k_3_threats.push_back({
                    inter,
                    inner_squares,
                    outer_squares
                });
            }
        }
    }

    // Record places where a win threat and a k-1 threat
    // can be created on the same turn.
    for (const auto& s1 : k_1_sets) {
        for (const auto& s2 : k_2_sets) {
            CoordSet inter = coord_intersection(s1, s2);

            if (inter.size() == 1) {
                Coord overlap = *inter.begin();

                CoordSet one_remaining =
                    coord_difference(s1, inter);

                CoordSet two_remaining =
                    coord_difference(s2, inter);

                k_1_k_2_threats.push_back({
                    overlap,
                    *one_remaining.begin(),
                    two_remaining
                });
            }
        }
    }

    // Higher-order overlapping logic.
    const auto ok2 = overlapping_k_2_threats;
    const auto ok3 = overlapping_k_3_threats;

    const std::size_t m1 = ok2.size();
    const std::size_t m2 = ok3.size();

    for (std::size_t i = 0; i < m1; ++i) {
        const CoordSet& s1 = ok2[i].squares;
        const CoordSet& edges1 = ok2[i].edges;

        // Record places where two sets of two parallel
        // k-1 threats can be created on the same turn.
        for (std::size_t j = i + 1; j < m1; ++j) {
            const CoordSet& s2 = ok2[j].squares;
            const CoordSet& edges2 = ok2[j].edges;

            CoordSet inter = coord_intersection(s1, s2);
            CoordSet blocked = coord_intersection(s1, edges2);

            if (inter.size() == 1 && blocked.empty()) {
                Coord overlap = *inter.begin();

                Coord a = *coord_difference(
                    s1, CoordSet{overlap}).begin();

                Coord b = *coord_difference(
                    s2, CoordSet{overlap}).begin();

                WinNextNextTurnThreat ref_a{
                    a, to_coord_list(edges1)
                };

                WinNextNextTurnThreat ref_b{
                    b, to_coord_list(edges2)
                };

                win_next_next_next_turn_threats.insert({
                    overlap,
                    {ref_a, ref_b}
                });
            }
        }

        // Record places where two parallel k-1 threats
        // and a win threat can be created on the same turn.
        for (const auto& s2 : k_1_sets) {
            CoordSet inter = coord_intersection(s1, s2);

            if (inter.size() == 1) {
                Coord overlap = *inter.begin();

                Coord a = *coord_difference(
                    s1, CoordSet{overlap}).begin();

                Coord forced = *coord_difference(
                    s2, CoordSet{overlap}).begin();

                WinNextNextTurnThreat ref{
                    a,
                    std::pair{
                        to_coord_list(edges1),
                        to_coord_list(s2)
                    }
                };

                if (forced != a && !edges1.contains(forced)) {
                    win_next_next_next_turn_threats.insert({
                        overlap,
                        {ref}
                    });
                }
            }
        }

        // Higher-order interaction between overlapping k-2
        // and k-3 threats.
        for (std::size_t j = 0; j < m2; ++j) {
            const CoordSet& s2 = ok3[j].squares;
            const CoordSet& inner2 = ok3[j].inner;
            const CoordSet& outer2 = ok3[j].outer;

            CoordSet inter1 = coord_intersection(s1, s2);

            if (inter1.size() != 1
                || !coord_intersection(s1, inner2).empty()
                || !coord_intersection(s1, outer2).empty()) {
                continue;
            }

            for (std::size_t k = 0; k < m1; ++k) {
                if (k == i) {
                    continue;
                }

                const CoordSet& s3 = ok2[k].squares;
                const CoordSet& edges3 = ok2[k].edges;

                CoordSet inter2 = coord_intersection(s3, inner2);

                if (inter2.size() != 1
                    || !coord_intersection(s3, s2).empty()
                    || !coord_intersection(s3, outer2).empty()) {
                    continue;
                }

                Coord overlap_a = *inter1.begin();
                Coord overlap_b = *inter2.begin();

                Coord other_square2 = *coord_difference(
                    s2, CoordSet{overlap_a}).begin();

                Coord other_square3 = *coord_difference(
                    s3, CoordSet{overlap_b}).begin();

                std::vector<WinNextNextTurnThreat> refs_a{
                    WinNextNextTurnThreat{
                        other_square2,
                        to_coord_list(outer2)
                    },
                    WinNextNextTurnThreat{
                        other_square3,
                        to_coord_list(edges3)
                    }
                };

                std::vector<WinNextNextTurnThreat> refs_b{
                    WinNextNextTurnThreat{
                        other_square2,
                        to_coord_list(outer2)
                    },
                    WinNextNextTurnThreat{
                        other_square3,
                        to_coord_list(edges3)
                    }
                };

                WinNextNextNextNextTurnThreat threat_a;
                threat_a.root = overlap_a;
                threat_a.first_ref = overlap_b;
                threat_a.refs = refs_a;
                win_next_next_next_next_turn_threats.insert(threat_a);

                WinNextNextNextNextTurnThreat threat_b;
                threat_b.root = overlap_b;
                threat_b.first_ref = overlap_a;
                threat_b.refs = refs_b;
                win_next_next_next_next_turn_threats.insert(threat_b);
            }
        }
    }

    const std::size_t m = k_1_k_2_threats.size();

    for (std::size_t i = 0; i < m; ++i) {
        const auto& t1 = k_1_k_2_threats[i];

        const Coord& square1 = t1.square;
        const Coord& one1 = t1.one;
        const CoordSet& two1 = t1.two;

        for (std::size_t j = i + 1; j < m; ++j) {
            const auto& t2 = k_1_k_2_threats[j];

            const Coord& square2 = t2.square;
            const Coord& one2 = t2.one;
            const CoordSet& two2 = t2.two;

            CoordSet inter = coord_intersection(two1, two2);

            if (inter.size() == 1) {
                Coord overlap = *inter.begin();

                Coord remaining1 = *coord_difference(
                    two1, CoordSet{overlap}).begin();

                Coord remaining2 = *coord_difference(
                    two2, CoordSet{overlap}).begin();

                win_next_next_next_turn_threats.insert({
                    overlap,
                    {
                        WinNextNextTurnThreat{
                            square1,
                            CoordList{one1, remaining1}
                        },
                        WinNextNextTurnThreat{
                            square2,
                            CoordList{one2, remaining2}
                        }
                    }
                });
            }
        }
    }
}

// update_threats --
void MinimaxAgent::update_threats(
    const Coord& new_square,
    std::set<Threat1>& k_1_threats,
    std::set<Threat2>& k_2_threats,
    std::set<Coord>& win_next_turn_threats,
    std::set<WinNextNextTurnThreat>& win_next_next_turn_threats,
    std::set<WinNextNextNextTurnThreat>& win_next_next_next_turn_threats,
    std::set<WinNextNextNextNextTurnThreat>& win_next_next_next_next_turn_threats
) const
{
    // Record any new threats resulting from playing a new move.

    std::vector<Threat1> to_remove;

    for (const auto& t : k_1_threats) {
        const Coord& a = std::get<0>(t);
        const Coord& b = std::get<1>(t);

        if (new_square == a || new_square == b) {
            if (a != new_square) {
                win_next_turn_threats.insert(a);
            }

            if (b != new_square) {
                win_next_turn_threats.insert(b);
            }

            to_remove.push_back(t);
        }
    }

    for (const auto& t : to_remove) {
        k_1_threats.erase(t);
    }

    // Update next-next-turn threats.
    std::vector<WinNextNextTurnThreat> to_remove_wnn;

    for (const auto& w : win_next_next_turn_threats) {
        const Coord& root = w.root;

        CoordList children;

        if (std::holds_alternative<CoordList>(w.children)) {
            children = std::get<CoordList>(w.children);
        } else {
            const auto& child_sets =
                std::get<std::pair<CoordList, CoordList>>(w.children);

            children.insert(
                children.end(),
                child_sets.first.begin(),
                child_sets.first.end()
            );

            children.insert(
                children.end(),
                child_sets.second.begin(),
                child_sets.second.end()
            );
        }

        if (new_square == root) {
            win_next_turn_threats.insert(
                children.begin(), children.end()
            );

            to_remove_wnn.push_back(w);
            continue;
        }

        for (const auto& sq : children) {
            if (new_square == sq) {
                win_next_turn_threats.insert(root);
                to_remove_wnn.push_back(w);
                break;
            }
        }
    }

    for (const auto& w : to_remove_wnn) {
        win_next_next_turn_threats.erase(w);
    }

    // Update k-2 threats.
    std::vector<Threat2> to_remove_k2;

    for (const auto& t : k_2_threats) {
        const Coord& a = std::get<0>(t);
        const Coord& b = std::get<1>(t);
        const Coord& c = std::get<2>(t);

        if (new_square == a
            || new_square == b
            || new_square == c) {

            if (a == new_square) {
                k_1_threats.insert(Threat1{b, c});
            } else if (b == new_square) {
                k_1_threats.insert(Threat1{a, c});
            } else {
                k_1_threats.insert(Threat1{a, b});
            }

            to_remove_k2.push_back(t);
        }
    }

    for (const auto& t : to_remove_k2) {
        k_2_threats.erase(t);
    }

    // Update next-next-next-turn threats.
    std::vector<WinNextNextNextTurnThreat> to_remove_wnnn;

    for (const auto& w : win_next_next_next_turn_threats) {
        const Coord& root = w.root;
        const auto& refs = w.refs;

        if (new_square == root) {
            win_next_next_turn_threats.insert(
                refs.begin(), refs.end()
            );

            to_remove_wnnn.push_back(w);
            continue;
        }

        for (const auto& ref : refs) {
            if (new_square == ref.root) {
                win_next_next_turn_threats.insert(ref);
                to_remove_wnnn.push_back(w);
                break;
            }
        }
    }

    for (const auto& w : to_remove_wnnn) {
        win_next_next_next_turn_threats.erase(w);
    }

    // Update next-next-next-next-turn threats.
    std::vector<WinNextNextNextNextTurnThreat> to_remove_wnnnn;

    for (const auto& w : win_next_next_next_next_turn_threats) {
        if (new_square == w.root) {
            // Reconstruct the nested third-order threat:
            // (first_ref, refs).
            WinNextNextNextTurnThreat activated{
                w.first_ref,
                w.refs
            };

            win_next_next_next_turn_threats.insert(activated);
            to_remove_wnnnn.push_back(w);
        }
    }

    for (const auto& w : to_remove_wnnnn) {
        win_next_next_next_next_turn_threats.erase(w);
    }
}


// block_threats --
void MinimaxAgent::block_threats(
    const Coord& blocking_square,
    std::set<Threat1>& k_1_threats,
    std::set<Threat2>& k_2_threats,
    std::set<Threat3>& k_3_threats,
    std::set<WinNextNextTurnThreat>& win_next_next_turn_threats,
    std::set<WinNextNextNextTurnThreat>& win_next_next_next_turn_threats,
    std::set<WinNextNextNextNextTurnThreat>& win_next_next_next_next_turn_threats
) const
{
    // Extract the children represented by either variant alternative.
    auto get_children = [](
        const WinNextNextTurnThreat& w
    ) -> std::vector<Coord> {
        if (std::holds_alternative<CoordList>(w.children)) {
            return std::get<CoordList>(w.children);
        }

        const auto& child_sets =
            std::get<std::pair<CoordList, CoordList>>(w.children);

        std::vector<Coord> children = child_sets.first;
        children.insert(
            children.end(),
            child_sets.second.begin(),
            child_sets.second.end()
        );

        return children;
    };

    /*
     * If blocking a win also blocks any other threats,
     * remove them from storage.
     */

    // Remove k-1 threats containing the blocking square.
    {
        std::vector<Threat1> to_remove;

        for (const auto& t : k_1_threats) {
            if (blocking_square == std::get<0>(t)
                || blocking_square == std::get<1>(t)) {
                to_remove.push_back(t);
            }
        }

        for (const auto& t : to_remove) {
            k_1_threats.erase(t);
        }
    }

    // Remove next-next-turn threats blocked by this square.
    {
        std::vector<WinNextNextTurnThreat> to_remove;

        for (const auto& w : win_next_next_turn_threats) {
            const Coord& root = w.root;
            const std::vector<Coord> children = get_children(w);

            if (blocking_square == root
                || std::find(
                    children.begin(), children.end(), blocking_square
                ) != children.end()) {
                to_remove.push_back(w);
            }
        }

        for (const auto& w : to_remove) {
            win_next_next_turn_threats.erase(w);
        }
    }

    // Remove k-2 threats containing the blocking square.
    {
        std::vector<Threat2> to_remove;

        for (const auto& t : k_2_threats) {
            if (blocking_square == std::get<0>(t)
                || blocking_square == std::get<1>(t)
                || blocking_square == std::get<2>(t)) {
                to_remove.push_back(t);
                continue;
            }
        }

        for (const auto& t : to_remove) {
            k_2_threats.erase(t);
        }
    }

    // Remove next-next-next-turn threats that are blocked.
    {
        std::vector<WinNextNextNextTurnThreat> to_remove;

        for (const auto& w : win_next_next_next_turn_threats) {
            const Coord& root = w.root;

            if (blocking_square == root) {
                to_remove.push_back(w);
                continue;
            }

            for (const auto& ref : w.refs) {
                const Coord& ref_root = ref.root;
                const std::vector<Coord> children = get_children(ref);

                if (blocking_square == ref_root
                    || std::find(
                        children.begin(), children.end(), blocking_square
                    ) != children.end()) {
                    to_remove.push_back(w);
                    break;
                }
            }
        }

        for (const auto& w : to_remove) {
            win_next_next_next_turn_threats.erase(w);
        }
    }

    // Remove k-3 threats containing the blocking square.
    {
        std::vector<Threat3> to_remove;

        for (const auto& t : k_3_threats) {
            if (blocking_square == std::get<0>(t)
                || blocking_square == std::get<1>(t)
                || blocking_square == std::get<2>(t)
                || blocking_square == std::get<3>(t)) {
                to_remove.push_back(t);
                continue;
            }
        }

        for (const auto& t : to_remove) {
            k_3_threats.erase(t);
        }
    }

    // Remove next-next-next-next-turn threats that are blocked.
    {
        std::vector<WinNextNextNextNextTurnThreat> to_remove;

        for (const auto& w : win_next_next_next_next_turn_threats) {
            const Coord& root = w.root;

            if (blocking_square == root
                || blocking_square == w.first_ref) {
                to_remove.push_back(w);
                continue;
            }

            for (const auto& ref : w.refs) {
                const Coord& ref_root = ref.root;
                const std::vector<Coord> children = get_children(ref);

                if (blocking_square == ref_root
                    || std::find(
                        children.begin(), children.end(), blocking_square
                    ) != children.end()) {
                    to_remove.push_back(w);
                    break;
                }
            }
        }

        for (const auto& w : to_remove) {
            win_next_next_next_next_turn_threats.erase(w);
        }
    }
}

// get_squares_from_wn3_threat --
CoordSet MinimaxAgent::get_squares_from_wn3_threat(
    const WinNextNextNextTurnThreat& threat
) const {
    const auto& refs = threat.refs;
    const auto& first = refs[0];

    CoordSet squares;
    squares.insert(first.root);

    auto add_children = [&squares](
        const WinNextNextTurnThreat& ref
    ) {
        if (std::holds_alternative<CoordList>(ref.children)) {
            const auto& children = std::get<CoordList>(ref.children);
            squares.insert(children.begin(), children.end());
        } else {
            const auto& child_sets =
                std::get<std::pair<CoordList, CoordList>>(ref.children);

            squares.insert(
                child_sets.first.begin(), child_sets.first.end()
            );
            squares.insert(
                child_sets.second.begin(), child_sets.second.end()
            );
        }
    };

    add_children(first);

    if (refs.size() == 2) {
        const auto& second = refs[1];
        squares.insert(second.root);
        add_children(second);
    }

    return squares;
}

// get_squares_from_wn4_threat --
CoordSet MinimaxAgent::get_squares_from_wn4_threat(
    const WinNextNextNextNextTurnThreat& threat
) const {
    CoordSet squares;
    squares.insert(threat.root);
    squares.insert(threat.first_ref);

    // The Python helper passes refs directly to the wn3 helper.
    // Adapt the stored wn4 references to the wn3 structure.
    WinNextNextNextTurnThreat nested{
        threat.first_ref,
        threat.refs
    };

    CoordSet nested_squares = get_squares_from_wn3_threat(nested);
    squares.insert(nested_squares.begin(), nested_squares.end());

    return squares;
}












// MinimaxAgent --
MinimaxAgent::MinimaxAgent(char piece, int max_depth, unsigned seed, int init_ff_branch_max)
    : piece_(piece), max_depth_(max_depth), seed_(seed), init_ff_branch_max_(std::max(0, init_ff_branch_max)) {
    if (piece_ != GameState::X_PIECE && piece_ != GameState::O_PIECE) {
        throw std::invalid_argument("piece must be X or O");
    }
    if (max_depth_ <= 0) {
        max_depth_ = 1;
    }
}

// name --
std::string MinimaxAgent::name() const {
    std::ostringstream oss;
    oss << "PythonParityMinimaxAgent " << piece_;
    return oss.str();
}


// choose_move --
GameState::Move MinimaxAgent::choose_move(const GameState& state, int depth) const {
    return choose_move_impl(state, depth, nullptr);
}

// choose_move --
GameState::Move MinimaxAgent::choose_move(const GameState& state, double time_limit_seconds) const {
    if (time_limit_seconds < 0.0) return choose_move_impl(state, -1, nullptr);
    const TimePoint timeout = Clock::now() + std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(time_limit_seconds));
    return choose_move_impl(state, -1, &timeout);
}


// choose_move_impl --
GameState::Move MinimaxAgent::choose_move_impl(
    const GameState& state,
    int depth,
    const TimePoint* timeout
) const {

    std::cout << "_____________________________________________\n\n";
    std::cout << state << "\n";
    std::cout << "               " << state.next_player() << " TO PLAY               \n\n";

    static_eval_calls = 0;

    // Default best move is first available empty space
    GameState::Move best_move{-1, -1};
    int max_depth = 0;
    for (int r = 0; r < state.rows(); ++r) {
        for (int c = 0; c < state.cols(); ++c) {
            if (state.board()[r][c] == GameState::EMPTY_PIECE) {
                ++max_depth;
                if (best_move.first < 0) {
                    best_move = {r, c};
                }
            }
        }
    }

    // Limit the maximum search depth
    max_depth = std::min(max_depth, depth);

    constexpr auto wrapup = std::chrono::milliseconds(100);
    int current_depth = 1;

    EvalResult e = static_eval_result(state);
    std::cout << "static_eval=(" << e.value << ", " << e.deadline << ", " << e.result << ")\n";

    while (current_depth <= max_depth) {

        bool timed_out = false;
        SearchResult result;
        depth_counter = {};
        depth_offset = current_depth;
        if (timeout && Clock::now() >= *timeout - wrapup) break;

        // Initialise Zobrist hash table
        auto z_hashing = make_zobrist_hashing(
            state.rows(),
            state.cols(),
            seed_ + static_cast<unsigned>(current_depth)
        );

        // Search for best value at current depth
        try {
            result = minimax(
                state,
                current_depth,
                -kInf,
                kInf,
                &z_hashing,
                0,
                init_ff_branch_max_,
                timeout,
                &timed_out
            );
        } catch (const std::exception& e) {
            std::cerr << "Error during minimax search: " << e.what() << std::endl;
            break;
        }

        if (!result.completed || timed_out ||
            (timeout && Clock::now() >= *timeout - wrapup)) {
            // Time limit reached, exit search
            break;
        }

        // Full search complete, update best_move
        best_move = result.move;
        double best_value = result.value;
        int best_deadline = result.deadline - 1;
        int best_fff = result.fff;
        std::string best_result = "Play";

        if ((state.next_player() == GameState::X_PIECE && result.result == 2) || (state.next_player() == GameState::O_PIECE && result.result == 3))
            best_result = "Win";
        else if ((state.next_player() == GameState::X_PIECE && result.result == 3) || (state.next_player() == GameState::O_PIECE && result.result == 2))
            best_result = "Loss";
        else if (result.result == 1)
            best_result = "Block";

        std::cout << "depth=" << current_depth
            << ", move=(" << best_move.first
            << ", " << best_move.second << ")"
            << ", value=" << best_value
            << ", deadline=" << best_deadline
            << ", result=" << best_result
            << ", fff=" << best_fff
            << '\n';

        // Endgame or forced block detected, stop search
        if (best_result != "Play") {
            std::cout << best_result << " found in " << best_deadline << " moves\n";
            break;
        }

        // Search again, one layer deeper
        current_depth += 1;
    }

    const bool best_move_valid = state.is_valid_move(best_move);
    if (!best_move_valid) {
        throw std::runtime_error(
            "no valid move found for the current board"
        );
    }

    std::cout << "\nCalled static_eval() "
        << static_eval_calls
        << " times\n";
    for (int i = 0; i <= 100; ++i) {
        if (depth_counter[i] == 0) break;
        std::cout << i << "\t| " << depth_counter[i] << "\n";
    }

    return best_move;
}


// minimax --
MinimaxAgent::SearchResult MinimaxAgent::minimax(const GameState& state, int depth_remaining, double alpha, double beta,
    ZobristHashing* z_hashing, int ff, int ff_branch_max, const TimePoint* timeout, bool* timed_out) const {

    depth_counter[depth_offset - depth_remaining + ff] += 1;

    // Exit early if reached time limit
    constexpr auto wrapup = std::chrono::milliseconds(100);
    if ((timed_out && *timed_out) || (timeout && Clock::now() >= *timeout - wrapup)) {
        if (timed_out) *timed_out = true;
        return SearchResult{GameState::Move{-1, -1}, 0.0, 0, 0, ff, false};
    }

    // Return static evaluation if reached depth limit or game over
    const auto outcome = state.winner();
    if (depth_remaining <= 0 || state.is_full() || outcome != GameState::Outcome::InProgress) {
        const EvalResult eval = static_eval_result(state);
        SearchResult result{GameState::Move{-1, -1}, eval.value, eval.deadline, eval.result, ff};
        if (z_hashing != nullptr) {
            auto& z_table = std::get<0>(*z_hashing);
            auto& z_memory = std::get<1>(*z_hashing);
            // auto& z_key = std::get<2>(*z_hashing);

            // z_memory[z_key] = std::make_tuple(result.move, result.value, result.deadline, result.result, result.fff);

            store_zobrist_entry(
                state, z_table, z_memory,
                result.move, result.value, result.deadline,
                result.result, result.fff
            );
        }
        return result;
    }

    // Otherwise do minimax
    const int h = state.rows();
    const int w = state.cols();
    const int k = state.k_in_row();
    const char a_piece = state.next_player();
    const char b_piece = (a_piece == GameState::X_PIECE) ? GameState::O_PIECE : GameState::X_PIECE;
    const double sign = (a_piece == GameState::X_PIECE) ? 1.0 : -1.0;

    // Initialise best move state
    GameState::Move best_move{-1, -1};
    double best_value = (a_piece == GameState::X_PIECE) ? -kInf : kInf;
    int best_deadline = 0;
    int best_result = (a_piece == GameState::X_PIECE) ? 3 : 2;
    int best_fff = ff;
    const int z_index = (a_piece == GameState::X_PIECE) ? 0 : 1;

    const std::vector<std::pair<int, int>> directions{{1, 0}, {0, 1}, {1, 1}, {-1, 1}};

    // Scan for wins and threats
    std::set<Coord> a_win_next_turn_threats;
    std::set<Coord> b_win_next_turn_threats;
    std::set<Threat1> a_k_1_threats;
    std::set<Threat1> b_k_1_threats;
    std::set<Threat2> a_k_2_threats;
    std::set<Threat2> b_k_2_threats;
    std::set<Threat3> a_k_3_threats;
    std::set<Threat3> b_k_3_threats;
    std::set<WinNextNextTurnThreat> a_win_next_next_turn_threats;
    std::set<WinNextNextTurnThreat> b_win_next_next_turn_threats;
    std::set<WinNextNextNextTurnThreat> a_win_next_next_next_turn_threats;
    std::set<WinNextNextNextTurnThreat> b_win_next_next_next_turn_threats;
    std::set<WinNextNextNextNextTurnThreat> a_win_next_next_next_next_turn_threats;
    std::set<WinNextNextNextNextTurnThreat> b_win_next_next_next_next_turn_threats;

    double a_value = 0.0;
    double b_value = 0.0;
    int a_wins = 0;
    int b_wins = 0;

    process_wins_and_threats(state, h, w, k, directions, a_piece, b_piece,
                            a_win_next_turn_threats, b_win_next_turn_threats,
                            a_k_1_threats, b_k_1_threats,
                            a_k_2_threats, b_k_2_threats,
                            a_k_3_threats, b_k_3_threats,
                            a_value, b_value, a_wins, b_wins);

    detect_threats(a_k_1_threats, a_k_2_threats, a_k_3_threats,
                a_win_next_next_turn_threats,
                a_win_next_next_next_turn_threats,
                a_win_next_next_next_next_turn_threats);

    detect_threats(b_k_1_threats, b_k_2_threats, b_k_3_threats,
                b_win_next_next_turn_threats,
                b_win_next_next_next_turn_threats,
                b_win_next_next_next_next_turn_threats);

    const auto to_moves = [&state](const std::set<Coord>& coords) {
        std::vector<GameState::Move> out;
        out.reserve(coords.size());
        for (const auto& c : coords) {
            out.emplace_back(c.first, c.second);
        }
        return out;
    };

    // Search immediate wins and threats first, then other remaining moves
    std::set<Coord> searched_moves;
    while (true) {
        if (!a_win_next_turn_threats.empty()) {
            searched_moves.insert(a_win_next_turn_threats.begin(), a_win_next_turn_threats.end());
            search_moves(
                state, depth_remaining, alpha, beta,
                z_index, ff, ff_branch_max, h,
                a_piece, to_moves(a_win_next_turn_threats),
                best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            break;
        }

        if (!b_win_next_turn_threats.empty()) {
            searched_moves.insert(b_win_next_turn_threats.begin(), b_win_next_turn_threats.end());
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(b_win_next_turn_threats), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            if (best_result == 0) {
                best_result = 1;
            }
            break;
        }

        if (!a_win_next_next_turn_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : a_win_next_next_turn_threats) {
                moves.insert(t.root);
            }
            searched_moves.insert(moves.begin(), moves.end());
            bool stop_search = search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(moves), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            if (stop_search) break;
        }

        if (!b_win_next_next_turn_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : b_win_next_next_turn_threats) {
                moves.insert(t.root);
                add_children_to_set(t.children, moves);
            }
            for (const auto& t : a_k_1_threats) {
                moves.insert(std::get<0>(t));
                moves.insert(std::get<1>(t));
            }
            searched_moves.insert(moves.begin(), moves.end());
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(moves), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            break;
        }

        if (!a_win_next_next_next_turn_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : a_win_next_next_next_turn_threats) {
                moves.insert(t.root);
            }
            searched_moves.insert(moves.begin(), moves.end());
            bool stop_search = search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(moves), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            if (stop_search) break;
        }

        if (!b_win_next_next_next_turn_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : b_win_next_next_next_turn_threats) {
                const Coord overlap = t.root;
                const auto& refs = t.refs;
                moves.insert(overlap);
                for (const auto& ref : refs) {
                    moves.insert(ref.root);
                    add_children_to_set(ref.children, moves);
                }
            }
            for (const auto& t : a_k_1_threats) {
                moves.insert(std::get<0>(t));
                moves.insert(std::get<1>(t));
            }
            for (const auto& t : a_k_2_threats) {
                moves.insert(std::get<0>(t));
                moves.insert(std::get<1>(t));
                moves.insert(std::get<2>(t));
            }
            searched_moves.insert(moves.begin(), moves.end());
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(moves), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            break;
        }

        if (!a_win_next_next_next_next_turn_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : a_win_next_next_next_next_turn_threats) {
                moves.insert(t.root);
            }
            searched_moves.insert(moves.begin(), moves.end());
            bool stop_search = search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(moves), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            if (stop_search) break;
        }

        if (!b_win_next_next_next_next_turn_threats.empty()) {
            std::set<Coord> moves;

            for (const auto& t : b_win_next_next_next_next_turn_threats) {
                // Python: first, b_win_nnnt_t = b_win_nnnnt_t
                //         moves.add(first)
                moves.insert(t.root);

                // Python: second, b_win_nnt_t_ref = b_win_nnnt_t
                //         moves.add(second)
                moves.insert(t.first_ref);

                // Python: for ref in b_win_nnt_t_ref
                //             moves.add(ref[0])
                //             moves.update(ref[1])
                for (const auto& ref : t.refs) {
                    moves.insert(ref.root);
                    add_children_to_set(ref.children, moves);
                }
            }
            for (const auto& t : a_k_1_threats) {
                moves.insert(std::get<0>(t));
                moves.insert(std::get<1>(t));
            }
            for (const auto& t : a_k_2_threats) {
                moves.insert(std::get<0>(t));
                moves.insert(std::get<1>(t));
                moves.insert(std::get<2>(t));
            }
            searched_moves.insert(moves.begin(), moves.end());
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(moves), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            break;
        }

        if (!a_k_1_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : a_k_1_threats) {
                moves.insert(std::get<0>(t));
                moves.insert(std::get<1>(t));
            }
            std::set<Coord> filtered;
            std::set_difference(moves.begin(), moves.end(), searched_moves.begin(), searched_moves.end(), std::inserter(filtered, filtered.begin()));
            searched_moves.insert(filtered.begin(), filtered.end());
            bool stop_search = search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            if (stop_search) break;
        }

        if (!a_k_2_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : a_k_2_threats) {
                moves.insert(std::get<0>(t));
                moves.insert(std::get<1>(t));
                moves.insert(std::get<2>(t));
            }
            std::set<Coord> filtered;
            std::set_difference(moves.begin(), moves.end(), searched_moves.begin(), searched_moves.end(), std::inserter(filtered, filtered.begin()));
            searched_moves.insert(filtered.begin(), filtered.end());
            bool stop_search = search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            if (stop_search) break;
        }

        if (!a_k_3_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : a_k_3_threats) {
                moves.insert(std::get<0>(t));
                moves.insert(std::get<1>(t));
                moves.insert(std::get<2>(t));
                moves.insert(std::get<3>(t));
            }
            std::set<Coord> filtered;
            std::set_difference(moves.begin(), moves.end(), searched_moves.begin(), searched_moves.end(), std::inserter(filtered, filtered.begin()));
            searched_moves.insert(filtered.begin(), filtered.end());
            bool stop_search = search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            if (stop_search) break;
        }

        if (!b_k_1_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : b_k_1_threats) {
                moves.insert(std::get<0>(t));
                moves.insert(std::get<1>(t));
            }
            std::set<Coord> filtered;
            std::set_difference(moves.begin(), moves.end(), searched_moves.begin(), searched_moves.end(), std::inserter(filtered, filtered.begin()));
            searched_moves.insert(filtered.begin(), filtered.end());
            bool stop_search = search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            if (stop_search) break;
        }

        if (!b_k_2_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : b_k_2_threats) {
                moves.insert(std::get<0>(t));
                moves.insert(std::get<1>(t));
                moves.insert(std::get<2>(t));
            }
            std::set<Coord> filtered;
            std::set_difference(moves.begin(), moves.end(), searched_moves.begin(), searched_moves.end(), std::inserter(filtered, filtered.begin()));
            searched_moves.insert(filtered.begin(), filtered.end());
            bool stop_search = search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            if (stop_search) break;
        }

        if (!b_k_3_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : b_k_3_threats) {
                moves.insert(std::get<0>(t));
                moves.insert(std::get<1>(t));
                moves.insert(std::get<2>(t));
                moves.insert(std::get<3>(t));
            }
            std::set<Coord> filtered;
            std::set_difference(moves.begin(), moves.end(), searched_moves.begin(), searched_moves.end(), std::inserter(filtered, filtered.begin()));
            searched_moves.insert(filtered.begin(), filtered.end());
            bool stop_search = search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            if (stop_search) break;
        }

        std::set<Coord> adjacent_1_moves;
        std::set<Coord> adjacent_2_moves;
        std::vector<Coord> remaining_moves;
        sort_remaining_moves(state, h, w, searched_moves, adjacent_1_moves, adjacent_2_moves, remaining_moves);

        if (!adjacent_1_moves.empty()) {
            searched_moves.insert(adjacent_1_moves.begin(), adjacent_1_moves.end());
            bool stop_search = search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(adjacent_1_moves), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            if (stop_search) break;
        }

        if (!adjacent_2_moves.empty()) {
            searched_moves.insert(adjacent_2_moves.begin(), adjacent_2_moves.end());
            bool stop_search = search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(adjacent_2_moves), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            if (stop_search) break;
        }

        std::map<GameState::Move, double> move_priorities;
        for (const auto& move : remaining_moves) {
            if (!state.is_valid_move(move.first, move.second)) {
                continue;
            }
            const GameState temp_state = state.make_move({move.first, move.second});
            move_priorities[{move.first, move.second}] = static_eval(temp_state);
        }

        std::vector<GameState::Move> moves_to_search;
        moves_to_search.reserve(move_priorities.size());
        for (const auto& [move, value] : move_priorities) {
            (void)value;
            moves_to_search.push_back(move);
        }
        std::stable_sort(
            moves_to_search.begin(),
            moves_to_search.end(),
            [&](const GameState::Move& lhs, const GameState::Move& rhs) {
                const double lhs_value = move_priorities.at(lhs);
                const double rhs_value = move_priorities.at(rhs);

                return sign > 0.0
                    ? lhs_value > rhs_value
                    : lhs_value < rhs_value;
            }
        );

        search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
            moves_to_search, best_move, best_value, best_deadline, best_result, best_fff,
            false, z_hashing, timeout, timed_out);
        break;
    }

    // Defensive fallback: a non-empty threat set can contain only stale or
    // placeholder coordinates. If that leaves no move selected, search every
    // legal square rather than returning {-1, -1} as a move.
    if (!(timed_out && *timed_out) &&
        !state.is_valid_move(best_move.first, best_move.second)) {
        std::vector<GameState::Move> legal_moves;
        legal_moves.reserve(static_cast<std::size_t>(h * w));
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                if (state.is_valid_move(r, c)) {
                    legal_moves.emplace_back(r, c);
                }
            }
        }
        if (!legal_moves.empty()) {
            search_moves(state, depth_remaining, alpha, beta, z_index,
                ff, ff_branch_max, h, a_piece, legal_moves,
                best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
        }
    }

    if (timed_out && *timed_out) {
        std::cerr << "Search timed out\n";
        exit(1);
    }
    if (z_hashing != nullptr) {
        auto& z_memory = std::get<1>(*z_hashing);
        // auto& z_key = std::get<2>(*z_hashing);

        // z_memory[z_key] = std::make_tuple(
        //     best_move,
        //     best_value,
        //     best_deadline + 1,
        //     best_result,
        //     best_fff
        // );
        store_zobrist_entry(
            state, std::get<0>(*z_hashing), z_memory,
            best_move, best_value, best_deadline + 1,
            best_result, best_fff
        );
    }

    return SearchResult{
        best_move,
        best_value,
        best_deadline + 1,
        best_result,
        best_fff,
        true
    };
}


// sort_remaining_moves --
void MinimaxAgent::sort_remaining_moves(
    const GameState& state,
    int h,
    int w,
    const std::set<Coord>& searched_moves,
    std::set<Coord>& adjacent_1_moves,
    std::set<Coord>& adjacent_2_moves,
    std::vector<Coord>& remaining_moves
) const {
    std::set<Coord> empty_moves;
    std::set<Coord> occupied_moves;

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            const Coord square{r, c};

            if (state.board()[r][c] == GameState::EMPTY_PIECE) {
                if (searched_moves.count(square) == 0) {
                    empty_moves.insert(square);
                }
            } else {
                occupied_moves.insert(square);
            }
        }
    }

    for (const auto& occupied : occupied_moves) {
        // Immediate neighbours: Chebyshev distance 1.
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) {
                    continue;
                }

                const Coord candidate{
                    occupied.first + dr,
                    occupied.second + dc
                };

                if (candidate.first >= 0 && candidate.first < h &&
                    candidate.second >= 0 && candidate.second < w &&
                    empty_moves.count(candidate) != 0) {
                    adjacent_1_moves.insert(candidate);
                }
            }
        }

        // Second neighbours: the full ring at Chebyshev distance 2.
        for (int dr = -2; dr <= 2; ++dr) {
            for (int dc = -2; dc <= 2; ++dc) {
                if (std::max(std::abs(dr), std::abs(dc)) != 2) {
                    continue;
                }

                const Coord candidate{
                    occupied.first + dr,
                    occupied.second + dc
                };

                if (candidate.first >= 0 && candidate.first < h &&
                    candidate.second >= 0 && candidate.second < w &&
                    empty_moves.count(candidate) != 0) {
                    adjacent_2_moves.insert(candidate);
                }
            }
        }
    }

    // A square classified as an immediate neighbour must not also be
    // classified as a second neighbour.
    for (const auto& move : adjacent_1_moves) {
        adjacent_2_moves.erase(move);
    }

    for (const auto& move : empty_moves) {
        if (adjacent_1_moves.count(move) == 0 &&
            adjacent_2_moves.count(move) == 0) {
            remaining_moves.push_back(move);
        }
    }
}


// eval_move --
bool MinimaxAgent::eval_move(
    double& alpha,
    double& beta,
    char a_piece,
    GameState::Move& best_move,
    double& best_value,
    int& best_deadline,
    int& best_result,
    int& best_fff,
    const GameState::Move& move,
    double value,
    int deadline,
    int result,
    int fff
) const {
    bool stop_search = false;

    if (a_piece == GameState::X_PIECE) {
        if (value > best_value ||
                (best_result == 3 && result == 3 && deadline > best_deadline) ||
                (best_result == 2 && result == 2 && deadline < best_deadline)) {
            best_move = move;
            best_value = value;
            best_deadline = deadline;
            best_result = result;
            best_fff = fff;
            if (result == 2)
                stop_search = true;
        }
        if (beta != kInf && best_value > beta)
            stop_search = true;
        alpha = std::max(alpha, best_value);
    } else if (a_piece == GameState::O_PIECE) {
        if (value < best_value ||
                (best_result == 2 && result == 2 && deadline > best_deadline) ||
                (best_result == 3 && result == 3 && deadline < best_deadline)) {
            best_move = move;
            best_value = value;
            best_deadline = deadline;
            best_result = result;
            best_fff = fff;
            if (result == 3)
                stop_search = true;
        }
        if (alpha != -kInf && best_value < alpha)
            stop_search = true;
        beta = std::min(beta, best_value);
    } else {
        std::cerr << "Invalid player piece: " << a_piece << "\n";
        exit(1);
    }

    return stop_search;
}

// search_moves --
bool MinimaxAgent::search_moves(
    const GameState& state,
    int depth_remaining,
    double& alpha,
    double& beta,
    int z_index,
    int ff,
    int ff_branch_max,
    int h,
    char a_piece,
    const std::vector<GameState::Move>& moves_to_search,
    GameState::Move& best_move,
    double& best_value,
    int& best_deadline,
    int& best_result,
    int& best_fff,
    bool can_fast_forward,
    ZobristHashing* z_hashing,
    const TimePoint* timeout,
    bool* timed_out
) const {
    (void)h;

    constexpr auto wrapup = std::chrono::milliseconds(100);

    bool stop_search = false;

    for (const auto& move : moves_to_search) {

        // Iterate until all spaces have been tried, exit early if time limit is reached
        if ((timed_out && *timed_out) ||
            (timeout && Clock::now() >= *timeout - wrapup)) {
            if (timed_out) {
                *timed_out = true;
            }
            break;
        }

        // Invalid generated candidates should be skipped rather than allowed
        // to corrupt the search or cause an invalid board transition.
        if (!state.is_valid_move(move)) {
            std::cerr << "Invalid move generated: ("
                << move.first << ", " << move.second << ")\n";
            exit(1);
        }


        auto& z_table = std::get<0>(*z_hashing);
        auto& z_memory = std::get<1>(*z_hashing);
        auto& z_key = std::get<2>(*z_hashing);
        const std::uint64_t old_z_key = z_key;

        // Play A in square (i,j), update Zobrist hash
        const GameState new_state = state.make_move(move);
        const std::uint64_t new_z_key = step_zobrist_key(old_z_key, z_table,move.first, move.second, z_index, state.cols());

        // Find the value of the new state.
        SearchResult child_result;

        if (!lookup_zobrist_entry(new_state, z_table, z_memory, child_result)) {
            // Cache miss: search the new state as before.
            z_key = new_z_key;
            const int new_depth_remaining = can_fast_forward ? depth_remaining : depth_remaining - 1;
            const int new_ff = can_fast_forward ? ff + 1 : ff;
            const int new_ff_branch_max = can_fast_forward ? std::max(0, ff_branch_max - 1) : ff_branch_max;

            try {
                child_result = minimax(new_state, new_depth_remaining,
                    alpha, beta, z_hashing,
                    new_ff, new_ff_branch_max,
                    timeout, timed_out
                );
            } catch (...) {
                std::cerr << "Error occurred during minimax search for move ("
                    << move.first << ", " << move.second << ")\n";
                exit(1);
            }

            z_key = old_z_key;
        }

        // Exit early if reached time limit
        if (!child_result.completed) {
            if (timed_out)
                *timed_out = true;
            break;
        }

        // Update best move, alpha and beta
        stop_search = eval_move(
            alpha, beta, a_piece, best_move, best_value, best_deadline, best_result, best_fff,
            move, child_result.value, child_result.deadline, child_result.result, child_result.fff
        );
        if (stop_search) {
            break;
        }
    }

    best_result = (best_result == 1) ? 0 : best_result;
    return stop_search;
}


// static_eval --
double MinimaxAgent::static_eval(const GameState& state) const {
    return static_eval_result(state).value;
}

// static_eval_result --
MinimaxAgent::EvalResult MinimaxAgent::static_eval_result(const GameState& state) const {
    //
    // Evaluates the given state. States good for X should be larger that states good for O.
    // :param state: state to evaluate
    // :return: evaluation of the state
    //

    ++static_eval_calls;

    const int h = state.rows();
    const int w = state.cols();
    const int k = state.k_in_row();
    const std::vector<std::pair<int, int>> directions{{1, 0}, {0, 1}, {1, 1}, {-1, 1}};

    const double win_value = std::pow(10.0, static_cast<double>(k + 21));

    char a_piece = state.next_player();
    char b_piece = (a_piece == GameState::X_PIECE) ? GameState::O_PIECE : GameState::X_PIECE;
    double a_sign = (a_piece == GameState::X_PIECE) ? 1.0 : -1.0;
    double b_sign = -a_sign;
    double a_value = 0.0;
    double b_value = 0.0;
    int a_wins = 0;
    int b_wins = 0;

    std::set<Coord> a_win_next_turn_threats;
    std::set<Coord> b_win_next_turn_threats;
    std::set<Threat1> a_k_1_threats;
    std::set<Threat1> b_k_1_threats;
    std::set<Threat2> a_k_2_threats;
    std::set<Threat2> b_k_2_threats;
    std::set<Threat3> a_k_3_threats;
    std::set<Threat3> b_k_3_threats;

    std::set<WinNextNextTurnThreat> a_win_next_next_turn_threats;
    std::set<WinNextNextTurnThreat> b_win_next_next_turn_threats;
    std::set<WinNextNextNextTurnThreat> a_win_next_next_next_turn_threats;
    std::set<WinNextNextNextTurnThreat> b_win_next_next_next_turn_threats;
    std::set<WinNextNextNextNextTurnThreat> a_win_next_next_next_next_turn_threats;
    std::set<WinNextNextNextNextTurnThreat> b_win_next_next_next_next_turn_threats;

    process_wins_and_threats(state, h, w, k, directions,
        a_piece, b_piece,
        a_win_next_turn_threats, b_win_next_turn_threats,
        a_k_1_threats, b_k_1_threats,
        a_k_2_threats, b_k_2_threats,
        a_k_3_threats, b_k_3_threats,
        a_value, b_value, a_wins, b_wins);

    const double a_win_value = (a_piece == GameState::X_PIECE) ? win_value : -win_value;
    const double b_win_value = -a_win_value;
    const int a_win_result = (a_piece == GameState::X_PIECE) ? 2 : 3;
    const int b_win_result = (b_piece == GameState::O_PIECE) ? 3 : 2;

    //
    // ===============================================
    // ====== A TO PLAY NEXT, B HAS JUST PLAYED ======
    // ===============================================
    //             Predict any forced wins
    //
    if (a_wins >= 1) {
        // A won on the previous turn
        // A wins in 0 moves
        return EvalResult{a_win_value, 0, a_win_result};
    }
    if (b_wins >= 1) {
        // B has just won on this turn
        // B wins in 1 move (B)
        return EvalResult{b_win_value, 1, b_win_result};
    }
    if (!a_win_next_turn_threats.empty()) {
        // A will win on the next turn
        // A wins in 2 moves (BA)
        return EvalResult{a_win_value, 2, a_win_result};
    }
    if (b_win_next_turn_threats.size() >= 2) {
        // Multiple ways for B to win, A can only block 1 on their turn, B wins next turn
        // B wins in 3 moves (BAB)
        return EvalResult{b_win_value, 3, b_win_result};
    }

    detect_threats(a_k_1_threats, a_k_2_threats, a_k_3_threats,
        a_win_next_next_turn_threats,
        a_win_next_next_next_turn_threats,
        a_win_next_next_next_next_turn_threats);
    detect_threats(b_k_1_threats, b_k_2_threats, b_k_3_threats,
        b_win_next_next_turn_threats,
        b_win_next_next_next_turn_threats,
        b_win_next_next_next_next_turn_threats);

    return calculate_value(state, h, w, k, directions,
        a_piece, b_piece, a_sign, b_sign,
        a_value, b_value, a_wins, b_wins,
        a_win_next_turn_threats, b_win_next_turn_threats,
        a_win_next_next_turn_threats, b_win_next_next_turn_threats,
        a_win_next_next_next_turn_threats, b_win_next_next_next_turn_threats,
        a_win_next_next_next_next_turn_threats, b_win_next_next_next_next_turn_threats,
        a_k_1_threats, b_k_1_threats,
        a_k_2_threats, b_k_2_threats,
        a_k_3_threats, b_k_3_threats,
        a_win_value, b_win_value,
        a_win_result, b_win_result);
}

// calculate_value --
MinimaxAgent::EvalResult MinimaxAgent::calculate_value(const GameState& state, int h, int w, int k,
    const std::vector<std::pair<int,int>>& directions,
    char a_piece, char b_piece, double a_sign, double b_sign,
    double a_value, double b_value, int a_wins, int b_wins,
    const std::set<MinimaxAgent::Coord>& a_win_next_turn_threats,
    const std::set<MinimaxAgent::Coord>& b_win_next_turn_threats,
    const std::set<MinimaxAgent::WinNextNextTurnThreat>& a_win_next_next_turn_threats,
    const std::set<MinimaxAgent::WinNextNextTurnThreat>& b_win_next_next_turn_threats,
    const std::set<MinimaxAgent::WinNextNextNextTurnThreat>& a_win_next_next_next_turn_threats,
    const std::set<MinimaxAgent::WinNextNextNextTurnThreat>& b_win_next_next_next_turn_threats,
    const std::set<MinimaxAgent::WinNextNextNextNextTurnThreat>& a_win_next_next_next_next_turn_threats,
    const std::set<MinimaxAgent::WinNextNextNextNextTurnThreat>& b_win_next_next_next_next_turn_threats,
    const std::set<MinimaxAgent::Threat1>& a_k_1_threats,
    const std::set<MinimaxAgent::Threat1>& b_k_1_threats,
    const std::set<MinimaxAgent::Threat2>& a_k_2_threats,
    const std::set<MinimaxAgent::Threat2>& b_k_2_threats,
    const std::set<MinimaxAgent::Threat3>& a_k_3_threats,
    const std::set<MinimaxAgent::Threat3>& b_k_3_threats,
    double a_win_value, double b_win_value,
    int a_win_result, int b_win_result) const {

    // At this point:
    //     len(a_wins) = 0
    //     len(b_wins) = 0
    //     len(a_win_next_turn_threats) = 0
    //     len(b_win_next_turn_threats) <= 1

    GameState state_mut = state;

    std::set<Coord> a_win_next_turn = a_win_next_turn_threats;
    std::set<Coord> b_win_next_turn = b_win_next_turn_threats;
    auto a_win_next_next = a_win_next_next_turn_threats;
    auto b_win_next_next = b_win_next_next_turn_threats;
    auto a_win_next_next_next = a_win_next_next_next_turn_threats;
    auto b_win_next_next_next = b_win_next_next_next_turn_threats;
    auto a_win_next_next_next_next = a_win_next_next_next_next_turn_threats;
    auto b_win_next_next_next_next = b_win_next_next_next_next_turn_threats;

    std::set<Threat1> a_k_1 = a_k_1_threats;
    std::set<Threat1> b_k_1 = b_k_1_threats;
    std::set<Threat2> a_k_2 = a_k_2_threats;
    std::set<Threat2> b_k_2 = b_k_2_threats;
    std::set<Threat3> a_k_3 = a_k_3_threats;
    std::set<Threat3> b_k_3 = b_k_3_threats;

    double value = 0.0;
    int fast_forward_counter = 0;

    // Fast forward through any series of forced moves (ff)
    while (true) {
        if (b_win_next_turn.size() == 1) {
            Coord blocking_square = *b_win_next_turn.begin();
            b_win_next_turn.erase(b_win_next_turn.begin());
            state_mut = state_mut.make_move(blocking_square);
            ++fast_forward_counter;

            update_threats(blocking_square, a_k_1, a_k_2, a_win_next_turn, a_win_next_next, a_win_next_next_next, a_win_next_next_next_next);

            for (const auto& [dx, dy] : directions) {
                for (int scan = 1 - k; scan < 1; ++scan) {
                    int i = blocking_square.first + scan * dx;
                    int j = blocking_square.second + scan * dy;
                    std::vector<Coord> last(4, {-10, -10});
                    int a_count = 0;
                    int b_count = 0;
                    bool a_blocked = false;
                    bool b_blocked = false;

                    for (int step = 0; step < k; ++step) {
                        if (!(0 <= i && i < h && 0 <= j && j < w)) {
                            a_blocked = true;
                            b_blocked = true;
                            break;
                        }

                        const char current_piece = state_mut.board()[i][j];
                        if (current_piece == '-') {
                            a_blocked = true;
                            b_blocked = true;
                            break;
                        } else if (current_piece == b_piece) {
                            ++b_count;
                            a_blocked = true;
                        } else if (current_piece == a_piece) {
                            ++a_count;
                            if (Coord(i, j) != blocking_square) {
                                b_blocked = true;
                            }
                        } else {
                            last = {Coord(i, j), last[0], last[1], last[2]};
                        }

                        i += dx;
                        j += dy;
                    }

                    if (b_count > 0 && !b_blocked) {
                        b_value -= std::pow(10.0, b_count - 1);
                    }
                    if (a_count > 0 && !a_blocked) {
                        update_wins_and_threats(k, last, a_count, a_value, a_wins,
                            a_win_next_turn, a_k_1, a_k_2, a_k_3);
                    }
                }
            }

            detect_threats(a_k_1, a_k_2, a_k_3, a_win_next_next, a_win_next_next_next, a_win_next_next_next_next);
            block_threats(blocking_square, b_k_1, b_k_2, b_k_3, b_win_next_next, b_win_next_next_next, b_win_next_next_next_next);

            std::swap(a_piece, b_piece);
            std::swap(a_sign, b_sign);
            std::swap(a_value, b_value);
            std::swap(a_wins, b_wins);
            std::swap(a_win_next_turn, b_win_next_turn);
            std::swap(a_win_next_next, b_win_next_next);
            std::swap(a_win_next_next_next, b_win_next_next_next);
            std::swap(a_win_next_next_next_next, b_win_next_next_next_next);
            std::swap(a_k_1, b_k_1);
            std::swap(a_k_2, b_k_2);
            std::swap(a_k_3, b_k_3);
            std::swap(a_win_value, b_win_value);
            std::swap(a_win_result, b_win_result);
        } else {
            if (b_win_next_turn.size() > 1) {
                // B has multiple wins on their next turn,
                // A can only block one of them, B wins on the following turn
                // B wins in 3 moves after fast forward (ff-BAB)
                return EvalResult{b_win_value, 3 + fast_forward_counter, b_win_result};
            }
            break;
        }
    }

    if (!a_win_next_next.empty()) {
        // A can play a piece such that they have multiple wins on their next turn,
        // B can only block one of them, A wins on the following turn
        // A wins in 4 moves after fast forward (ff-BABA)
        return EvalResult{a_win_value, 4 + fast_forward_counter, a_win_result};
    }

    if (!b_win_next_next.empty() && a_k_1.empty()) {
        bool found_win = false;
        for (const auto& w1 : b_win_next_next_turn_threats) {
            if (found_win) {
                break;
            }

            CoordSet w1_children;
            add_children_to_set(w1.children, w1_children);

            for (const auto& w2 : b_win_next_next_turn_threats) {
                // Equivalent to Python's difference({w1}).
                if (w1 == w2) {
                    continue;
                }

                CoordSet w2_children;
                add_children_to_set(w2.children, w2_children);

                if (w1.root != w2.root
                    && w2_children.count(w1.root) == 0
                    && w1_children.count(w2.root) == 0
                    && coord_intersection(w1_children, w2_children).empty()) {
                    found_win = true;
                    break;
                }
            }
        }
        if (found_win) {
            return EvalResult{b_win_value, 5 + fast_forward_counter, b_win_result};
        }
    }

    if (!a_win_next_next_next.empty()) {
        bool found_win = false;
        if (b_k_1.empty()) {
            found_win = true;
        } else if (!a_k_1.empty()) {
            std::map<Coord, Coord> b_k_1_threat_squares;
            for (const auto& t1 : b_k_1) {
                b_k_1_threat_squares[std::get<0>(t1)] = std::get<1>(t1);
                b_k_1_threat_squares[std::get<1>(t1)] = std::get<0>(t1);
            }

            for (const auto& t1 : a_k_1) {
                if (found_win) break;
                const Coord t1a = std::get<0>(t1);
                const Coord t1b = std::get<1>(t1);

                for (const auto& w1 : a_win_next_next_next) {
                    if (found_win) {
                        break;
                    }

                    const Coord& root = w1.root;
                    const auto& refs = w1.refs;

                    // Python:
                    // w1[0] in t1
                    // or t1[0] == w1[1][0][0]
                    // or (len(w1[1]) == 2 and t1[1] == w1[1][1][0])
                    const bool match_root =
                        (root == t1a || root == t1b);

                    const bool match_first =
                        !refs.empty() &&
                        (refs[0].root == t1a || refs[0].root == t1b);

                    const bool match_second =
                        refs.size() == 2 &&
                        (refs[1].root == t1b);

                    if (!(match_root || match_first || match_second)) {
                        continue;
                    }

                    const Coord b_forced_square =
                        (root == t1a) ? t1b : t1a;

                    std::set<Coord> next_a_squares;

                    for (const auto& ref : refs) {
                        next_a_squares.insert(ref.root);
                    }

                    auto it = b_k_1_threat_squares.find(b_forced_square);

                    if (it == b_k_1_threat_squares.end() ||
                        next_a_squares.count(it->second) > 0) {
                        found_win = true;
                    }
                }
            }
        }
        if (found_win) {
            return EvalResult{a_win_value, 6 + fast_forward_counter, a_win_result};
        }
    }

    if (!b_win_next_next_next.empty() && a_k_1.empty()) {
        bool found_win = false;

        // Compare a two-step threat against a three-step threat.
        for (const auto& w1 : b_win_next_next_turn_threats) {
            if (found_win) {
                break;
            }

            for (const auto& w2 : b_win_next_next_next_turn_threats) {
                if (found_win) {
                    break;
                }

                const Coord& root1 = w1.root;
                const auto& children1 = w1.children;
                const Coord& root2 = w2.root;

                const CoordSet w2_squares = get_squares_from_wn3_threat(w2);

                // Collect all children of w1 into a set.
                CoordSet w1_children;
                add_children_to_set(children1, w1_children);

                if (root1 != root2
                    && w2_squares.count(root1) == 0
                    && w1_children.count(root2) == 0
                    && coord_intersection(w1_children, w2_squares).empty()) {

                    found_win = true;

                    for (const auto& t1 : a_k_2_threats) {
                        const CoordSet t1_squares{
                            std::get<0>(t1),
                            std::get<1>(t1),
                            std::get<2>(t1)
                        };

                        if (t1_squares.count(root1) > 0
                            || !coord_intersection(w1_children, t1_squares).empty()) {
                            found_win = false;
                            break;
                        }
                    }
                }
            }
        }

        // Compare pairs of three-step threats.
        if (!found_win && b_win_next_next_next.size() >= 2) {
            std::set<WinNextNextNextTurnThreat> completed;

            for (const auto& w1 : b_win_next_next_next_turn_threats) {
                completed.insert(w1);

                if (found_win) {
                    break;
                }

                for (const auto& w2 : b_win_next_next_next_turn_threats) {
                    if (completed.count(w2) > 0) {
                        continue;
                    }

                    if (found_win) {
                        break;
                    }

                    const CoordSet w1_squares = get_squares_from_wn3_threat(w1);
                    const CoordSet w2_squares = get_squares_from_wn3_threat(w2);

                    if ((w1.refs.size() == 2 || w2.refs.size() == 2)
                        && !a_k_2_threats.empty()) {
                        break;
                    }

                    if (w1.root != w2.root
                        && w2_squares.count(w1.root) == 0
                        && w1_squares.count(w2.root) == 0
                        && coord_intersection(w1_squares, w2_squares).empty()) {
                        found_win = true;
                    }
                }
            }
        }

        if (found_win) {
            return EvalResult{
                b_win_value,
                7 + fast_forward_counter,
                b_win_result
            };
        }
    }

    if (!a_win_next_next_next_next.empty()) {
        bool found_win = (b_k_1.empty());
        if (found_win) {
            return EvalResult{a_win_value, 8 + fast_forward_counter, a_win_result};
        }
    }

if (value == 0.0 &&
    !b_win_next_next_next_next.empty() &&
    a_k_1.empty()) {

    bool found_win = false;

    for (const auto& w1 : b_win_next_next_turn_threats) {
        if (found_win) {
            break;
        }

        for (const auto& w2 : b_win_next_next_next_next_turn_threats) {
            if (found_win) {
                break;
            }

            const Coord& root1 = w1.root;
            const CoordSet w2_squares = get_squares_from_wn4_threat(w2);

            CoordSet w1_children;
            add_children_to_set(w1.children, w1_children);

            if (root1 != w2.root
                && w2_squares.count(root1) == 0
                && w1_children.count(w2.root) == 0
                && coord_intersection(w1_children, w2_squares).empty()) {

                found_win = true;

                for (const auto& t1 : a_k_2_threats) {
                    const CoordSet t1_squares{
                        std::get<0>(t1),
                        std::get<1>(t1),
                        std::get<2>(t1)
                    };

                    if (t1_squares.count(root1) > 0
                        || !coord_intersection(w1_children, t1_squares).empty()) {
                        found_win = false;
                    }
                }
            }
        }
    }

    if (found_win) {
        return EvalResult{
            b_win_value,
            9 + fast_forward_counter,
            b_win_result
        };
    }
}

    return EvalResult{std::round(a_sign * a_value + b_sign * b_value), 0, 0};
}