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

// make_zobrist_hashing --
MinimaxAgent::ZobristHashing MinimaxAgent::make_zobrist_hashing(int rows, int cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::uint32_t> dist(0u, std::numeric_limits<std::uint32_t>::max());

    ZobristTable z_table(static_cast<std::size_t>(rows * cols), std::vector<std::uint32_t>(2, 0u));
    for (auto& entry : z_table) {
        entry[0] = dist(rng);
        entry[1] = dist(rng);
    }
    return std::make_tuple(z_table, ZobristMemory{}, std::uint64_t{0});
}

// step_zobrist_key --
std::uint64_t MinimaxAgent::step_zobrist_key(std::uint64_t current_key, const ZobristTable& z_table, int row, int col, int z_index,
    int board_cols) {
    if (row < 0 || col < 0) {
        return current_key;
    }

    int width = board_cols;
    if (width <= 0) {
        width = static_cast<int>(std::sqrt(static_cast<double>(z_table.size())));
    }
    if (width <= 0) {
        return current_key;
    }

    const std::size_t offset = static_cast<std::size_t>(row * width + col);
    if (offset >= z_table.size()) {
        return current_key;
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
        if (last[0].first != -1 && last[0].second != -1) {
            win_next_turn_threats.insert(last[0]);
        }
        return;
    }
    if (count == k - 2) {
        // k-2 out of k squares are filled, player can fill to k-1 next turn
        if (last[0].first != -1 && last[0].second != -1 && last[1].first != -1 && last[1].second != -1) {
            k_1_threats.insert(std::make_tuple(last[0], last[1]));
        }
        return;
    }
    if (count == k - 3) {
        // k-3 out of k squares are filled, player can fill to k-2 next turn
        if (last[0].first != -1 && last[0].second != -1 && last[1].first != -1 && last[1].second != -1 &&
            last[2].first != -1 && last[2].second != -1) {
            k_2_threats.insert(std::make_tuple(last[0], last[1], last[2]));
        }
        return;
    }
    if (count == k - 4) {
        // k-4 out of k squares are filled, player can fill to k-3 next turn
        if (last[0].first != -1 && last[0].second != -1 && last[1].first != -1 && last[1].second != -1 &&
            last[2].first != -1 && last[2].second != -1 && last[3].first != -1 && last[3].second != -1) {
            k_3_threats.insert(std::make_tuple(last[0], last[1], last[2], last[3]));
        }
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
                std::vector<Coord> last(4, {-1, -1});

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
                    update_wins_and_threats(k, last, a_count, a_value, a_wins,
                        a_win_next_turn_threats, a_k_1_threats, a_k_2_threats, a_k_3_threats);
                } else if (b_count > 0 && !b_blocked) {
                    update_wins_and_threats(k, last, b_count, b_value, b_wins,
                        b_win_next_turn_threats, b_k_1_threats, b_k_2_threats, b_k_3_threats);
                }
            }
        }
    }
}

// detect_threats --
void MinimaxAgent::detect_threats(const std::set<Threat1>& k_1_threats,
    const std::set<Threat2>& k_2_threats,
    const std::set<Threat3>& k_3_threats,
    std::set<std::tuple<Coord, std::tuple<Coord, Coord>>>& win_next_next_turn_threats,
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>>& win_next_next_next_turn_threats,
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>>& win_next_next_next_next_turn_threats) const {
    auto set_intersection = [](const std::set<Coord>& lhs, const std::set<Coord>& rhs) {
        std::set<Coord> out;
        std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::inserter(out, out.begin()));
        return out;
    };

    auto set_difference = [](const std::set<Coord>& lhs, const std::set<Coord>& rhs) {
        std::set<Coord> out;
        std::set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::inserter(out, out.begin()));
        return out;
    };

    std::vector<std::set<Coord>> k_1_sets;
    std::vector<std::set<Coord>> k_2_sets;
    std::vector<std::set<Coord>> k_3_sets;

    for (const auto& t : k_1_threats) {
        k_1_sets.push_back(std::set<Coord>{std::get<0>(t), std::get<1>(t)});
    }
    for (const auto& t : k_2_threats) {
        k_2_sets.push_back(std::set<Coord>{std::get<0>(t), std::get<1>(t), std::get<2>(t)});
    }
    for (const auto& t : k_3_threats) {
        k_3_sets.push_back(std::set<Coord>{std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t)});
    }

    // Record places where two win threats can be created on the same turn
    for (const auto& s1 : k_1_sets) {
        for (const auto& s2 : k_1_sets) {
            const auto inter = set_intersection(s1, s2);
            if (inter.size() == 1) {
                const Coord overlap = *inter.begin();
                std::set<Coord> diff1 = set_difference(s1, s2);
                std::set<Coord> diff2 = set_difference(s2, s1);
                std::vector<Coord> diff1_vec(diff1.begin(), diff1.end());
                std::vector<Coord> diff2_vec(diff2.begin(), diff2.end());
                std::vector<Coord> win_next_turn;
                win_next_turn.insert(win_next_turn.end(), diff1_vec.begin(), diff1_vec.end());
                win_next_turn.insert(win_next_turn.end(), diff2_vec.begin(), diff2_vec.end());
                if (win_next_turn.size() == 2) {
                    win_next_next_turn_threats.insert(std::make_tuple(overlap, std::make_tuple(win_next_turn[0], win_next_turn[1])));
                }
            }
        }
    }

    // Record places where two parallel k-1 threats can be created on the same turn
    std::vector<std::pair<std::set<Coord>, std::set<Coord>>> overlapping_k_2_threats;
    for (const auto& s1 : k_2_sets) {
        for (const auto& s2 : k_2_sets) {
            const auto inter = set_intersection(s1, s2);
            if (inter.size() == 2) {
                std::set<Coord> edge_squares = set_difference(s1, s2);
                std::set<Coord> edge_squares2 = set_difference(s2, s1);
                edge_squares.insert(edge_squares2.begin(), edge_squares2.end());
                overlapping_k_2_threats.emplace_back(inter, edge_squares);
            }
        }
    }

    // Record places where two parallel k-2 threats can be created on the same turn
    std::vector<std::tuple<std::set<Coord>, std::set<Coord>, std::set<Coord>>> overlapping_k_3_threats;
    for (const auto& s1 : k_3_sets) {
        for (const auto& s2 : k_3_sets) {
            const auto inter = set_intersection(s1, s2);
            if (inter.size() == 2 && s1.size() != 2) {
                std::vector<Coord> overlapping_squares(inter.begin(), inter.end());
                if (overlapping_squares.size() != 2) {
                    continue;
                }
                const Coord os1 = overlapping_squares[0];
                const Coord os2 = overlapping_squares[1];
                if (std::abs(os1.first - os2.first) <= 1 && std::abs(os1.second - os2.second) <= 1) {
                    continue;
                }

                const auto other_squares = set_difference(s1, s2);
                std::set<Coord> inner_squares;
                std::set<Coord> outer_squares = set_difference(s1, s2);
                for (const auto& square : other_squares) {
                    bool is_inner = false;
                    for (const auto& overlap : inter) {
                        if (std::abs(square.first - overlap.first) <= 1 && std::abs(square.second - overlap.second) <= 1) {
                            inner_squares.insert(square);
                            is_inner = true;
                            break;
                        }
                    }
                    if (is_inner) {
                        outer_squares.erase(square);
                    }
                }
                overlapping_k_3_threats.emplace_back(inter, inner_squares, outer_squares);
            }
        }
    }

    // Record places where a win threat and a k-1 threat can be created on the same turn
    std::vector<std::tuple<Coord, Coord, std::set<Coord>>> k_1_k_2_threats;
    for (const auto& s1 : k_1_sets) {
        for (const auto& s2 : k_2_sets) {
            const auto inter = set_intersection(s1, s2);
            if (inter.size() == 1) {
                const Coord overlap = *inter.begin();
                std::set<Coord> one_remaining = set_difference(s1, inter);
                std::set<Coord> two_remaining = set_difference(s2, inter);
                k_1_k_2_threats.emplace_back(overlap, *one_remaining.begin(), two_remaining);
            }
        }
    }

    // Higher order overlapping logic
    const auto ok2 = overlapping_k_2_threats;
    const auto ok3 = overlapping_k_3_threats;
    const size_t m1 = ok2.size();
    const size_t m2 = ok3.size();

    for (size_t i = 0; i < m1; ++i) {
        const auto& s1 = ok2[i].first;
        const auto& edges1 = ok2[i].second;

        // Record places where two sets of two parallel k-1 threats can be created on the same turn
        for (size_t j = i + 1; j < m1; ++j) {
            const auto& s2 = ok2[j].first;
            const auto& edges2 = ok2[j].second;
            const auto inter = set_intersection(s1, s2);
            if (inter.size() == 1 && set_intersection(s1, edges2).empty()) {
                const Coord overlap = *inter.begin();
                std::set<Coord> s1_minus_inter = set_difference(s1, inter);
                std::set<Coord> s2_minus_inter = set_difference(s2, inter);
                const Coord a = *s1_minus_inter.begin();
                const Coord b = *s2_minus_inter.begin();
                win_next_next_next_turn_threats.insert(std::make_tuple(overlap, std::make_tuple(a, b, Coord{-1, -1})));
            }
        }

        // Record places where two parallel k-1 threats and a win threat can be created on the same turn
        for (const auto& s2 : k_1_sets) {
            const auto inter = set_intersection(s1, s2);
            if (inter.size() == 1) {
                const Coord overlap = *inter.begin();
                std::set<Coord> s1_minus_inter = set_difference(s1, inter);
                std::set<Coord> s2_minus_inter = set_difference(s2, inter);
                const Coord a = *s1_minus_inter.begin();
                const Coord forced = *s2_minus_inter.begin();
                if (forced != a && edges1.find(forced) == edges1.end()) {
                    win_next_next_next_turn_threats.insert(std::make_tuple(overlap, std::make_tuple(a, Coord{-1, -1}, Coord{-1, -1})));
                }
            }
        }

        // Something
        for (size_t j = 0; j < m2; ++j) {
            const auto& s2 = std::get<0>(ok3[j]);
            const auto& inner2 = std::get<1>(ok3[j]);
            const auto& outer2 = std::get<2>(ok3[j]);
            const auto inter1 = set_intersection(s1, s2);
            if (inter1.size() != 1 || !set_intersection(s1, inner2).empty() || !set_intersection(s1, outer2).empty()) {
                continue;
            }

            for (size_t k = 0; k < m1; ++k) {
                if (k == i) {
                    continue;
                }
                const auto& s3 = ok2[k].first;
                const auto inter2 = set_intersection(s3, inner2);
                if (inter2.size() == 1 && set_intersection(s3, s2).empty() && set_intersection(s3, outer2).empty()) {
                    const Coord overlap_a = *inter1.begin();
                    const Coord overlap_b = *inter2.begin();
                    std::set<Coord> s2_minus_overlap_a = set_difference(s2, std::set<Coord>{overlap_a});
                    std::set<Coord> s3_minus_overlap_b = set_difference(s3, std::set<Coord>{overlap_b});
                    const Coord other_square2 = *s2_minus_overlap_a.begin();
                    const Coord other_square3 = *s3_minus_overlap_b.begin();

                    win_next_next_next_next_turn_threats.insert(std::make_tuple(overlap_a, std::make_tuple(overlap_b, other_square2, other_square3, Coord{-1, -1})));
                    win_next_next_next_next_turn_threats.insert(std::make_tuple(overlap_b, std::make_tuple(overlap_a, other_square2, other_square3, Coord{-1, -1})));
                }
            }
        }
    }

    const size_t m = k_1_k_2_threats.size();
    for (size_t i = 0; i < m; ++i) {
        const Coord square1 = std::get<0>(k_1_k_2_threats[i]);
        const Coord one1 = std::get<1>(k_1_k_2_threats[i]);
        const std::set<Coord>& two1 = std::get<2>(k_1_k_2_threats[i]);
        for (size_t j = i + 1; j < m; ++j) {
            const Coord square2 = std::get<0>(k_1_k_2_threats[j]);
            const Coord one2 = std::get<1>(k_1_k_2_threats[j]);
            const std::set<Coord>& two2 = std::get<2>(k_1_k_2_threats[j]);
            const auto inter = set_intersection(two1, two2);
            if (inter.size() == 1) {
                const Coord overlap = *inter.begin();
                std::set<Coord> two1_minus_overlap = set_difference(two1, std::set<Coord>{overlap});
                std::set<Coord> two2_minus_overlap = set_difference(two2, std::set<Coord>{overlap});
                win_next_next_next_turn_threats.insert(std::make_tuple(overlap, std::make_tuple(square1, one1, *two1_minus_overlap.begin())));
                win_next_next_next_turn_threats.insert(std::make_tuple(overlap, std::make_tuple(square2, one2, *two2_minus_overlap.begin())));
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
    std::set<std::tuple<Coord, std::tuple<Coord, Coord>>>& win_next_next_turn_threats,
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>>& win_next_next_next_turn_threats,
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>>& win_next_next_next_next_turn_threats
) const {
    // detect_threats() uses negative coordinates, e.g. (-1, -1),
    // as placeholders for missing squares.
    const auto valid_coord = [](const Coord& c) {
        return c.first >= 0 && c.second >= 0;
    };

    // A placeholder must never be treated as a real move.
    if (!valid_coord(new_square)) {
        return;
    }

    // ---------------------------------------------------------
    // 1. Reduce one-square-pair threats to immediate win threats.
    // ---------------------------------------------------------
    for (auto it = k_1_threats.begin(); it != k_1_threats.end();) {
        const Coord a = std::get<0>(*it);
        const Coord b = std::get<1>(*it);

        if (a == new_square || b == new_square) {
            if (a != new_square && valid_coord(a)) {
                win_next_turn_threats.insert(a);
            }
            if (b != new_square && valid_coord(b)) {
                win_next_turn_threats.insert(b);
            }

            it = k_1_threats.erase(it);
        } else {
            // Remove malformed threats even if they weren't activated.
            if (!valid_coord(a) || !valid_coord(b)) {
                it = k_1_threats.erase(it);
            } else {
                ++it;
            }
        }
    }

    // ---------------------------------------------------------
    // 2. Reduce next-next-turn threats.
    // ---------------------------------------------------------
    for (auto it = win_next_next_turn_threats.begin();
         it != win_next_next_turn_threats.end();) {
        const Coord root = std::get<0>(*it);
        const auto& children = std::get<1>(*it);

        const Coord a = std::get<0>(children);
        const Coord b = std::get<1>(children);

        if (!valid_coord(root) || !valid_coord(a) || !valid_coord(b)) {
            it = win_next_next_turn_threats.erase(it);
            continue;
        }

        if (new_square == root) {
            if (a != new_square) {
                win_next_turn_threats.insert(a);
            }
            if (b != new_square) {
                win_next_turn_threats.insert(b);
            }

            it = win_next_next_turn_threats.erase(it);
            continue;
        }

        if (new_square == a || new_square == b) {
            win_next_turn_threats.insert(root);
            it = win_next_next_turn_threats.erase(it);
            continue;
        }

        ++it;
    }

    // ---------------------------------------------------------
    // 3. Reduce three-square threats.
    // ---------------------------------------------------------
    for (auto it = k_2_threats.begin(); it != k_2_threats.end();) {
        const Coord a = std::get<0>(*it);
        const Coord b = std::get<1>(*it);
        const Coord c = std::get<2>(*it);

        if (!valid_coord(a) || !valid_coord(b) || !valid_coord(c)) {
            it = k_2_threats.erase(it);
            continue;
        }

        if (a == new_square || b == new_square || c == new_square) {
            Coord first;
            Coord second;

            if (a == new_square) {
                first = b;
                second = c;
            } else if (b == new_square) {
                first = a;
                second = c;
            } else {
                first = a;
                second = b;
            }

            if (valid_coord(first) && valid_coord(second)
                && first != second) {
                k_1_threats.insert(std::make_tuple(first, second));
            }

            it = k_2_threats.erase(it);
            continue;
        }

        ++it;
    }

    // ---------------------------------------------------------
    // 4. Reduce next-next-next-turn threats.
    // ---------------------------------------------------------
    for (auto it = win_next_next_next_turn_threats.begin();
         it != win_next_next_next_turn_threats.end();) {
        const Coord root = std::get<0>(*it);
        const auto& refs = std::get<1>(*it);

        const Coord a = std::get<0>(refs);
        const Coord b = std::get<1>(refs);
        const Coord c = std::get<2>(refs);

        if (!valid_coord(root) ||
            !valid_coord(a) ||
            !valid_coord(b) ||
            !valid_coord(c)) {
            it = win_next_next_next_turn_threats.erase(it);
            continue;
        }

        if (new_square == root) {
            // The root move has been played. The remaining three
            // valid coordinates form a next-next-turn threat.
            win_next_next_turn_threats.insert(
                std::make_tuple(a, std::make_tuple(b, c)));

            win_next_next_next_turn_threats.erase(it++);
            continue;
        }

        if (new_square == a || new_square == b || new_square == c) {
            win_next_turn_threats.insert(root);
            it = win_next_next_next_turn_threats.erase(it);
            continue;
        }

        ++it;
    }

    // ---------------------------------------------------------
    // 5. Reduce next-next-next-next-turn threats.
    // ---------------------------------------------------------
    for (auto it = win_next_next_next_next_turn_threats.begin();
         it != win_next_next_next_next_turn_threats.end();) {
        const Coord root = std::get<0>(*it);
        const auto& refs = std::get<1>(*it);

        const Coord a = std::get<0>(refs);
        const Coord b = std::get<1>(refs);
        const Coord c = std::get<2>(refs);
        const Coord d = std::get<3>(refs);

        if (!valid_coord(root)) {
            it = win_next_next_next_next_turn_threats.erase(it);
            continue;
        }

        // Only real, distinct coordinates may be used to form
        // lower-order threats. Keep their original order.
        std::vector<Coord> valid_refs;
        for (const Coord& sq : {a, b, c, d}) {
            if (valid_coord(sq) &&
                std::find(valid_refs.begin(), valid_refs.end(), sq)
                    == valid_refs.end()) {
                valid_refs.push_back(sq);
            }
        }

        if (new_square == root) {
            // A valid next-next-turn threat requires three distinct
            // coordinates: one root and two child coordinates.
            if (valid_refs.size() >= 3) {
                for (std::size_t i = 0; i < valid_refs.size(); ++i) {
                    const Coord child_root = valid_refs[i];

                    std::vector<Coord> others;
                    for (std::size_t j = 0; j < valid_refs.size(); ++j) {
                        if (j != i) {
                            others.push_back(valid_refs[j]);
                        }
                    }

                    for (std::size_t j = 0; j < others.size(); ++j) {
                        for (std::size_t k = j + 1; k < others.size(); ++k) {
                            win_next_next_turn_threats.insert(
                                std::make_tuple(
                                    child_root,
                                    std::make_tuple(others[j], others[k])));
                        }
                    }
                }
            }

            it = win_next_next_next_next_turn_threats.erase(it);
            continue;
        }

        // If the move occupies one of the reference squares, this
        // threat can yield an immediate win at its root.
        if (std::find(valid_refs.begin(), valid_refs.end(), new_square)
            != valid_refs.end()) {
            win_next_turn_threats.insert(root);
            it = win_next_next_next_next_turn_threats.erase(it);
            continue;
        }

        // Discard malformed threats rather than retaining placeholders.
        if (valid_refs.size() != 4) {
            it = win_next_next_next_next_turn_threats.erase(it);
            continue;
        }

        ++it;
    }
}


// block_threats --
void MinimaxAgent::block_threats(const Coord& blocking_square,
    std::set<Threat1>& k_1_threats,
    std::set<Threat2>& k_2_threats,
    std::set<Threat3>& k_3_threats,
    std::set<std::tuple<Coord, std::tuple<Coord, Coord>>>& win_next_next_turn_threats,
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>>& win_next_next_next_turn_threats,
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>>& win_next_next_next_next_turn_threats) const {
    //
    // If blocking a win also blocks any other threats, remove them from storage.
    //
    for (auto it = k_1_threats.begin(); it != k_1_threats.end();) {
        const auto& t = *it;
        if (std::get<0>(t) == blocking_square || std::get<1>(t) == blocking_square) {
            it = k_1_threats.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = win_next_next_turn_threats.begin(); it != win_next_next_turn_threats.end();) {
        const auto& w = *it;
        const Coord root = std::get<0>(w);
        const auto& children = std::get<1>(w);
        std::set<Coord> child_set{std::get<0>(children), std::get<1>(children)};
        if (blocking_square == root || child_set.count(blocking_square) > 0) {
            it = win_next_next_turn_threats.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = k_2_threats.begin(); it != k_2_threats.end();) {
        const auto& t = *it;
        if (std::get<0>(t) == blocking_square || std::get<1>(t) == blocking_square || std::get<2>(t) == blocking_square) {
            it = k_2_threats.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = win_next_next_next_turn_threats.begin(); it != win_next_next_next_turn_threats.end();) {
        const auto& w = *it;
        const Coord root = std::get<0>(w);
        const auto& refs = std::get<1>(w);
        const Coord ref0 = std::get<0>(refs);
        const Coord ref1 = std::get<1>(refs);
        const Coord ref2 = std::get<2>(refs);
        const std::set<Coord> child_set{ref0, ref1, ref2};
        if (blocking_square == root || child_set.count(blocking_square) > 0) {
            it = win_next_next_next_turn_threats.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = k_3_threats.begin(); it != k_3_threats.end();) {
        const auto& t = *it;
        if (std::get<0>(t) == blocking_square || std::get<1>(t) == blocking_square ||
            std::get<2>(t) == blocking_square || std::get<3>(t) == blocking_square) {
            it = k_3_threats.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = win_next_next_next_next_turn_threats.begin(); it != win_next_next_next_next_turn_threats.end();) {
        const auto& w = *it;
        const Coord root = std::get<0>(w);
        const auto& refs = std::get<1>(w);
        const Coord ref0 = std::get<0>(refs);
        const Coord ref1 = std::get<1>(refs);
        const Coord ref2 = std::get<2>(refs);
        const Coord ref3 = std::get<3>(refs);
        if (blocking_square == root || blocking_square == ref0 || blocking_square == ref1 ||
            blocking_square == ref2 || blocking_square == ref3) {
            it = win_next_next_next_next_turn_threats.erase(it);
        } else {
            ++it;
        }
    }
}

// get_squares_from_wn3_threat --
std::set<MinimaxAgent::Coord> MinimaxAgent::get_squares_from_wn3_threat(const std::tuple<Coord, std::vector<std::pair<Coord, std::vector<Coord>>>>& threat) const {
    std::set<Coord> squares;
    flatten_threat_squares(threat, squares);
    return squares;
}

// get_squares_from_wn4_threat --
std::set<MinimaxAgent::Coord> MinimaxAgent::get_squares_from_wn4_threat(const std::pair<Coord, std::vector<std::pair<Coord, std::vector<Coord>>>>& threat) const {
    std::set<Coord> squares;
    flatten_threat_squares(threat, squares);
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
        depth_counter = {0, 0, 0, 0, 0};
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
    for (int i = 0; i <= max_depth; ++i) {
        if (depth_counter[i] == 0) break;
        std::cout << i << "\t| " << depth_counter[i] << "\n";
    }

    return best_move;
}


// minimax --
MinimaxAgent::SearchResult MinimaxAgent::minimax(const GameState& state, int depth_remaining, double alpha, double beta,
    ZobristHashing* z_hashing, int ff, int ff_branch_max, const TimePoint* timeout, bool* timed_out) const {

    depth_counter[depth_offset - depth_remaining + ff] += 1;

    constexpr auto wrapup = std::chrono::milliseconds(100);
    if ((timed_out && *timed_out) || (timeout && Clock::now() >= *timeout - wrapup)) {
        if (timed_out) *timed_out = true;
        return SearchResult{GameState::Move{-1, -1}, 0.0, 0, 0, ff, false};
    }
    const auto outcome = state.winner();
    if (outcome != GameState::Outcome::InProgress || depth_remaining <= 0 || state.is_full()) {
        const EvalResult eval = static_eval_result(state);
        SearchResult result{GameState::Move{-1, -1}, eval.value, eval.deadline, eval.result, ff};
        if (z_hashing != nullptr) {
            auto& z_memory = std::get<1>(*z_hashing);
            auto& z_key = std::get<2>(*z_hashing);
            z_memory[z_key] = std::make_tuple(result.move, result.value, result.deadline, result.result, result.fff);
        }
        return result;
    }

    const int h = state.rows();
    const int w = state.cols();
    const int k = state.k_in_row();
    const char a_piece = state.next_player();
    const char b_piece = (a_piece == GameState::X_PIECE) ? GameState::O_PIECE : GameState::X_PIECE;
    const double sign = (a_piece == GameState::X_PIECE) ? 1.0 : -1.0;

    GameState::Move best_move{-1, -1};
    double best_value = (a_piece == GameState::X_PIECE) ? -kInf : kInf;
    int best_deadline = 0;
    int best_result = (a_piece == GameState::X_PIECE) ? 3 : 2;
    int best_fff = ff;
    const int z_index = (a_piece == GameState::X_PIECE) ? 0 : 1;

    const std::vector<std::pair<int, int>> directions{{1, 0}, {0, 1}, {1, 1}, {-1, 1}};

    std::set<Coord> a_win_next_turn_threats;
    std::set<Coord> b_win_next_turn_threats;
    std::set<Threat1> a_k_1_threats;
    std::set<Threat1> b_k_1_threats;
    std::set<Threat2> a_k_2_threats;
    std::set<Threat2> b_k_2_threats;
    std::set<Threat3> a_k_3_threats;
    std::set<Threat3> b_k_3_threats;
    std::set<std::tuple<Coord, std::tuple<Coord, Coord>>> a_win_next_next_turn_threats;
    std::set<std::tuple<Coord, std::tuple<Coord, Coord>>> b_win_next_next_turn_threats;
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>> a_win_next_next_next_turn_threats;
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>> b_win_next_next_next_turn_threats;
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>> a_win_next_next_next_next_turn_threats;
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>> b_win_next_next_next_next_turn_threats;

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

    // Threat tuples may contain {-1, -1} placeholders. Never turn those
    // placeholders (or occupied/out-of-bounds coordinates) into search moves.
    const auto to_moves = [&state](const std::set<Coord>& coords) {
        std::vector<GameState::Move> out;
        out.reserve(coords.size());
        for (const auto& c : coords) {
            if (state.is_valid_move(c.first, c.second)) {
                out.emplace_back(c.first, c.second);
            }
        }
        return out;
    };

    std::set<Coord> searched_moves;
    while (true) {
        if (!a_win_next_turn_threats.empty()) {
            searched_moves.insert(a_win_next_turn_threats.begin(), a_win_next_turn_threats.end());
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(a_win_next_turn_threats), best_move, best_value, best_deadline, best_result, best_fff,
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
                moves.insert(std::get<0>(t));
            }
            searched_moves.insert(moves.begin(), moves.end());
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(moves), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            break;
        }

        if (!b_win_next_next_turn_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : b_win_next_next_turn_threats) {
                moves.insert(std::get<0>(t));
                const auto& children = std::get<1>(t);
                moves.insert(std::get<0>(children));
                moves.insert(std::get<1>(children));
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
                moves.insert(std::get<0>(t));
            }
            searched_moves.insert(moves.begin(), moves.end());
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(moves), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            break;
        }

        if (!b_win_next_next_next_turn_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : b_win_next_next_next_turn_threats) {
                const Coord overlap = std::get<0>(t);
                const auto& refs = std::get<1>(t);
                moves.insert(overlap);
                moves.insert(std::get<0>(refs));
                moves.insert(std::get<1>(refs));
                moves.insert(std::get<2>(refs));
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
                moves.insert(std::get<0>(t));
            }
            searched_moves.insert(moves.begin(), moves.end());
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(moves), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            break;
        }

        if (!b_win_next_next_next_next_turn_threats.empty()) {
            std::set<Coord> moves;
            for (const auto& t : b_win_next_next_next_next_turn_threats) {
                const Coord first = std::get<0>(t);
                const auto& second = std::get<1>(t);
                moves.insert(first);
                moves.insert(std::get<0>(second));
                moves.insert(std::get<1>(second));
                moves.insert(std::get<2>(second));
                moves.insert(std::get<3>(second));
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
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                static_cast<bool>(static_cast<int>(searched_moves.size()) < ff_branch_max), z_hashing, timeout, timed_out);
            break;
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
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            break;
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
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            break;
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
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            break;
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
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            break;
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
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(filtered), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            break;
        }

        std::set<Coord> adjacent_1_moves;
        std::set<Coord> adjacent_2_moves;
        std::vector<Coord> remaining_moves;
        sort_remaining_moves(state, h, w, searched_moves, adjacent_1_moves, adjacent_2_moves, remaining_moves);

        if (!adjacent_1_moves.empty()) {
            searched_moves.insert(adjacent_1_moves.begin(), adjacent_1_moves.end());
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(adjacent_1_moves), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            break;
        }

        if (!adjacent_2_moves.empty()) {
            searched_moves.insert(adjacent_2_moves.begin(), adjacent_2_moves.end());
            search_moves(state, depth_remaining, alpha, beta, z_index, ff, ff_branch_max, h, a_piece,
                to_moves(adjacent_2_moves), best_move, best_value, best_deadline, best_result, best_fff,
                false, z_hashing, timeout, timed_out);
            break;
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
        return SearchResult{GameState::Move{-1, -1}, 0.0, 0, 0, ff, false};
    }
    if (z_hashing != nullptr) {
        auto& z_memory = std::get<1>(*z_hashing);
        auto& z_key = std::get<2>(*z_hashing);

        z_memory[z_key] = std::make_tuple(
            best_move,
            best_value,
            best_deadline + 1,
            best_result,
            best_fff
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
        const bool better_value = value > best_value;
        const bool better_win_deadline =
            best_result == 3 && result == 3 &&
            deadline > best_deadline;
        const bool better_loss_deadline =
            best_result == 2 && result == 2 &&
            deadline < best_deadline;

        if (better_value || better_win_deadline || better_loss_deadline) {
            best_move = move;
            best_value = value;
            best_deadline = deadline;
            best_result = result;
            best_fff = fff;

            if (result == 2) {
                stop_search = true;
            }
        }

        if (beta != kInf && best_value > beta) {
            stop_search = true;
        }

        alpha = std::max(alpha, best_value);
    } else {
        const bool better_value = value < best_value;
        const bool better_win_deadline =
            best_result == 2 && result == 2 &&
            deadline > best_deadline;
        const bool better_loss_deadline =
            best_result == 3 && result == 3 &&
            deadline < best_deadline;

        if (better_value || better_win_deadline || better_loss_deadline) {
            best_move = move;
            best_value = value;
            best_deadline = deadline;
            best_result = result;
            best_fff = fff;

            if (result == 3) {
                stop_search = true;
            }
        }

        if (alpha != -kInf && best_value < alpha) {
            stop_search = true;
        }

        beta = std::min(beta, best_value);
    }

    return stop_search;
}

// search_moves --
void MinimaxAgent::search_moves(
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

    for (const auto& move : moves_to_search) {
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
            continue;
        }

        const GameState new_state = state.make_move(move);
        const int new_depth_remaining =
            can_fast_forward ? depth_remaining : depth_remaining - 1;

        SearchResult child_result{
            GameState::Move{-1, -1},
            0.0,
            0,
            0,
            ff,
            true
        };

        if (z_hashing != nullptr) {
            auto& z_table = std::get<0>(*z_hashing);
            auto& z_memory = std::get<1>(*z_hashing);
            auto& z_key = std::get<2>(*z_hashing);

            const std::uint64_t old_z_key = z_key;
            const std::uint64_t new_z_key = step_zobrist_key(
                old_z_key,
                z_table,
                move.first,
                move.second,
                z_index,
                state.cols()
            );

            const auto cached = z_memory.find(new_z_key);

            if (cached != z_memory.end()) {
                child_result = SearchResult{
                    std::get<0>(cached->second),
                    std::get<1>(cached->second),
                    std::get<2>(cached->second),
                    std::get<3>(cached->second),
                    std::get<4>(cached->second),
                    true
                };
            } else {
                z_key = new_z_key;

                try {
                    child_result = minimax(
                        new_state,
                        new_depth_remaining,
                        alpha,
                        beta,
                        z_hashing,
                        ff + (can_fast_forward ? 1 : 0),
                        can_fast_forward
                            ? std::max(0, ff_branch_max - 1)
                            : ff_branch_max,
                        timeout,
                        timed_out
                    );
                } catch (...) {
                    z_key = old_z_key;
                    throw;
                }

                z_key = old_z_key;
            }
        } else {
            // Preserve fast-forward state even when hashing is disabled.
            child_result = minimax(
                new_state,
                new_depth_remaining,
                alpha,
                beta,
                nullptr,
                ff + (can_fast_forward ? 1 : 0),
                can_fast_forward
                    ? std::max(0, ff_branch_max - 1)
                    : ff_branch_max,
                timeout,
                timed_out
            );
        }

        if (!child_result.completed) {
            if (timed_out) {
                *timed_out = true;
            }
            break;
        }

        const bool stop_search = eval_move(
            alpha,
            beta,
            a_piece,
            best_move,
            best_value,
            best_deadline,
            best_result,
            best_fff,
            move,
            child_result.value,
            child_result.deadline,
            child_result.result,
            child_result.fff
        );

        if (stop_search) {
            break;
        }
    }
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

    std::set<std::tuple<Coord, std::tuple<Coord, Coord>>> a_win_next_next_turn_threats;
    std::set<std::tuple<Coord, std::tuple<Coord, Coord>>> b_win_next_next_turn_threats;
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>> a_win_next_next_next_turn_threats;
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>> b_win_next_next_next_turn_threats;
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>> a_win_next_next_next_next_turn_threats;
    std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>> b_win_next_next_next_next_turn_threats;

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
    const std::set<std::tuple<MinimaxAgent::Coord, std::tuple<MinimaxAgent::Coord, MinimaxAgent::Coord>>>& a_win_next_next_turn_threats,
    const std::set<std::tuple<MinimaxAgent::Coord, std::tuple<MinimaxAgent::Coord, MinimaxAgent::Coord>>>& b_win_next_next_turn_threats,
    const std::set<std::tuple<MinimaxAgent::Coord, std::tuple<MinimaxAgent::Coord, MinimaxAgent::Coord, MinimaxAgent::Coord>>>& a_win_next_next_next_turn_threats,
    const std::set<std::tuple<MinimaxAgent::Coord, std::tuple<MinimaxAgent::Coord, MinimaxAgent::Coord, MinimaxAgent::Coord>>>& b_win_next_next_next_turn_threats,
    const std::set<std::tuple<MinimaxAgent::Coord, std::tuple<MinimaxAgent::Coord, MinimaxAgent::Coord, MinimaxAgent::Coord, MinimaxAgent::Coord>>>& a_win_next_next_next_next_turn_threats,
    const std::set<std::tuple<MinimaxAgent::Coord, std::tuple<MinimaxAgent::Coord, MinimaxAgent::Coord, MinimaxAgent::Coord, MinimaxAgent::Coord>>>& b_win_next_next_next_next_turn_threats,
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

    auto is_disjoint = [](const std::set<Coord>& lhs, const std::set<Coord>& rhs) {
        for (const auto& sq : lhs) {
            if (rhs.count(sq) > 0) {
                return false;
            }
        }
        return true;
    };

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
                    std::vector<Coord> last(4, {-1, -1});
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
        for (auto it1 = b_win_next_next.begin(); it1 != b_win_next_next.end() && !found_win; ++it1) {
            for (auto it2 = b_win_next_next.begin(); it2 != b_win_next_next.end(); ++it2) {
                if (it1 == it2) continue;

                const Coord root1 = std::get<0>(*it1);
                const Coord root2 = std::get<0>(*it2);
                const auto child1 = std::get<1>(*it1);
                const auto child2 = std::get<1>(*it2);
                const Coord c1a = std::get<0>(child1);
                const Coord c1b = std::get<1>(child1);
                const Coord c2a = std::get<0>(child2);
                const Coord c2b = std::get<1>(child2);

                std::set<Coord> s1{root1, c1a, c1b};
                std::set<Coord> s2{root2, c2a, c2b};
                bool overlap = false;
                for (const auto& sq : s1) {
                    if (s2.count(sq) > 0) {
                        overlap = true;
                        break;
                    }
                }

                if (root1 != root2 &&
                    root1 != c2a && root1 != c2b &&
                    root2 != c1a && root2 != c1b &&
                    !overlap) {
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
                    if (found_win) break;

                    const Coord root = std::get<0>(w1);
                    const auto refs = std::get<1>(w1);
                    const Coord first0 = std::get<0>(refs);
                    const Coord first1 = std::get<1>(refs);
                    const Coord first2 = std::get<2>(refs);
                    const bool match_root = (root == t1a || root == t1b);
                    const bool match_first0 = (first0 == t1a || first0 == t1b);
                    const bool match_first1 = (first1 == t1a || first1 == t1b);
                    const bool match_first2 = (first2 == t1a || first2 == t1b);

                    if (!(match_root || match_first0 || match_first1 || match_first2)) {
                        continue;
                    }

                    const Coord b_forced_square = (root == t1a) ? t1b : t1a;
                    std::set<Coord> next_a_squares{first0, first1, first2};
                    auto it = b_k_1_threat_squares.find(b_forced_square);
                    if (it == b_k_1_threat_squares.end() || next_a_squares.count(it->second) > 0) {
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

        for (const auto& w1 : b_win_next_next) {
            if (found_win) break;
            const Coord root1 = std::get<0>(w1);
            const auto child1 = std::get<1>(w1);
            const Coord c1a = std::get<0>(child1);
            const Coord c1b = std::get<1>(child1);
            const std::set<Coord> w1_children{c1a, c1b};

            for (const auto& w2 : b_win_next_next_next) {
                if (found_win) break;
                const Coord root2 = std::get<0>(w2);
                const auto refs2 = std::get<1>(w2);
                const Coord r20 = std::get<0>(refs2);
                const Coord r21 = std::get<1>(refs2);
                const Coord r22 = std::get<2>(refs2);
                const std::set<Coord> w2_squares{root2, r20, r21, r22};

                if (root1 != root2 &&
                    w2_squares.count(root1) == 0 &&
                    w1_children.count(root2) == 0 &&
                    is_disjoint(w1_children, w2_squares)) {
                    found_win = true;
                    for (const auto& t1 : a_k_2) {
                        const Coord t0 = std::get<0>(t1);
                        const Coord t1v = std::get<1>(t1);
                        const Coord t2v = std::get<2>(t1);
                        if (root1 == t0 || root1 == t1v || root1 == t2v ||
                            c1a == t0 || c1a == t1v || c1a == t2v ||
                            c1b == t0 || c1b == t1v || c1b == t2v) {
                            found_win = false;
                            break;
                        }
                    }
                }
            }
        }

        if (!found_win && b_win_next_next_next.size() >= 2) {
            std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>> completed;
            for (const auto& w1 : b_win_next_next_next) {
                completed.insert(w1);
                if (found_win) break;
                for (const auto& w2 : b_win_next_next_next) {
                    if (completed.count(w2) > 0) continue;
                    if (found_win) break;

                    auto w1_squares = std::set<Coord>{};
                    auto w2_squares = std::set<Coord>{};
                    const Coord w1root = std::get<0>(w1);
                    const auto w1refs = std::get<1>(w1);
                    w1_squares.insert(w1root);
                    w1_squares.insert(std::get<0>(w1refs));
                    w1_squares.insert(std::get<1>(w1refs));
                    w1_squares.insert(std::get<2>(w1refs));
                    const Coord w2root = std::get<0>(w2);
                    const auto w2refs = std::get<1>(w2);
                    w2_squares.insert(w2root);
                    w2_squares.insert(std::get<0>(w2refs));
                    w2_squares.insert(std::get<1>(w2refs));
                    w2_squares.insert(std::get<2>(w2refs));

                    const auto w1_refs_len = std::tuple_size<std::remove_reference_t<decltype(std::get<1>(w1))>>::value;
                    const auto w2_refs_len = std::tuple_size<std::remove_reference_t<decltype(std::get<1>(w2))>>::value;
                    if ((w1_refs_len == 2 || w2_refs_len == 2) && !a_k_2.empty()) {
                        break;
                    }
                    if (w1root != w2root &&
                        w1_squares.count(w2root) == 0 &&
                        w2_squares.count(w1root) == 0 &&
                        is_disjoint(w1_squares, w2_squares)) {
                        found_win = true;
                    }
                }
            }
        }

        if (found_win) {
            return EvalResult{b_win_value, 7 + fast_forward_counter, b_win_result};
        }
    }

    if (!a_win_next_next_next_next.empty()) {
        bool found_win = (b_k_1.empty());
        if (found_win) {
            return EvalResult{a_win_value, 8 + fast_forward_counter, a_win_result};
        }
    }

    if (value == 0.0 && !b_win_next_next_next_next.empty() && a_k_1.empty()) {
        bool found_win = false;

        for (const auto& w1 : b_win_next_next) {
            if (found_win) break;
            const Coord root1 = std::get<0>(w1);
            const auto child1 = std::get<1>(w1);
            const Coord c1a = std::get<0>(child1);
            const Coord c1b = std::get<1>(child1);
            const std::set<Coord> w1_children{c1a, c1b};

            for (const auto& w2 : b_win_next_next_next_next) {
                if (found_win) break;
                const Coord root2 = std::get<0>(w2);
                const auto refs2 = std::get<1>(w2);
                const Coord r20 = std::get<0>(refs2);
                const Coord r21 = std::get<1>(refs2);
                const Coord r22 = std::get<2>(refs2);
                const Coord r23 = std::get<3>(refs2);
                const std::set<Coord> w2_squares{root2, r20, r21, r22, r23};

                if (root1 != root2 &&
                    w2_squares.count(root1) == 0 &&
                    w1_children.count(root2) == 0 &&
                    is_disjoint(w1_children, w2_squares)) {
                    found_win = true;
                    for (const auto& t1 : a_k_2) {
                        const Coord t0 = std::get<0>(t1);
                        const Coord t1v = std::get<1>(t1);
                        const Coord t2v = std::get<2>(t1);
                        if (root1 == t0 || root1 == t1v || root1 == t2v ||
                            c1a == t0 || c1a == t1v || c1a == t2v ||
                            c1b == t0 || c1b == t1v || c1b == t2v) {
                            found_win = false;
                            break;
                        }
                    }
                }
            }
        }

        if (found_win) {
            return EvalResult{b_win_value, 9 + fast_forward_counter, b_win_result};
        }
    }

    return EvalResult{std::round(a_sign * a_value + b_sign * b_value), 0, 0};
}