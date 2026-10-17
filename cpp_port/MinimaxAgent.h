#ifndef MINIMAX_AGENT_H
#define MINIMAX_AGENT_H

#include "GameState.h"
#include <cstdint>
#include <chrono>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <array>

class MinimaxAgent {
public:
    using ZobristTable = std::vector<std::vector<std::uint32_t>>;
    using ZobristMemory = std::unordered_map<std::uint64_t, std::tuple<GameState::Move, double, int, int, int>>;
    using ZobristHashing = std::tuple<ZobristTable, ZobristMemory, std::uint64_t>;

    static ZobristHashing make_zobrist_hashing(int rows, int cols, unsigned seed = 0);
    static std::uint64_t step_zobrist_key(std::uint64_t current_key, const ZobristTable& z_table, int row, int col, int z_index, int board_cols = -1);
    MinimaxAgent(char piece, int max_depth = 3, unsigned seed = 0, int init_ff_branch_max = 0);
    GameState::Move choose_move(const GameState& state, int depth = -1) const;
    // Python-agent-compatible overload: search iteratively until this many seconds remain elapsed.
    GameState::Move choose_move(const GameState& state, double time_limit_seconds) const;
    double static_eval(const GameState& state) const;
    std::string name() const;

private:
    struct EvalResult { double value; int deadline; int result; };
    struct SearchResult { GameState::Move move; double value; int deadline; int result; int fff; bool completed = true; };
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    char piece_;
    int max_depth_;
    unsigned seed_;
    int init_ff_branch_max_;

    mutable int static_eval_calls = 0;
    mutable std::array<int, 5> depth_counter = {};
    mutable int depth_offset = 0;

    using Coord = std::pair<int, int>;
    using Threat1 = std::tuple<Coord, Coord>;
    using Threat2 = std::tuple<Coord, Coord, Coord>;
    using Threat3 = std::tuple<Coord, Coord, Coord, Coord>;

    std::vector<GameState::Move> legal_moves(const GameState& state) const;
    static char opponent(char piece);
    static bool is_winning_move(const GameState& state, const GameState::Move& move, char player);
    static int score_window(const std::vector<char>& window, int k, char player);

    void update_wins_and_threats(int k, const std::vector<Coord>& last, int count, double& value, int& wins,
        std::set<Coord>& win_next_turn_threats, std::set<Threat1>& k_1_threats,
        std::set<Threat2>& k_2_threats, std::set<Threat3>& k_3_threats) const;
    void process_wins_and_threats(const GameState& state, int h, int w, int k,
        const std::vector<std::pair<int, int>>& directions, char a_piece, char b_piece,
        std::set<Coord>& a_win_next_turn_threats, std::set<Coord>& b_win_next_turn_threats,
        std::set<Threat1>& a_k_1_threats, std::set<Threat1>& b_k_1_threats,
        std::set<Threat2>& a_k_2_threats, std::set<Threat2>& b_k_2_threats,
        std::set<Threat3>& a_k_3_threats, std::set<Threat3>& b_k_3_threats,
        double& a_value, double& b_value, int& a_wins, int& b_wins) const;
    void detect_threats(const std::set<Threat1>& k_1_threats, const std::set<Threat2>& k_2_threats,
        const std::set<Threat3>& k_3_threats,
        std::set<std::tuple<Coord, std::tuple<Coord, Coord>>>& win_next_next_turn_threats,
        std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>>& win_next_next_next_turn_threats,
        std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>>& win_next_next_next_next_turn_threats) const;
    void update_threats(const Coord& new_square, std::set<Threat1>& k_1_threats, std::set<Threat2>& k_2_threats,
        std::set<Coord>& win_next_turn_threats,
        std::set<std::tuple<Coord, std::tuple<Coord, Coord>>>& win_next_next_turn_threats,
        std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>>& win_next_next_next_turn_threats,
        std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>>& win_next_next_next_next_turn_threats) const;
    void block_threats(const Coord& blocking_square, std::set<Threat1>& k_1_threats, std::set<Threat2>& k_2_threats,
        std::set<Threat3>& k_3_threats,
        std::set<std::tuple<Coord, std::tuple<Coord, Coord>>>& win_next_next_turn_threats,
        std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>>& win_next_next_next_turn_threats,
        std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>>& win_next_next_next_next_turn_threats) const;
    std::set<Coord> get_squares_from_wn3_threat(const std::tuple<Coord, std::vector<std::pair<Coord, std::vector<Coord>>>>&) const;
    std::set<Coord> get_squares_from_wn4_threat(const std::pair<Coord, std::vector<std::pair<Coord, std::vector<Coord>>>>&) const;

    GameState::Move choose_move_impl(const GameState& state, int depth, const TimePoint* timeout) const;
    SearchResult minimax(const GameState& state, int depth_remaining, double alpha, double beta,
        ZobristHashing* z_hashing = nullptr, int ff = 0, int ff_branch_max = 0,
        const TimePoint* timeout = nullptr, bool* timed_out = nullptr) const;
    void sort_remaining_moves(const GameState& state, int h, int w, const std::set<Coord>& searched_moves,
        std::set<Coord>& adjacent_1_moves, std::set<Coord>& adjacent_2_moves, std::vector<Coord>& remaining_moves) const;
    bool eval_move(double& alpha, double& beta, char a_piece, GameState::Move& best_move, double& best_value,
        int& best_deadline, int& best_result, int& best_fff,
        const GameState::Move& move, double value, int deadline, int result, int fff) const;
    void search_moves(const GameState& state, int depth_remaining, double& alpha, double& beta,
        int z_index, int ff, int ff_branch_max, int h, char a_piece,
        const std::vector<GameState::Move>& moves_to_search, GameState::Move& best_move,
        double& best_value, int& best_deadline, int& best_result, int& best_fff,
        bool can_fast_forward, ZobristHashing* z_hashing = nullptr,
        const TimePoint* timeout = nullptr, bool* timed_out = nullptr) const;
    EvalResult static_eval_result(const GameState& state) const;
    EvalResult calculate_value(const GameState& state, int h, int w, int k,
        const std::vector<std::pair<int,int>>& directions, char a_piece, char b_piece,
        double a_sign, double b_sign, double a_value, double b_value, int a_wins, int b_wins,
        const std::set<Coord>& a_win_next_turn_threats, const std::set<Coord>& b_win_next_turn_threats,
        const std::set<std::tuple<Coord, std::tuple<Coord, Coord>>>& a_win_next_next_turn_threats,
        const std::set<std::tuple<Coord, std::tuple<Coord, Coord>>>& b_win_next_next_turn_threats,
        const std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>>& a_win_next_next_next_turn_threats,
        const std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord>>>& b_win_next_next_next_turn_threats,
        const std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>>& a_win_next_next_next_next_turn_threats,
        const std::set<std::tuple<Coord, std::tuple<Coord, Coord, Coord, Coord>>>& b_win_next_next_next_next_turn_threats,
        const std::set<Threat1>& a_k_1_threats, const std::set<Threat1>& b_k_1_threats,
        const std::set<Threat2>& a_k_2_threats, const std::set<Threat2>& b_k_2_threats,
        const std::set<Threat3>& a_k_3_threats, const std::set<Threat3>& b_k_3_threats,
        double a_win_value, double b_win_value, int a_win_result, int b_win_result) const;
};
#endif
