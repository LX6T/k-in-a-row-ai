#include "MinimaxAgent.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

GameState make_state_from_rows(const std::vector<std::string>& rows, char next, int k) {
    const int rows_count = static_cast<int>(rows.size());
    const int cols_count = static_cast<int>(rows[0].size());
    std::vector<std::vector<char>> board(rows_count, std::vector<char>(cols_count, GameState::EMPTY_PIECE));

    for (int r = 0; r < rows_count; ++r) {
        for (int c = 0; c < cols_count; ++c) {
            const char ch = rows[r][c];
            if (ch == 'X') {
                board[r][c] = GameState::X_PIECE;
            } else if (ch == 'O') {
                board[r][c] = GameState::O_PIECE;
            } else if (ch == '.') {
                board[r][c] = GameState::EMPTY_PIECE;
            } else {
                throw std::runtime_error("unsupported board char");
            }
        }
    }

    return GameState(rows_count, cols_count, k, next, board);
}

}  // namespace

int main() {
    try {
        {
            GameState empty_state = GameState::empty(3, 3, 3, GameState::X_PIECE);
            MinimaxAgent agent(GameState::X_PIECE, 2);
            auto move = agent.choose_move(empty_state, 2);
            std::cout << "empty-state move: (" << move.first << ", " << move.second << ")\n";
            expect(empty_state.is_valid_move(move), "empty board must select a legal move");
        }

        {
            auto win_state = make_state_from_rows({"XX.", "O.O", "O.."}, GameState::X_PIECE, 3);
            MinimaxAgent agent(GameState::X_PIECE, 2);
            auto move = agent.choose_move(win_state, 2);
            std::cout << "win-state move: (" << move.first << ", " << move.second << ")\n";
            expect(move == GameState::Move(0, 2), "immediate win should be selected");
        }

        {
            auto block_state = make_state_from_rows({"OO.", "X..", "..."}, GameState::X_PIECE, 3);
            MinimaxAgent agent(GameState::X_PIECE, 2);
            auto move = agent.choose_move(block_state, 2);
            std::cout << "block-state move: (" << move.first << ", " << move.second << ")\n";
            expect(move == GameState::Move(0, 2), "agent should block an immediate opponent win");
        }

        {
            GameState state = GameState::empty(5, 5, 5, GameState::X_PIECE);
            MinimaxAgent agent(GameState::X_PIECE, 2);
            auto move = agent.choose_move(state, 2);
            std::cout << "mid-board move: (" << move.first << ", " << move.second << ")\n";
            expect(state.is_valid_move(move), "mid board move must be legal");
        }

        {
            const auto strategic_state = make_state_from_rows({"O..", "...", ".X."}, GameState::X_PIECE, 3);
            MinimaxAgent agent(GameState::X_PIECE, 3);
            const auto move = agent.choose_move(strategic_state, 3);
            std::cout << "strategic move: (" << move.first << ", " << move.second << ")\n";
            expect(move == GameState::Move(2, 0), "choose_move should match the Python source-of-truth choice on the strategic test board");
        }

        {
            MinimaxAgent agent(GameState::X_PIECE, 2);
            const auto win_state = make_state_from_rows({"XX.", "O.O", "O.."}, GameState::X_PIECE, 3);
            const double win_eval = agent.static_eval(win_state);
            expect(std::abs(win_eval - 1.0e24) < 1.0e18, "static_eval should reward an immediate X win");
        }

        {
            MinimaxAgent agent(GameState::X_PIECE, 2);
            const auto block_state = make_state_from_rows({"OO.", "X..", "..."}, GameState::X_PIECE, 3);
            const double block_eval = agent.static_eval(block_state);
            expect(std::abs(block_eval + 1.0e24) < 1.0e18, "static_eval should heavily penalize an unblocked O win");
        }

        {
            MinimaxAgent agent(GameState::X_PIECE, 2);
            const auto empty_state = GameState::empty(3, 3, 3, GameState::X_PIECE);
            const double eval = agent.static_eval(empty_state);
            expect(std::abs(eval) < 1e-9, "empty board should evaluate neutrally");
        }

        {
            auto hashing = MinimaxAgent::make_zobrist_hashing(3, 3, 1234);
            const auto& table = std::get<0>(hashing);
            expect(table.size() == 9, "zobrist table should have one entry per square");
            expect(table[0].size() == 2, "zobrist table entry should contain X and O hash values");
            const uint64_t key = MinimaxAgent::step_zobrist_key(0, table, 1, 2, 0);
            const uint64_t expected = static_cast<uint64_t>(table[1 * 3 + 2][0]);
            expect(key == expected, "zobrist key update should xor the square's player hash");
        }

        std::cout << "All C++ minimax unit tests passed." << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "test failure: " << ex.what() << std::endl;
        return 1;
    }
}
