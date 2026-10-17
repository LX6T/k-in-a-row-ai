#include "GameState.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace {

bool in_bounds(int row, int col, int rows, int cols) {
    return row >= 0 && row < rows && col >= 0 && col < cols;
}

}

GameState::GameState(int rows, int cols, int k, char next_player, std::vector<std::vector<char>> board)
    : rows_(rows), cols_(cols), k_(k), next_player_(next_player), board_(std::move(board)) {
    if (rows_ <= 0 || cols_ <= 0) {
        throw std::invalid_argument("board dimensions must be positive");
    }
    if (k_ <= 0 || k_ > std::max(rows_, cols_)) {
        throw std::invalid_argument("k must be greater than zero and not exceed board size");
    }
    if (board_.empty()) {
        board_.assign(rows_, std::vector<char>(cols_, EMPTY_PIECE));
    } else {
        if (static_cast<int>(board_.size()) != rows_ || !std::all_of(board_.begin(), board_.end(),
                [&](const auto& row) { return static_cast<int>(row.size()) == cols_; })) {
            throw std::invalid_argument("board dimensions do not match rows and columns");
        }
    }
}

GameState GameState::empty(int rows, int cols, int k, char first) {
    return GameState(rows, cols, k, first, {});
}

bool GameState::is_valid_move(int row, int col) const {
    return in_bounds(row, col, rows_, cols_) && board_[row][col] == EMPTY_PIECE;
}

bool GameState::is_valid_move(const Move& move) const {
    return is_valid_move(move.first, move.second);
}

bool GameState::is_full() const {
    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            if (board_[r][c] == EMPTY_PIECE) {
                return false;
            }
        }
    }
    return true;
}

GameState GameState::make_move(const Move& move) const {
    if (!is_valid_move(move)) {
        throw std::invalid_argument("move is not valid for the current board");
    }

    auto new_board = board_;
    new_board[move.first][move.second] = next_player_;

    char next_player = (next_player_ == X_PIECE) ? O_PIECE : X_PIECE;
    return GameState(rows_, cols_, k_, next_player, new_board);
}

GameState::Outcome GameState::winner() const {
    const std::vector<std::pair<int, int>> directions = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            char piece = board_[r][c];
            if (piece == EMPTY_PIECE) {
                continue;
            }

            for (const auto& [dr, dc] : directions) {
                int count = 1;
                int rr = r + dr;
                int cc = c + dc;

                while (in_bounds(rr, cc, rows_, cols_) && board_[rr][cc] == piece) {
                    ++count;
                    rr += dr;
                    cc += dc;
                }

                if (count >= k_) {
                    return piece == X_PIECE ? Outcome::XWins : Outcome::OWins;
                }
            }
        }
    }

    if (is_full()) {
        return Outcome::Draw;
    }

    return Outcome::InProgress;
}

void GameState::print(std::ostream& os) const {
    os << "   ";
    for (int c = 0; c < cols_; ++c) {
        os << " " << c << "  ";
    }
    
    os << "\n  +";
    for (int c = 0; c < cols_; ++c) {
        os << "---";
        if (c != cols_ - 1) {
            os << "+";
        }
    }
    os << "+\n";

    for (int r = 0; r < rows_; ++r) {
        os << std::to_string(r) << " |";
        for (int c = 0; c < cols_; ++c) {
            os << " " << board_[r][c] << " |";
        }
        os << "\n";
    }

    os << "  +";
    for (int c = 0; c < cols_; ++c) {
        os << "---";
        if (c != cols_ - 1) {
            os << "+";
        }
    }
    os << "+\n";
}

std::string GameState::to_string() const {
    std::ostringstream oss;
    print(oss);
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const GameState& state) {
    state.print(os);
    return os;
}
