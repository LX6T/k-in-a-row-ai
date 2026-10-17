#ifndef GAME_STATE_H
#define GAME_STATE_H

#include <cstddef>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

class GameState {
public:
    enum class Outcome {
        InProgress,
        XWins,
        OWins,
        Draw
    };

    using Move = std::pair<int, int>;

    static constexpr char X_PIECE = 'X';
    static constexpr char O_PIECE = 'O';
    static constexpr char EMPTY_PIECE = ' ';

    GameState() = default;
    GameState(int rows, int cols, int k, char next_player, std::vector<std::vector<char>> board = {});

    static GameState empty(int rows, int cols, int k, char first = X_PIECE);

    bool is_valid_move(int row, int col) const;
    bool is_valid_move(const Move& move) const;
    bool is_full() const;
    GameState make_move(const Move& move) const;

    Outcome winner() const;

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int k_in_row() const { return k_; }
    char next_player() const { return next_player_; }
    const std::vector<std::vector<char>>& board() const { return board_; }

    void print(std::ostream& os) const;
    std::string to_string() const;

private:
    int rows_ = 0;
    int cols_ = 0;
    int k_ = 0;
    char next_player_ = X_PIECE;
    std::vector<std::vector<char>> board_;
};

std::ostream& operator<<(std::ostream& os, const GameState& state);

#endif
