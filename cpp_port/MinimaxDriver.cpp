#include "GameState.h"
#include "MinimaxAgent.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start < s.size()) {
        const auto end = s.find(delim, start);
        const auto token = (end == std::string::npos) ? s.substr(start) : s.substr(start, end - start);
        if (!token.empty()) {
            parts.push_back(token);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return parts;
}

GameState parse_board(const std::string& board, int rows, int cols, int k, char next_player) {
    std::vector<std::vector<char>> parsed(rows, std::vector<char>(cols, GameState::EMPTY_PIECE));
    const auto rows_tokens = split(board, ';');
    if (rows_tokens.size() != static_cast<std::size_t>(rows)) {
        throw std::runtime_error("board string must contain one row per board row");
    }

    for (int r = 0; r < rows; ++r) {
        const auto& row_string = rows_tokens[r];
        for (int c = 0; c < cols; ++c) {
            if (c >= static_cast<int>(row_string.size())) {
                parsed[r][c] = GameState::EMPTY_PIECE;
                continue;
            }
            const char ch = row_string[c];
            if (ch == 'X' || ch == 'O' || ch == ' ' || ch == '.') {
                parsed[r][c] = (ch == 'X' || ch == 'x') ? GameState::X_PIECE : ((ch == 'O' || ch == 'o') ? GameState::O_PIECE : GameState::EMPTY_PIECE);
            } else {
                throw std::runtime_error(std::string("unsupported board token: ") + ch);
            }
        }
    }

    return GameState(rows, cols, k, next_player, parsed);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int rows = 3;
        int cols = 3;
        int k = 3;
        char next_player = GameState::X_PIECE;
        std::string board = "...;...;...";

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--rows" && i + 1 < argc) rows = std::stoi(argv[++i]);
            else if (arg == "--cols" && i + 1 < argc) cols = std::stoi(argv[++i]);
            else if (arg == "--k" && i + 1 < argc) k = std::stoi(argv[++i]);
            else if (arg == "--next" && i + 1 < argc) next_player = argv[++i][0];
            else if (arg == "--board" && i + 1 < argc) board = argv[++i];
            else if (arg == "--help") {
                std::cout << "Usage: minimax_driver --rows 3 --cols 3 --k 3 --next X --board \"...;...;...\"\n";
                return 0;
            }
        }

        const GameState state = parse_board(board, rows, cols, k, next_player);
        MinimaxAgent agent(next_player, 2);
        const auto move = agent.choose_move(state, 2);
        std::cout << move.first << "," << move.second << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
