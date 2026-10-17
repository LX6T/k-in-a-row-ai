#include "MinimaxAgent.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string trim(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\n' || s[start] == '\r' || s[start] == '\t')) {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start && (s[end - 1] == ' ' || s[end - 1] == '\n' || s[end - 1] == '\r' || s[end - 1] == '\t')) {
        --end;
    }
    return s.substr(start, end - start);
}

char parse_player(const std::string& value) {
    const std::string v = trim(value);
    if (v == "X" || v == "x") return GameState::X_PIECE;
    if (v == "O" || v == "o") return GameState::O_PIECE;
    throw std::runtime_error("next_player must be X or O");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int rows = 12;
        int cols = 12;
        char next_player = GameState::X_PIECE;
        std::string mode = "choose_move";

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--rows" && i + 1 < argc) {
                rows = std::stoi(argv[++i]);
            } else if (arg == "--cols" && i + 1 < argc) {
                cols = std::stoi(argv[++i]);
            } else if (arg == "--next" && i + 1 < argc) {
                next_player = parse_player(argv[++i]);
            } else if (arg == "--mode" && i + 1 < argc) {
                mode = argv[++i];
            } else if (arg == "--help") {
                std::cout << "Usage: choose_move_cli --rows 12 --cols 12 --next X [--mode choose_move|static_eval]\n";
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        std::vector<std::vector<char>> board(rows, std::vector<char>(cols, GameState::EMPTY_PIECE));
        std::string line;
        int row_index = 0;
        while (std::getline(std::cin, line) && row_index < rows) {
            line = trim(line);
            if (line.empty()) {
                continue;
            }
            if (line.size() != static_cast<std::size_t>(cols)) {
                throw std::runtime_error("board row length mismatch: expected " + std::to_string(cols) + ", got " + std::to_string(line.size()));
            }
            for (int col = 0; col < cols; ++col) {
                const char ch = line[col];
                if (ch == '.') {
                    board[row_index][col] = GameState::EMPTY_PIECE;
                } else if (ch == 'X' || ch == 'O') {
                    board[row_index][col] = ch;
                } else {
                    throw std::runtime_error("invalid board char: " + std::string(1, ch));
                }
            }
            ++row_index;
        }

        if (row_index != rows) {
            throw std::runtime_error("expected " + std::to_string(rows) + " rows but only read " + std::to_string(row_index));
        }

        GameState state(rows, cols, 5, next_player, board);
        MinimaxAgent agent(next_player, 3);

        if (mode == "static_eval") {
            std::cout << agent.static_eval(state) << std::endl;
            return 0;
        }
        if (mode == "choose_move") {
            const auto move = agent.choose_move(state, 2);
            std::cout << move.first << "," << move.second << std::endl;
            return 0;
        }
        throw std::runtime_error("unsupported mode: " + mode);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }
}
