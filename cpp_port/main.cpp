#include "GameState.h"
#include "MinimaxAgent.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class Agent {
public:
    virtual ~Agent() = default;
    virtual GameState::Move choose_move(const GameState& state) = 0;
    virtual std::string name() const = 0;
};

class HumanAgent : public Agent {
public:
    explicit HumanAgent(char piece) : piece_(piece) {}

    GameState::Move choose_move(const GameState& state) override {
        while (true) {
            std::cout << "Player " << piece_ << ", enter move as: row col\n> ";
            std::string line;
            if (!std::getline(std::cin, line)) {
                throw std::runtime_error("input closed");
            }

            std::istringstream iss(line);
            int row = 0;
            int col = 0;
            if (iss >> row >> col) {
                if (state.is_valid_move(row, col)) {
                    return {row, col};
                }
                std::cout << "Invalid move. The square must be empty and on the board.\n";
            } else {
                std::cout << "Please enter two integers separated by whitespace.\n";
            }
        }
    }

    std::string name() const override {
        return std::string("Human ") + piece_;
    }

private:
    char piece_;
};

class RandomBot : public Agent {
public:
    explicit RandomBot(char piece, unsigned seed = 0)
        : piece_(piece), rng_(seed == 0 ? static_cast<unsigned>(std::random_device{}()) : seed) {}

    GameState::Move choose_move(const GameState& state) override {
        std::vector<GameState::Move> moves;
        for (int r = 0; r < state.rows(); ++r) {
            for (int c = 0; c < state.cols(); ++c) {
                if (state.is_valid_move(r, c)) {
                    moves.emplace_back(r, c);
                }
            }
        }

        if (moves.empty()) {
            throw std::runtime_error("no legal moves available");
        }

        std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
        return moves[dist(rng_)];
    }

    std::string name() const override {
        return std::string("RandomBot ") + piece_;
    }

private:
    char piece_;
    std::mt19937 rng_;
};

class MinimaxBot : public Agent {
public:
    explicit MinimaxBot(char piece, int max_depth, unsigned seed = 0)
        : agent_(piece, max_depth, seed) {
            max_depth_ = max_depth;
        }

    GameState::Move choose_move(const GameState& state) override {
        return agent_.choose_move(state, max_depth_);
    }

    std::string name() const override {
        return std::string("MinimaxBot ") + agent_.name();
    }

private:
    MinimaxAgent agent_;
    int max_depth_;
};

// std::unique_ptr<Agent> make_agent(int type, char piece) {
//     switch (type) {
//         case 0:
//             return std::make_unique<MinimaxBot>(piece);
//         case 1:
//             return std::make_unique<HumanAgent>(piece);
//         default:
//             return std::make_unique<HumanAgent>(piece);
//     }
// }

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [rows cols k players]\n"
              << "  rows = board height (default: 9)\n"
              << "  cols = board width (default: 9)\n"
              << "  k    = pieces in a row to win (default: 5)\n"
              << "  players = 0 bot-vs-bot, 1 human-vs-bot, 2 human-vs-human (default: 0)\n"
              << "Examples:\n"
              << "  " << program_name << "\n"
              << "  " << program_name << " 3 3 3 1\n"
              << "  " << program_name << " 7 9 5 1\n";
}

int main(int argc, char* argv[]) {
    int rows = 9;
    int cols = 9;
    int k = 5;
    int max_depth = 4;
    int players = 0;

    if (argc >= 2 && std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    if (argc >= 2) rows = std::stoi(argv[1]);
    if (argc >= 3) cols = std::stoi(argv[2]);
    if (argc >= 4) k = std::stoi(argv[3]);
    if (argc >= 5) max_depth = std::stoi(argv[4]);
    if (argc >= 6) players = std::stoi(argv[5]);

    if (rows <= 0 || cols <= 0 || k <= 0) {
        std::cerr << "Rows, columns, and k must all be positive.\n";
        return 1;
    }

    if (k > std::max(rows, cols)) {
        std::cerr << "k cannot be larger than the board dimensions.\n";
        return 1;
    }

    if (players < 0 || players > 2) {
        std::cerr << "players must be 0, 1, or 2.\n";
        return 1;
    }

    GameState state = GameState::empty(rows, cols, k, GameState::X_PIECE);

    std::unique_ptr<Agent> x_agent;
    std::unique_ptr<Agent> o_agent;

    if (players == 0) {
        x_agent = std::make_unique<MinimaxBot>(GameState::X_PIECE, max_depth);
        o_agent = std::make_unique<MinimaxBot>(GameState::O_PIECE, max_depth);
    } else if (players == 1) {
        x_agent = std::make_unique<HumanAgent>(GameState::X_PIECE);
        o_agent = std::make_unique<MinimaxBot>(GameState::O_PIECE, max_depth);
    } else {
        x_agent = std::make_unique<HumanAgent>(GameState::X_PIECE);
        o_agent = std::make_unique<HumanAgent>(GameState::O_PIECE);
    }

    while (true) {

        const char current_player = state.next_player();
        Agent* current_agent = (current_player == GameState::X_PIECE) ? x_agent.get() : o_agent.get();

        try {
            const auto move = current_agent->choose_move(state);
            const bool valid = state.is_valid_move(move);
            if (!valid) {
                std::cerr << current_agent->name()
                        << " selected an invalid move.\n";
                return 1;
            }
            state = state.make_move(move);
        } catch (const std::exception& ex) {
            std::cerr << "Error: " << ex.what() << "\n";
            return 1;
        }

        const auto outcome = state.winner();
        if (outcome == GameState::Outcome::InProgress) {
            continue;
        }

        std::cout << "_____________________________________________\n\n";
        std::cout << state << "\n";
        if (outcome == GameState::Outcome::Draw) {
            std::cout << "The game ends in a draw.\n";
        } else if (outcome == GameState::Outcome::XWins) {
            std::cout << "Player X wins!\n";
        } else {
            std::cout << "Player O wins!\n";
        }
        break;
    }

    return 0;
}
