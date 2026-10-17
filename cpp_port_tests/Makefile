CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic
TEST_BIN := test_minimax
CLI_BIN := choose_move_cli

all: $(TEST_BIN)

$(TEST_BIN): test_minimax.cpp ../cpp_port/MinimaxAgent.cpp ../cpp_port/GameState.cpp
	$(CXX) $(CXXFLAGS) test_minimax.cpp ../cpp_port/MinimaxAgent.cpp ../cpp_port/GameState.cpp -I../cpp_port -o $(TEST_BIN)

$(CLI_BIN): choose_move_cli.cpp ../cpp_port/MinimaxAgent.cpp ../cpp_port/GameState.cpp
	$(CXX) $(CXXFLAGS) choose_move_cli.cpp ../cpp_port/MinimaxAgent.cpp ../cpp_port/GameState.cpp -I../cpp_port -o $(CLI_BIN)

run: $(TEST_BIN)
	./$(TEST_BIN)

randomized_parity: $(CLI_BIN)
	PYTHONPATH=.. python3 randomized_parity_12x12.py

clean:
	rm -f $(TEST_BIN) $(CLI_BIN)

.PHONY: all run randomized_parity clean
