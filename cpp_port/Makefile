CXX := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -pedantic
TARGET := k_in_row
MINIMAX_TARGET := minimax_driver

all: $(TARGET) $(MINIMAX_TARGET)

$(TARGET): main.cpp GameState.cpp GameState.h MinimaxAgent.cpp MinimaxAgent.h
	$(CXX) $(CXXFLAGS) main.cpp MinimaxAgent.cpp GameState.cpp -o $(TARGET)

$(MINIMAX_TARGET): MinimaxDriver.cpp MinimaxAgent.cpp GameState.cpp GameState.h MinimaxAgent.h
	$(CXX) $(CXXFLAGS) MinimaxDriver.cpp MinimaxAgent.cpp GameState.cpp -o $(MINIMAX_TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(MINIMAX_TARGET)

.PHONY: all run clean
