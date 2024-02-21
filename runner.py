"""
runner.py
author: CSE 415 course staff

Contains a class for running games between two agents. This is the main entry point for running your own games.
Just run python3 runner.py or use whatever method you like to run this file.

You probably should change the section at the bottom of this file for your own testing. Any changes you make will not be
reflected in our grading, so you probably shouldn't change the GameRunner class.
"""

import game
import agent
import transcript
import random
import sys


class GameRunner:
    agents: dict[str, agent.Agent]

    def __init__(self, x_agent: agent.Agent, o_agent: agent.Agent):
        self.agents = {game.X_PIECE: x_agent, game.O_PIECE: o_agent}

    def run_game(self, initial_state: game.GameState, time_limit=None, silent=False, transcript_name=None):
        """
        Runs a game between the two agents using the given starting state.
        :param initial_state: starting state
        :param time_limit: time (in seconds) given to each player for their move
        :param silent: True to suppress most console output
        :param transcript_name: name of file (without extension) to save game transcript to. None will not save anything
        :return: winner of the game ('X' or 'O')
        """
        state = initial_state.copy()
        if silent:
            def p(text=''):
                pass
        else:
            def p(text=''):
                print(text)

        t = transcript.Transcript()

        p("Players, introduce yourselves!\n"
          "==============================")
        t.runner_comment("Players, introduce yourselves!")
        for piece in self.agents.keys():
            p()
            p(f"Playing as {piece}:")
            p(self.agents[piece].introduce())
            p()
            t.runner_comment(f"Playing as {piece}:")
            t.player_comment(self.agents[piece].introduce(), piece)
        p(' vs '.join(a.nickname() for a in self.agents.values()))
        t.runner_comment(' vs '.join(a.nickname() for a in self.agents.values()))
        p("Let the game begin!")
        t.runner_comment("Here is the starting board:")
        t.print_move(None, None, None, state)
        t.runner_comment("Let the game begin!")

        while not (winner := state.winner()):
            curr_agent = self.agents[state.next_player]
            piece = state.next_player
            try:
                move = curr_agent.get_move(state, time_limit)
                if not state.is_valid_move(move):
                    raise ValueError
                state = state.make_move(move)
                t.print_move(curr_agent.nickname(), piece, move, state)
                p(state)
                p()
            except TimeoutError:
                print(f"player {curr_agent.nickname()} failed to return a move within the time limit")
                t.runner_comment(f"player {curr_agent.nickname()} failed to return a move within the time limit")
                winner = game.X_PIECE if piece == game.O_PIECE else game.O_PIECE
                break
            except ValueError:
                print(f"player {curr_agent.nickname()} did not return a valid move")
                t.runner_comment(f"player {curr_agent.nickname()} did not return a valid move")
                winner = game.X_PIECE if piece == game.O_PIECE else game.O_PIECE
                break
            except:
                print(f"exception during {curr_agent.nickname()}'s play")
                t.runner_comment(f"exception during {curr_agent.nickname()}'s play")
                winner = game.X_PIECE if piece == game.O_PIECE else game.O_PIECE
                break
        if winner == "draw":
            print("Game ends in a draw!")
            t.runner_comment("Game ends in a draw!")
        else:
            print(f"Player {winner}, aka {self.agents[winner].nickname()} wins the game!")
            t.runner_comment(f"Player {winner}, aka {self.agents[winner].nickname()} wins the game!")

        if transcript_name:
            t.generate(transcript_name, pdf=True)

        return winner


if __name__ == '__main__':
    """
    This is what to change to run your own games. Some examples of starting game boards are below, but you can also add
    your own by looking at the examples in game.py. Load in whatever agents you're playing, or even use the same agent
    twice. Then construct the GameRunner and call run_game with your starting state.
    
    Note: to generate pdf transcripts, you must run `pip install pyppeteer`. Otherwise an html transcript will be made.
    Remember to change the transcript name, otherwise your old transcript will be overwritten!
    """
    import minimax_agent

    if len(sys.argv) > 1:
        rows = int(sys.argv[1])
        cols = int(sys.argv[2])
        k = int(sys.argv[3])
        players = int(sys.argv[4])   # 0 or 1
        time_limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
        auto_moves = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    else:
        rows = 7
        cols = 7
        k = 5
        players = 0
        time_limit = 1.0
        auto_moves = 0

    """Various starting board configurations"""
    s = game.GameState.empty((rows, cols), k)
    # s = game.GameState.no_corners()
    # s = game.GameState.no_corners_small()
    # s = game.GameState.tic_tac_toe()

    """Initialise agents and game runner"""
    if players == 1:
        a1 = agent.Agent(s, game.X_PIECE)
    else:
        a1 = minimax_agent.MinimaxAgent(s, game.X_PIECE)
    a2 = minimax_agent.MinimaxAgent(s, game.O_PIECE)
    r = GameRunner(x_agent=a1, o_agent=a2)

    """
    Pre-moves a certain number of times for a unique starting board configuration.
    Comment out this code for a blank board.
    """
    for i in range(auto_moves):
        while not s.is_valid_move(move := (random.randint(0, s.w - 1), random.randint(0, s.h - 1))):
            pass
        s = s.make_move(move)

    # print(s)
    # print(a1.static_eval(s))

    r.run_game(s, time_limit=time_limit, silent=True, transcript_name="out")
