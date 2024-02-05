"""
transcript.py
author: CSE 415 course staff

This file generates a file transcript of your game. You do not need to read or understand this file.
You should not modify this file.
"""

import game


CLASSES = {'X': 'x-text', 'O': 'o-text', 'runner': 'runner-text'}
TOKENS = {'X': '<span class="x-token">X</span>',
          'O': '<span class="o-token">O</span>',
          '-': '<span class="brick-token">-</span>'}


class Transcript:
    _data: str

    def __init__(self):
        self._data = '<html><head><title>K-in-a-Row game</title>'
        self._data += '<style>'
        self._data += ('.runner-text:after { content: \'\'; position: absolute; bottom: 0; left: 50%; width: 0; '
                       'height: 0; border: 10px solid transparent; border-top-color: gray; border-bottom: 0; '
                       'border-left: 0; margin-left: -10px; margin-bottom: -10px; }')
        self._data += ('.runner-text { position: relative; border: 2px solid gray; border-radius: 0.4em; '
                       'width: fit-content; margin: 0 auto 12 auto; padding: 5; }')
        self._data += ('.x-text:after { content: \'\'; position: absolute; bottom: 0; left: 25%; width: 0; height: 0; '
                       'border: 10px solid transparent; border-top-color: blue; border-bottom: 0; border-right: 0; '
                       'margin-left: -10px; margin-bottom: -10px; }')
        self._data += ('.x-text { position: relative; border: 2px solid blue; border-radius: 0.4em; '
                       'width: fit-content; margin: 0 auto 12 0;  padding: 5;}')
        self._data += ('.o-text:after { content: \'\'; position: absolute; bottom: 0; left: 75%; width: 0; height: 0; '
                       'border: 10px solid transparent; border-top-color: red; border-bottom: 0; border-left: 0; '
                       'margin-left: -10px; margin-bottom: -10px; }')
        self._data += ('.o-text { position: relative; border: 2px solid red; border-radius: 0.4em; width: fit-content; '
                       'margin: 0 0 12 auto; padding: 5; }')
        self._data += '.main { margin: 0 auto; width: fit-content; }'
        self._data += 'table { border-collapse: collapse; width: fit-content; margin: 0 auto;}'
        self._data += 'tr td { border: 2px solid black; width: 25; height: 25; text-align: center; }'
        self._data += '.o-token { color: red; font-family: Arial, sans-serif; font-weight: bold; }'
        self._data += '.x-token { color: blue; font-family: Arial, sans-serif; font-weight: bold; }'
        self._data += '.brick-token { color: black; font-family: Arial, sans-serif; font-weight: bold; }'
        self._data += '</style>'
        self._data += '</head><body><div class="main">'
        pass

    def _add_p(self, t: str, c: str):
        t = t.replace("\n", "<br/>")
        self._data += f'<p class="{CLASSES[c] if c else ""}">{t}</p>'

    def start_game(self, xi, xn, oi, on):
        self._data += f'<p class="{CLASSES["runner"]}">Players, introduce yourselves!</p>'
        self._add_p("Players, introduce yourselves!", "runner")
        self._data += f'<p class="{CLASSES["runner"]}">Playing as X:</p>'
        self._data += f'<p class="{CLASSES["X"]}"></p>'
        self._data += f'<p class="{CLASSES["X"]}"></p>'

        self._data += f'<p class="{CLASSES["runner"]}">Playing as O:</p>'

        pass

    def print_move(self, player, token, move, state: game.GameState):
        if player:
            self._add_p(f'{player} plays {move}', token)

        self._data += '<table>'

        for row in state.board:
            self._data += '<tr>'
            for col in row:
                self._data += '<td>'
                if not col.isspace():
                    self._data += TOKENS[col]
                self._data += '</td>'
            self._data += '</tr>'
        self._data += '</table><br>'

    def runner_comment(self, text):
        self._add_p(text, "runner")

    def player_comment(self, text, player):
        self._add_p(text, player)

    def generate(self, filename, pdf=False):
        self._data += '</div></body></html>'
        error = False
        if pdf:
            try:
                import asyncio
                from pyppeteer import launch
            except:
                error = True
            if not error:
                async def gen_pdf(content, path):
                    browser = await launch()
                    page = await browser.newPage()
                    await page.setContent(content)
                    await page.pdf({'path': path, 'format': 'letter'})
                    await browser.close()

                try:
                    asyncio.get_event_loop().run_until_complete(gen_pdf(self._data, filename + '.pdf'))
                    print(f'transcript written to {filename}.pdf')
                except:
                    error = True
        if not pdf or error:
            with open(filename + '.html', 'w') as file:
                file.write(self._data)
                print(f'transcript written to {filename}.html')
