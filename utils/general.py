from typing import Literal

def colorize(string: str, color: Literal['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']) -> str:
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
    }
    reset = '\033[0m'
    color_code = colors.get(color, '')
    return f"{color_code}{string}{reset}"

def cprint(string: str, color: Literal['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'], end: str = '\n') -> None:
    print(colorize(string, color), end=end)