"""Учебный набор задач (блоки А–Г)."""

from __future__ import annotations

from dataclasses import dataclass

BLOCKS = {
    "А": ("Разделяющиеся переменные", "separable"),
    "Б": ("Линейные", "linear"),
    "В": ("Однородные", "homogeneous"),
    "Г": ("Свести к системе первого порядка", "system"),
}


@dataclass(frozen=True)
class Problem:
    code: str
    text: str
    ivp: str | None = None  # начальные условия по умолчанию (для численного режима)

    @property
    def block(self):
        return self.code[0]

    @property
    def kind(self):
        return BLOCKS[self.block][1]

    @property
    def block_title(self):
        return BLOCKS[self.block][0]


PROBLEMS = [
    Problem("А1", "y' = x/y"),
    Problem("А2", "y' = e^(x+y)"),
    Problem("А3", "y' = y*cos(x)"),
    Problem("А4", "y' = x^2/(y(1+x^3))"),
    Problem("А5", "y' = 1+y^2"),
    Problem("А6", "y' = (y/x)*ln(x), x>0"),
    Problem("А7", "y' = e^x/y"),
    Problem("А8", "y' = 1/(x*y)"),
    Problem("Б1", "y' + 2y = 4"),
    Problem("Б2", "y' - y = x"),
    Problem("Б3", "y' + y*tan(x) = sin(2x)"),
    Problem("Б4", "x*y' + y = x^2, x>0"),
    Problem("Б5", "y' + 2x*y = x*e^(-x^2)"),
    Problem("Б6", "y' + y = 1/(1+x^2)"),
    Problem("Б7", "y' + (2/x)*y = x^3, x>0"),
    Problem("Б8", "y' - y = e^(2x)"),
    Problem("В1", "y' = y/x"),
    Problem("В2", "y' = (x+y)/(x-y)"),
    Problem("В3", "y' = (y^2-x^2)/(2x*y)"),
    Problem("В4", "y' = x/y + y/x"),
    Problem("В5", "x*y' = y + x*e^(y/x)"),
    Problem("В6", "y' = y/x + x/y"),
    Problem("Г1", "y'' + 4y = 0", "y(0)=1, y'(0)=0"),
    Problem("Г2", "y''' - y = 0", "y(0)=1, y'(0)=0, y''(0)=0"),
    Problem("Г3", "y'' + 2y' + 5y = sin(x)", "y(0)=0, y'(0)=1"),
    Problem("Г4", "theta'' + sin(theta) = 0", "theta(0)=1, theta'(0)=0"),
]

BY_CODE = {p.code: p for p in PROBLEMS}

# Латинские буквы, которыми удобно набирать номер: B5 -> Б5, A1 -> А1.
LATIN = {"A": "А", "B": "Б", "V": "В", "G": "Г", "W": "В"}
SIMILAR = {"а": "А", "б": "Б", "в": "В", "г": "Г"}


def normalize(code: str) -> str:
    code = code.strip().replace(" ", "")
    if not code:
        return code
    first = code[0]
    first = LATIN.get(first.upper(), SIMILAR.get(first.lower(), first.upper()))
    return first + code[1:]


def find(code: str) -> Problem:
    key = normalize(code)
    if key not in BY_CODE:
        codes = ", ".join(p.code for p in PROBLEMS)
        raise KeyError(f"задачи «{code}» нет. Доступны: {codes}")
    return BY_CODE[key]
