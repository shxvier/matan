"""Структуры данных, которые возвращают решатели. Печатью занимается только cli.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

# Произвольная постоянная во всех ответах.
C = sp.Symbol("C")

TYPE_NAMES = {
    "separable": "с разделяющимися переменными",
    "linear": "линейное первого порядка",
    "homogeneous": "однородное",
    "system": "уравнение высшего порядка → система первого порядка",
    "other": "тип не из изучаемых (решено SymPy)",
}


@dataclass
class Step:
    """Один шаг решения: подпись на русском и формула (или None)."""

    title: str
    expr: Any = None
    note: str | None = None


@dataclass
class ParsedODE:
    text: str  # исходная запись
    eq: sp.Eq  # уравнение с функцией y(x) и её производными
    func: sp.Function  # y
    x: sp.Symbol  # независимая переменная
    order: int
    domain: sp.Basic | None = None  # например x > 0

    @property
    def y(self):
        return self.func(self.x)


@dataclass
class Classification:
    types: list[str]  # все подходящие типы, по возрастанию сложности метода
    reasons: dict[str, str]  # тип -> признак, по которому он определён
    f: sp.Basic | None = None  # правая часть y' = f(x, y) (y — символ)
    chosen: str | None = None

    @property
    def others(self):
        return [t for t in self.types if t != self.chosen]


@dataclass
class Verification:
    ok: bool | None = None  # подстановка в уравнение
    method: str = ""
    dsolve: Any = None  # ответ sympy.dsolve
    equivalent: bool | None = None  # совпадает с dsolve с точностью до C
    note: str = ""


@dataclass
class SystemResult:
    n: int
    new_vars: list[sp.Eq]  # y1 = y, y2 = y', ...
    equations: list[sp.Eq]  # y1' = y2, ...
    funcs: list[sp.Function]  # y1, y2, ...
    rhs: list[sp.Basic]  # правые части через символы y1..yn
    symbols: list[sp.Symbol]  # символы y1..yn
    linear: bool = False
    constant_coeffs: bool = False
    A: sp.Matrix | None = None
    F: sp.Matrix | None = None


@dataclass
class NumericResult:
    xs: Any  # точки таблицы
    values: Any  # массив n × len(xs)
    dense_x: Any = None  # густая сетка для графика
    dense_values: Any = None
    names: list[str] = field(default_factory=list)
    ivp: dict = field(default_factory=dict)
    success: bool = True
    message: str = ""


@dataclass
class Solution:
    ode: ParsedODE
    classification: Classification
    kind: str
    steps: list[Step] = field(default_factory=list)
    general: list[sp.Eq] = field(default_factory=list)  # явные решения y = ...
    implicit: sp.Eq | None = None  # общий интеграл Φ(x, y) = C
    singular: list[tuple[sp.Eq, str]] = field(default_factory=list)  # (решение, пояснение)
    particular: list[sp.Eq] = field(default_factory=list)
    particular_implicit: sp.Eq | None = None
    ic: tuple | None = None  # (x0, y0)
    nonelementary: bool = False
    verification: Verification | None = None
    system: SystemResult | None = None
    numeric: NumericResult | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def type_name(self):
        return TYPE_NAMES[self.kind]

    def add(self, title, expr=None, note=None):
        self.steps.append(Step(title, expr, note))
