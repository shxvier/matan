"""Общие функции: разбор математических выражений, введённых пользователем."""

import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Имена, которые пользователь может написать «по-школьному».
BASE_NAMES = {
    "e": sp.E,
    "pi": sp.pi,
    "ln": sp.log,
    "lg": lambda t: sp.log(t, 10),
    "tg": sp.tan,
    "ctg": sp.cot,
    "arctg": sp.atan,
    "arcctg": sp.acot,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "sqrt": sp.sqrt,
}


class ParseError(ValueError):
    pass


def parse(text, extra_names=None, decimal_comma=True):
    """Разбирает выражение вида '2x + 3sin(x)', 'x^2', 'e^(3x)' в объект SymPy.

    decimal_comma=True разрешает писать дроби через запятую (0,5); выключите,
    если запятая нужна для разделения аргументов функций: diff(x^2, x).
    """
    text = text.strip()
    if decimal_comma:
        text = text.replace(",", ".")
    if not text:
        raise ParseError("пустое выражение")
    names = dict(BASE_NAMES)
    # Имена вида x1, x2, a10 — отдельные переменные, а не x*1.
    for name in re.findall(r"(?<![A-Za-z_])[A-Za-z]\d+", text):
        names[name] = sp.Symbol(name)
    if extra_names:
        names.update(extra_names)
    try:
        return parse_expr(text, local_dict=names, transformations=TRANSFORMATIONS)
    except Exception as exc:  # noqa: BLE001 - любые ошибки парсера сводим к одной
        raise ParseError(f"не удалось разобрать «{text}»: {exc}") from None


def parse_equation(text, extra_names=None):
    """Разбирает уравнение 'левая = правая' и возвращает выражение 'левая - правая'."""
    if text.count("=") != 1:
        raise ParseError("в уравнении должен быть ровно один знак «=»")
    left, right = text.split("=")
    return parse(left, extra_names) - parse(right, extra_names)


def show(expr):
    """Красивый многострочный вывод выражения."""
    if isinstance(expr, sp.Basic):
        expr = expr.replace(sp.log, sp.Function("ln"))
    return sp.pretty(expr, use_unicode=True)


def print_block(title, expr):
    lines = show(expr).splitlines()
    if len(lines) == 1:
        print(f"{title} {lines[0]}")
        return
    print(title)
    for line in lines:
        print("    " + line)
