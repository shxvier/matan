"""Разбор ОДУ, записанного «как в тетради»: y' = x/y, x*y' + y = x^2, theta'' + sin(theta) = 0."""

from __future__ import annotations

import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from solver.model import ParsedODE

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)

NAMES = {
    "e": sp.E,
    "pi": sp.pi,
    "ln": sp.log,
    "lg": lambda t: sp.log(t, 10),
    "tg": sp.tan,
    "ctg": sp.cot,
    "arctg": sp.atan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "abs": sp.Abs,
}

UNICODE = {"−": "-", "–": "-", "·": "*", "×": "*", "²": "^2", "³": "^3", "θ": "theta", "π": "pi", "′": "'", "’": "'"}
GREEK = {"theta", "phi", "psi", "alpha", "beta", "omega", "rho", "eta", "xi", "tau", "chi", "nu", "mu"}
KNOWN_FUNCS = set(NAMES) | {"atan", "asin", "acos", "sinh", "cosh", "tanh", "Abs", "log"}


class ParseError(ValueError):
    """Ошибка во вводе: сообщение уже на русском."""


def _clean(text: str) -> str:
    for old, new in UNICODE.items():
        text = text.replace(old, new)
    return text.strip()


def parse_expression(text: str, names: dict | None = None) -> sp.Basic:
    """Выражение без производных: '2xy', 'e^x', 'ln(x)'."""
    text = _clean(text)
    if not text:
        raise ParseError("пустое выражение")
    if text.count("(") != text.count(")"):
        raise ParseError(f"не совпадает число открывающих и закрывающих скобок в «{text}»")
    if re.search(r"[+\-*/^]\s*$", text):
        raise ParseError(f"выражение «{text}» обрывается на знаке операции")
    bad = re.search(r"[^\w\s+\-*/^().,'<>=!]", text)
    if bad:
        raise ParseError(f"недопустимый символ «{bad.group(0)}» в «{text}»")
    if re.search(r"[а-яА-ЯёЁ]", text):
        raise ParseError(f"русские буквы в формуле «{text}» — переменные и функции пишутся латиницей")
    local = dict(NAMES)
    if names:
        local.update(names)
    for m in re.finditer(r"([A-Za-z_]\w*)\s*\(", text):
        name = m.group(1)
        if len(name) > 1 and name not in local and name not in KNOWN_FUNCS and not hasattr(sp, name):
            raise ParseError(f"неизвестная функция «{name}» в «{text}»")
    try:
        return parse_expr(text, local_dict=local, transformations=TRANSFORMATIONS)
    except Exception as exc:  # noqa: BLE001 - любые ошибки разбора сводим к одной
        raise ParseError(f"не удалось разобрать «{text}» ({type(exc).__name__}): проверьте знаки и скобки") from None


def _split_domain(text: str):
    """'y' = (y/x) ln x, x>0' -> ('y' = (y/x) ln x', 'x>0')."""
    m = re.search(r"[,;]\s*([A-Za-z]\w*\s*(?:>=|<=|>|<|!=)\s*[^,;=]+)\s*$", text)
    if m:
        return text[: m.start()].strip(), m.group(1).strip()
    return text, None


def parse_ode(text: str) -> ParsedODE:
    raw = text
    text = _clean(text)
    if not text:
        raise ParseError("введите уравнение, например: y' = x/y")
    text, domain_text = _split_domain(text)

    text = re.sub(r"\bdy\s*/\s*dx\b", "y'", text)
    text = re.sub(r"([A-Za-z_]\w*)\s*\(\s*[xt]\s*\)\s*('+)", r"\1\2", text)  # y(x)' -> y'
    tokens = list(re.finditer(r"([A-Za-z_][A-Za-z_0-9]*)\s*('+)", text))
    if not tokens:
        raise ParseError("в уравнении нет производной — пишите y', y'' (или dy/dx)")

    def split_name(ident):
        """'xy' в «2xy'» — это x·y', а 'theta' — имя функции целиком."""
        if len(ident) == 1 or ident in GREEK or not ident.isalpha():
            return "", ident
        return ident[:-1], ident[-1]

    names = {split_name(m.group(1))[1] for m in tokens}
    if len(names) > 1:
        raise ParseError(f"производные разных функций ({', '.join(sorted(names))}) — нужна одна неизвестная функция")
    fname = names.pop()
    order = max(len(m.group(2)) for m in tokens)
    if order > 9:
        raise ParseError("порядок уравнения больше 9 — это вряд ли учебная задача")

    # Независимая переменная: x, а если в уравнении нет x, но есть t — t.
    var_name = "x"
    if fname == "x":
        raise ParseError("неизвестная функция не может называться x")
    without_f = re.sub(rf"(?<![A-Za-z_]){fname}(?![A-Za-z_0-9])", "", text)
    if not re.search(r"(?<![A-Za-z_])x(?![A-Za-z_0-9])", without_f) and re.search(r"(?<![A-Za-z_])t(?![A-Za-z_0-9])", without_f):
        var_name = "t"

    # Область определения: x>0 даёт x положительный, иначе просто вещественный.
    domain = None
    x = sp.Symbol(var_name, real=True)
    if domain_text:
        dm = re.fullmatch(r"([A-Za-z]\w*)\s*(>=|<=|>|<|!=)\s*(.+)", domain_text)
        if not dm or dm.group(1) != var_name:
            raise ParseError(f"область «{domain_text}» должна быть условием на {var_name}, например {var_name}>0")
        bound = parse_expression(dm.group(3))
        if dm.group(2) == ">" and bound == 0:
            x = sp.Symbol(var_name, positive=True)
        elif dm.group(2) == "<" and bound == 0:
            x = sp.Symbol(var_name, negative=True)
        domain = sp.Rel(x, bound, dm.group(2))

    # y(x) -> y, производные -> служебные символы D1_, D2_, ...; сама функция -> символ F_.
    text = re.sub(rf"(?<![A-Za-z_0-9]){fname}\s*\(\s*{var_name}\s*\)", fname, text)

    def replace_derivative(m):
        prefix, _ = split_name(m.group(1))
        return f" {prefix} D{len(m.group(2))}_ "

    text = re.sub(r"([A-Za-z_][A-Za-z_0-9]*)\s*('+)", replace_derivative, text)
    placeholders = {f"D{k}_": sp.Symbol(f"D{k}_") for k in range(1, order + 1)}
    placeholders[fname] = sp.Symbol("F_")
    if "'" in text:
        raise ParseError("непонятный штрих «'» — производная пишется сразу после имени функции: y'")

    if text.count("=") > 1:
        raise ParseError("в уравнении должен быть один знак «=»")
    if "=" in text:
        left, right = text.split("=")
        if not left.strip() or not right.strip():
            raise ParseError("одна из частей уравнения пустая")
    else:
        left, right = text, "0"

    names_map = {**placeholders, var_name: x}
    lhs, rhs = parse_expression(left, names_map), parse_expression(right, names_map)

    func = sp.Function(fname)
    yx = func(x)
    subs = {placeholders[fname]: yx}
    for k in range(1, order + 1):
        subs[placeholders[f"D{k}_"]] = yx.diff(x, k)
    expr = (lhs - rhs).xreplace(subs)

    extra = expr.free_symbols - {x}
    if extra:
        raise ParseError(f"лишние обозначения: {', '.join(sorted(map(str, extra)))} (неизвестная функция — {fname}, переменная — {var_name})")
    if not expr.has(yx.diff(x, order)):
        raise ParseError("старшая производная сокращается — проверьте запись")

    eq = sp.Eq((lhs).xreplace(subs), (rhs).xreplace(subs))
    return ParsedODE(text=raw.strip(), eq=eq, func=func, x=x, order=order, domain=domain)


def parse_conditions(text: str, ode: ParsedODE) -> dict[int, tuple]:
    """'y(0)=1, y'(0)=0' -> {0: (x0, 1), 1: (x0, 0)}."""
    text = _clean(text)
    if not text:
        raise ParseError("пустое начальное условие")
    result: dict[int, tuple] = {}
    name = ode.func.__name__
    for part in re.split(r"[,;]\s*(?=[A-Za-z_])", text):
        m = re.fullmatch(rf"\s*{name}\s*('*)\s*\((.+?)\)\s*=\s*(.+?)\s*", part)
        if not m:
            raise ParseError(f"условие «{part.strip()}» записывайте так: {name}(0)=1, {name}'(0)=0")
        k = len(m.group(1))
        if k >= ode.order:
            raise ParseError(f"для уравнения порядка {ode.order} нужны условия на {name}, …, производную порядка {ode.order - 1}")
        if k in result:
            raise ParseError(f"условие на {name}{chr(39) * k} задано дважды")
        x0, v0 = parse_expression(m.group(2)), parse_expression(m.group(3))
        if x0.free_symbols or v0.free_symbols:
            raise ParseError("в начальных условиях должны быть числа")
        result[k] = (x0, v0)
    points = {v[0] for v in result.values()}
    if len(points) > 1:
        raise ParseError("все начальные условия должны быть заданы в одной точке")
    missing = [k for k in range(ode.order) if k not in result]
    if missing:
        names = ", ".join(f"{name}{chr(39) * k}(x0)" for k in missing)
        raise ParseError(f"не хватает начальных условий: {names}")
    x0 = points.pop()
    if ode.domain is not None and ode.domain.subs(ode.x, x0) == sp.false:
        raise ParseError(f"точка {ode.x}0 = {x0} не входит в область {ode.domain}")
    return result
