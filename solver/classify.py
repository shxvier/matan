"""Автоопределение типа ОДУ первого порядка."""

from __future__ import annotations

import sympy as sp

from solver.model import Classification, ParsedODE
from solver.parser import ParseError

# Порядок — от простейшего метода к более сложному.
ORDER = ["separable", "linear", "homogeneous"]


def normal_form(ode: ParsedODE):
    """y' = f(x, y), где y — обычный символ. Возвращает (f, Y, коэффициент при y')."""
    y, x = ode.y, ode.x
    Y, P = sp.Symbol(ode.func.__name__), sp.Symbol("P_")
    expr = (ode.eq.lhs - ode.eq.rhs).subs(y.diff(x), P).subs(y, Y)
    coeff = sp.simplify(sp.diff(expr, P))
    roots = sp.solve(expr, P)
    if not roots:
        raise ParseError("не удаётся выразить y' из уравнения")
    if len(roots) > 1:
        raise ParseError("y' выражается неоднозначно (несколько ветвей) — запишите уравнение в виде y' = f(x, y)")
    return sp.simplify(roots[0]), Y, coeff


def separable_parts(f, x, Y):
    """f = X(x)·Y(y) -> (X, Y) или None."""
    if not f.has(Y):
        return f, sp.Integer(1)
    if not f.has(x):
        return sp.Integer(1), f
    parts = sp.separatevars(sp.factor(f), symbols=[x, Y], dict=True)
    if parts is None:
        parts = sp.separatevars(sp.powsimp(sp.expand_power_exp(f)), symbols=[x, Y], dict=True)
    if parts is None:
        return None
    return sp.simplify(parts["coeff"] * parts[x]), sp.simplify(parts[Y])


def linear_parts(f, Y):
    """f = a(x)·y + b(x) -> (p, q) для y' + p·y = q, или None."""
    a = sp.simplify(sp.diff(f, Y))
    b = sp.simplify(f - a * Y)
    if a.has(Y) or b.has(Y) or a == 0:
        return None
    return sp.simplify(-a), b


def is_homogeneous(f, x, Y):
    if not f.has(Y) or not f.has(x):
        return False
    t = sp.Symbol("t", positive=True)
    scaled = f.subs({x: t * x, Y: t * Y}, simultaneous=True)
    return sp.simplify(scaled - f) == 0


def classify(ode: ParsedODE, prefer: str | None = None) -> Classification:
    if ode.order > 1:
        cls = Classification(types=["system"], reasons={"system": f"порядок уравнения {ode.order} > 1"})
        cls.chosen = "system"
        return cls

    f, Y, coeff = normal_form(ode)
    x = ode.x
    types, reasons = [], {}

    parts = separable_parts(f, x, Y)
    if parts is not None:
        X, G = parts
        types.append("separable")
        reasons["separable"] = (
            f"правая часть раскладывается в произведение f(x)·g(y): "
            f"f(x) = {sp.sstr(X)}, g(y) = {sp.sstr(G)}"
        )

    lin = linear_parts(f, Y)
    if lin is not None:
        p, q = lin
        types.append("linear")
        extra = f" (после деления на коэффициент при y': {sp.sstr(coeff)})" if coeff != 1 else ""
        reasons["linear"] = f"y и y' входят линейно: y' + p(x)·y = q(x), p = {sp.sstr(p)}, q = {sp.sstr(q)}{extra}"

    if is_homogeneous(f, x, Y):
        types.append("homogeneous")
        reasons["homogeneous"] = "f(tx, ty) = f(x, y) — правая часть зависит только от y/x"

    if not types:
        types, reasons = ["other"], {"other": "не подходит ни один из изучаемых типов"}

    cls = Classification(types=types, reasons=reasons, f=f)
    cls.chosen = prefer if prefer in types else types[0]
    return cls
