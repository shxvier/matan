"""Общие помощники решателей: интегралы, константы, выражение y, начальные условия."""

from __future__ import annotations

import sympy as sp
from sympy.solvers.ode.ode import constantsimp

from solver.model import C, Step

# Функции, появление которых означает «интеграл не выражается в элементарных функциях».
NONELEMENTARY = (sp.erf, sp.erfi, sp.erfc, sp.Ei, sp.Si, sp.Ci, sp.li, sp.fresnels, sp.fresnelc,
                 sp.expint, sp.uppergamma, sp.lowergamma, sp.polylog, sp.Shi, sp.Chi)


def integrate(expr, var):
    """∫ expr d(var). Возвращает (результат, элементарен ли он).

    Неэлементарный интеграл остаётся в виде Integral(...)."""
    result = sp.integrate(expr, var)
    if result.has(sp.Integral) or any(result.has(f) for f in NONELEMENTARY):
        return sp.Integral(expr, var), False
    return sp.simplify(result), True


def abs_logs(expr, *, only_var=None):
    """ln(u) -> ln|u| для аргументов, которые могут быть отрицательными."""

    def fix(arg):
        if isinstance(arg, sp.Abs) or arg.is_positive:
            return sp.log(arg)
        if only_var is not None and not arg.free_symbols <= {only_var}:
            return sp.log(arg)
        return sp.log(sp.Abs(arg))

    return expr.replace(sp.log, fix)


def tidy_constant(expr):
    """Переобозначение произвольной постоянной: e^(C + x) -> C·e^x, 2C -> C, -C -> C и т. п."""
    expr = sp.simplify(expr)
    try:
        expr = constantsimp(expr, [C])
    except Exception:  # noqa: BLE001 - не смогли упростить — оставляем как есть
        pass
    candidates = [expr]
    if expr.has(C):
        flipped = sp.simplify(expr.subs(C, -C))
        candidates.append(flipped)
        for e in list(candidates):
            if e.has(sp.log):
                candidates.append(sp.expand_log(e, force=True))
    return min(candidates, key=lambda e: (sp.count_ops(e), len(str(e))))


def solve_for(eq: sp.Eq, Y: sp.Symbol):
    """Явные решения уравнения eq относительно Y (пустой список, если не выражается)."""
    if not eq.has(Y):
        return []
    try:
        sols = sp.solve(eq, Y)
    except (NotImplementedError, ValueError, TypeError):
        return []
    sols = [s for s in sols if not s.has(sp.RootOf, sp.Piecewise, sp.LambertW)]
    return [tidy_constant(s) for s in sols]


def real_roots(expr, var):
    """Вещественные корни-константы уравнения expr = 0."""
    try:
        roots = sp.solve(expr, var)
    except (NotImplementedError, ValueError):
        return []
    return [r for r in roots if not r.free_symbols and r.is_real is not False]


def included_in(value, explicit, x):
    """Значение C, при котором y = value получается из общего решения (или None)."""
    for sol in explicit:
        try:
            cs = sp.solve(sp.Eq(sol, value), C)
        except NotImplementedError:
            continue
        for c in cs:
            if not c.has(x):
                return c
        if sp.simplify(sol.subs(C, 0) - value) == 0:
            return sp.Integer(0)
    return None


def invariant(sol: sp.Eq, y, Y):
    """Выражение Φ(x, Y), равное C вдоль решений, или None."""
    expr = (sol.lhs - sol.rhs).subs(y, Y)
    if not expr.has(C):
        return None
    try:
        cs = sp.solve(expr, C, check=False)
    except NotImplementedError:
        return None
    return cs[0] if cs else None


def apply_ic(sol, ode, x0, y0):
    """Находит C из y(x0) = y0, добавляет шаги и частное решение в sol."""
    x, y = ode.x, ode.y
    sol.ic = (x0, y0)
    sol.add(f"Подставляем начальное условие {ode.func.__name__}({x0}) = {y0}")

    candidates = []
    for g in sol.general:
        try:
            cs = sp.solve(sp.Eq(g.rhs.subs(x, x0), y0), C)
        except NotImplementedError:
            continue
        for c in cs:
            if c.is_real is False:
                continue
            candidates.append((g, sp.simplify(c)))
    if candidates:
        for g, c in candidates:
            sol.add("Получаем уравнение для C", sp.Eq(g.rhs.subs(x, x0), y0))
            sol.add("Находим постоянную", sp.Eq(C, c))
            part = sp.Eq(y, sp.simplify(g.rhs.subs(C, c)))
            if part not in sol.particular:
                sol.particular.append(part)
        if sol.particular:
            sol.add("Частное решение", sol.particular[0] if len(sol.particular) == 1 else sol.particular)
            return
    if sol.implicit is not None:
        Y = sp.Symbol("Y")
        expr = sol.implicit.subs(y, Y)
        cs = sp.solve(expr.subs({x: x0, Y: y0}), C)
        cs = [c for c in cs if c.is_real is not False]
        if cs:
            c = sp.simplify(cs[0])
            sol.add("Подставляем x0, y0 в общий интеграл и находим C", sp.Eq(C, c))
            sol.particular_implicit = sp.Eq(sol.implicit.lhs.subs(C, c), sol.implicit.rhs.subs(C, c))
            sol.add("Частный интеграл", sol.particular_implicit)
            return
    for s, _ in sol.singular:
        if sp.simplify(s.rhs.subs(x, x0) - y0) == 0:
            sol.particular.append(s)
            sol.add("Условию удовлетворяет особое решение", s)
            return
    sol.notes.append("не удалось подобрать C под начальное условие")
    sol.steps.append(Step("Подобрать C под начальное условие не удалось"))
