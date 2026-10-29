"""Уравнения с разделяющимися переменными: y' = f(x)·g(y)."""

from __future__ import annotations

import sympy as sp

from solver.classify import separable_parts
from solver.model import C, Solution
from solver.util import abs_logs, included_in, integrate, real_roots, solve_for

# Дифференциалы некоммутативны, чтобы печатались справа: x·dx, а не dx·x.
DX = sp.Symbol("dx", commutative=False)
DY = sp.Symbol("dy", commutative=False)


def differential(name):
    return sp.Symbol(f"d{name}", commutative=False)


def solve_separable(sol: Solution, f, Y):
    ode = sol.ode
    x, y = ode.x, ode.y
    X, G = separable_parts(f, x, Y)
    dY, dX = differential(Y.name), differential(x.name)
    yp = sp.Symbol(f"{Y.name}'")

    sol.add("Записываем уравнение в виде y' = f(x)·g(y)", sp.Eq(yp, sp.Mul(X, G, evaluate=False)))
    sol.add("Множители", [sp.Eq(sp.Symbol(f"f({x})"), X), sp.Eq(sp.Symbol(f"g({Y})"), G)])

    if G == 1:
        sol.add("Правая часть не зависит от y: dy = f(x)·dx", sp.Eq(dY, X * dX))
        left_integrand = sp.Integer(1)
    else:
        sol.add("Заменяем y' = dy/dx и разделяем переменные (при g(y) ≠ 0)", sp.Eq((1 / G) * dY, X * dX))
        left_integrand = sp.simplify(1 / G)

    sol.add("Интегрируем обе части", sp.Eq(sp.Integral(left_integrand, Y), sp.Integral(X, x)))
    left, ok_left = integrate(left_integrand, Y)
    right, ok_right = integrate(X, x)
    if not (ok_left and ok_right):
        sol.nonelementary = True
        sol.notes.append("интеграл не выражается в элементарных функциях — оставлен в виде ∫")

    right = abs_logs(right, only_var=x) if ok_right else right
    shown_left = abs_logs(left) if ok_left else left
    implicit = sp.Eq(shown_left, right + C)
    note = None
    if ok_left and left.has(sp.log):
        note = "ln берётся по модулю; при потенцировании ±e^C снова обозначаем через C"
    sol.add("Вычисляем интегралы — получаем общий интеграл (C — произвольная постоянная)", implicit, note)

    explicit = solve_for(sp.Eq(left, right + C), Y) if ok_left else []
    if explicit:
        sol.general = [sp.Eq(y, s) for s in explicit]
        title = "Выражаем y — общее решение" if len(explicit) == 1 else "Выражаем y — общее решение (несколько ветвей)"
        sol.add(title, sol.general if len(sol.general) > 1 else sol.general[0])
    else:
        sol.add("y явно не выражается — ответ записываем в виде общего интеграла", implicit)
    sol.implicit = sp.Eq(implicit.lhs.subs(Y, y), implicit.rhs)

    if G != 1:
        roots = real_roots(G, Y)
        if roots:
            sol.add("Проверяем решения, потерянные при делении на g(y): g(y) = 0", sp.Eq(G, 0))
            for r in roots:
                c = included_in(r, explicit, x)
                if c is not None:
                    text = f"входит в общее решение при C = {sp.sstr(c)}"
                else:
                    text = "особое решение (не получается из общего ни при каком C)"
                sol.singular.append((sp.Eq(y, r), text))
                sol.add(f"y = {sp.sstr(r)}: {text}", sp.Eq(y, r))
        else:
            sol.add("Уравнение g(y) = 0 вещественных корней не имеет — особых решений нет")
