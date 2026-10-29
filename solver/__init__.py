"""Решатель ОДУ первого порядка и сведение ОДУ высших порядков к системам.

Логика без ввода-вывода: функции возвращают Solution со списком шагов."""

from __future__ import annotations

import sympy as sp

from solver.classify import classify, normal_form
from solver.homogeneous import solve_homogeneous
from solver.linear import apply_ic_nonelementary, solve_linear
from solver.model import C, ParsedODE, Solution
from solver.parser import ParseError, parse_conditions, parse_ode
from solver.problems import find
from solver.separable import solve_separable
from solver.systems import check_system, reduce_to_system, solve_numeric
from solver.util import apply_ic, tidy_constant
from solver.verify import Verification, run_dsolve, verify

__all__ = ["solve", "solve_problem", "solve_numeric", "ParseError", "parse_ode"]


def solve(source, ic: str | None = None, prefer: str | None = None, check=True, compare=True) -> Solution:
    """Решает уравнение (строку или ParsedODE). ic — «y(0)=1» (для порядка > 1 — все условия)."""
    ode = parse_ode(source) if isinstance(source, str) else source
    cls = classify(ode, prefer)
    sol = Solution(ode=ode, classification=cls, kind=cls.chosen)

    if ode.order > 1:
        res = reduce_to_system(sol)
        if res.linear and res.constant_coeffs:
            general = sp.dsolve(ode.eq)
            sol.general = general if isinstance(general, list) else [general]
            sol.general = [sp.Eq(g.lhs, g.rhs.subs({s: s for s in g.rhs.free_symbols})) for g in sol.general]
            sol.add("Для справки: общее решение исходного уравнения (sympy.dsolve, C₁, C₂, … — постоянные)",
                    sol.general[0] if len(sol.general) == 1 else sol.general)
            if ic:
                conds = parse_conditions(ic, ode)
                ics = {ode.y.diff(ode.x, k).subs(ode.x, v[0]) if k else ode.func(v[0]): v[1] for k, v in conds.items()}
                part = sp.dsolve(ode.eq, ics=ics)
                sol.particular = part if isinstance(part, list) else [part]
                sol.ic = tuple(conds.values())
                sol.add("Частное решение при заданных начальных условиях", sol.particular[0])
        if check:
            sol.verification = Verification(ok=check_system(ode, res), method="обратная подстановка новых функций в систему даёт исходное уравнение")
            if sol.general:
                ok = all(sp.checkodesol(ode.eq, g)[0] for g in sol.general + sol.particular)
                sol.verification.note = "общее решение проверено подстановкой (checkodesol)" if ok else ""
        return sol

    f, Y, coeff = normal_form(ode)

    mu = integrand = None
    elementary = True
    if cls.chosen == "separable":
        solve_separable(sol, f, Y)
    elif cls.chosen == "linear":
        mu, integrand, elementary = solve_linear(sol, f, Y, coeff)
    elif cls.chosen == "homogeneous":
        solve_homogeneous(sol, f, Y)
    else:
        solve_other(sol)

    if ic:
        conds = parse_conditions(ic, ode)
        x0, y0 = conds[0]
        if cls.chosen == "linear" and not elementary:
            apply_ic_nonelementary(sol, mu, integrand, x0, y0)
        else:
            apply_ic(sol, ode, x0, y0)

    if check:
        sol.verification = verify(sol, compare=compare)
    return sol


def solve_other(sol: Solution):
    ode = sol.ode
    sol.add("Уравнение не относится к изучаемым типам — решаем с помощью sympy.dsolve")
    d = run_dsolve(ode.eq)
    if d is None:
        sol.notes.append("sympy.dsolve не нашёл решения")
        sol.add("Решение в замкнутом виде найти не удалось — можно решить численно (--ivp \"y(0)=1\")")
        return
    if any(g.has(sp.Order) for g in (d if isinstance(d, list) else [d])):
        sol.notes.append("ответ — приближение степенным рядом в окрестности x = 0")
    C1 = sp.Symbol("C1")
    d = [g.subs(C1, C) for g in (d if isinstance(d, list) else [d])]
    explicit = [g for g in d if g.lhs == ode.y]
    if explicit:
        sol.general = [sp.Eq(g.lhs, g.rhs if g.has(sp.Order) else tidy_constant(g.rhs)) for g in explicit]
        sol.add("Общее решение", sol.general if len(sol.general) > 1 else sol.general[0])
    else:
        sol.implicit = d[0]
        sol.add("Общий интеграл", d[0])


def solve_problem(code: str, ic: str | None = None, check=True, compare=True) -> Solution:
    problem = find(code)
    return solve(problem.text, ic=ic, prefer=problem.kind, check=check, compare=compare)
