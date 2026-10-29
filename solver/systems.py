"""Сведение ОДУ n-го порядка к нормальной системе первого порядка и её численное решение."""

from __future__ import annotations

import numpy as np
import sympy as sp

from solver.model import NumericResult, Solution, SystemResult
from solver.parser import ParseError, parse_conditions

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def pretty_name(name: str) -> str:
    """theta -> θ, y -> y (как печатает sympy)."""
    return sp.pretty(sp.Symbol(name), use_unicode=True)


def deriv_name(name: str, k: int) -> str:
    base = pretty_name(name)
    if k <= 3:
        return base + "'" * k
    return f"{base}⁽{str(k).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}⁾"


def highest_derivative(ode):
    """y⁽ⁿ⁾ = G(x, y, y', …, y⁽ⁿ⁻¹⁾); производные заменены символами Z0…Zn."""
    x, y, n = ode.x, ode.y, ode.order
    Z = [sp.Symbol(f"Z{k}_") for k in range(n + 1)]
    expr = ode.eq.lhs - ode.eq.rhs
    for k in range(n, 0, -1):
        expr = expr.subs(y.diff(x, k), Z[k])
    expr = expr.subs(y, Z[0])
    roots = sp.solve(expr, Z[n])
    if not roots:
        raise ParseError("не удаётся выразить старшую производную")
    if len(roots) > 1:
        raise ParseError("старшая производная выражается неоднозначно — запишите уравнение в виде y⁽ⁿ⁾ = …")
    return sp.expand(roots[0]), Z


def reduce_to_system(sol: Solution, with_steps=True):
    ode = sol.ode
    x, n, name = ode.x, ode.order, ode.func.__name__
    G, Z = highest_derivative(ode)

    shown = {Z[k]: sp.Symbol(deriv_name(name, k)) for k in range(n)}
    plain = [sp.Symbol(f"{name}{k + 1}") for k in range(n)]
    shown_new = [sp.Symbol(pretty_name(name) + str(k + 1).translate(SUB)) for k in range(n)]
    shown_new_d = [sp.Symbol(pretty_name(name) + str(k + 1).translate(SUB) + "'") for k in range(n)]

    rhs = [plain[k + 1] for k in range(n - 1)] + [G.subs({Z[k]: plain[k] for k in range(n)})]
    shown_rhs = [r.subs(dict(zip(plain, shown_new))) for r in rhs]
    new_vars = [sp.Eq(shown_new[k], shown[Z[k]]) for k in range(n)]
    equations = [sp.Eq(shown_new_d[k], shown_rhs[k]) for k in range(n)]

    res = SystemResult(n=n, new_vars=new_vars, equations=equations, funcs=[], rhs=rhs, symbols=plain)
    if with_steps:
        sol.add("Выражаем старшую производную", sp.Eq(sp.Symbol(deriv_name(name, n)), G.subs(shown)))
        sol.add(f"Вводим новые неизвестные функции (их {n} — по порядку уравнения)", new_vars)
        sol.add("Производная каждой новой функции — следующая функция, последняя — из уравнения", equations)

    # Линейность и постоянство коэффициентов
    J = sp.Matrix([[sp.diff(r, s) for s in plain] for r in rhs])
    res.linear = all(not e.has(*plain) for e in J)
    if res.linear:
        res.A = J
        res.F = sp.simplify(sp.Matrix(rhs) - J * sp.Matrix(plain))
        res.constant_coeffs = not J.has(x)
        if with_steps:
            Yv = sp.ImmutableMatrix(shown_new)
            A = sp.ImmutableMatrix(res.A)
            prod = sp.MatMul(A, Yv, evaluate=False)
            form = prod if res.F.is_zero_matrix else sp.MatAdd(prod, sp.ImmutableMatrix(res.F), evaluate=False)
            kind = "с постоянными коэффициентами" if res.constant_coeffs else "с переменными коэффициентами"
            title = "Матричная форма Y' = A·Y" + ("" if res.F.is_zero_matrix else " + F(x)")
            sol.add(f"Система линейная {kind}. {title}", sp.Eq(sp.ImmutableMatrix(shown_new_d), form))
            sol.add("Матрица системы A", res.A)
            if not res.F.is_zero_matrix:
                sol.add("Неоднородность F(x)", res.F)
            if res.constant_coeffs:
                lam = sp.Symbol("λ")
                charpoly = sp.factor(res.A.charpoly(lam).as_expr())
                sol.add("Характеристическое уравнение det(A − λE) = 0", sp.Eq(charpoly, 0))
                eigen = res.A.eigenvals()
                multiple = [f"{sp.sstr(v)} (кратность {m})" for v, m in eigen.items() if m > 1]
                note = "кратные корни: " + ", ".join(multiple) if multiple else None
                sol.add("Собственные числа матрицы A", [sp.Eq(lam, v) for v in eigen], note)
    elif with_steps:
        sol.add("Система нелинейная — аналитическое решение не ищем; "
                "для численного решения задайте начальные условия (--ivp)")
    sol.system = res
    return res


def check_system(ode, res: SystemResult) -> bool:
    """Подставляем y₁ = y, y₂ = y', … обратно: должна получиться исходная уравнение."""
    x, y, n = ode.x, ode.y, ode.order
    back = {res.symbols[k]: y.diff(x, k) for k in range(n)}
    expr = (ode.eq.lhs - ode.eq.rhs).subs(y.diff(x, n), res.rhs[-1].subs(back))
    return sp.simplify(expr) == 0


def solve_numeric(sol: Solution, ivp_text: str, interval=None, npoints=13) -> NumericResult:
    from scipy.integrate import solve_ivp

    ode = sol.ode
    res = sol.system or reduce_to_system(sol, with_steps=False)
    conds = parse_conditions(ivp_text, ode)
    x0 = float(conds[0][0])
    y0 = [float(conds[k][1]) for k in range(ode.order)]
    a, b = (x0, x0 + 10.0) if interval is None else (float(interval[0]), float(interval[1]))
    if not a < b:
        raise ParseError("начало отрезка должно быть меньше конца: --interval 0 10")
    if not a <= x0 <= b:
        raise ParseError(f"точка начальных условий {x0:g} должна лежать на отрезке [{a:g}; {b:g}]")

    f = sp.lambdify((ode.x, res.symbols), res.rhs, modules="numpy")

    def rhs(t, Y):
        return np.array(f(t, Y), dtype=float)

    pieces = []
    message, ok = "", True
    for end in (a, b):
        if end == x0:
            continue
        r = solve_ivp(rhs, (x0, end), y0, method="RK45", dense_output=True, rtol=1e-9, atol=1e-11)
        ok = ok and r.success
        message = r.message
        pieces.append((min(x0, end), max(x0, end), r.sol))

    def evaluate(points):
        out = np.empty((ode.order, len(points)))
        for i, p in enumerate(points):
            for lo, hi, s in pieces:
                if lo <= p <= hi:
                    out[:, i] = s(p)
                    break
            else:
                out[:, i] = y0
        return out

    xs = np.linspace(a, b, npoints)
    dense = np.linspace(a, b, 600)
    names = [deriv_name(ode.func.__name__, k) for k in range(ode.order)]
    result = NumericResult(xs=xs, values=evaluate(xs), dense_x=dense, dense_values=evaluate(dense),
                           names=names, ivp={k: v for k, v in conds.items()}, success=ok, message=str(message))
    sol.numeric = result
    sol.add(f"Численное решение задачи Коши методом Рунге–Кутты (solve_ivp, RK45) на [{a:g}; {b:g}]",
            [sp.Eq(sp.Symbol(f"{names[k]}({sp.sstr(conds[k][0])})"), conds[k][1]) for k in range(ode.order)])
    return result
