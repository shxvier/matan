"""Однородные уравнения y' = F(y/x): замена y = u·x."""

from __future__ import annotations

import sympy as sp

from solver.model import C, Solution
from solver.separable import differential
from solver.util import included_in, integrate, real_roots, solve_for


def solve_homogeneous(sol: Solution, f, Y):
    ode = sol.ode
    x, y = ode.x, ode.y
    t = sp.Symbol("t", positive=True)
    u = sp.Symbol("u")
    # Штрихованные символы некоммутативны, чтобы печаталось x·u', а не u'·x.
    yp, up = sp.Symbol(f"{Y.name}'"), sp.Symbol("u'", commutative=False)

    sol.add("Записываем уравнение в виде y' = F(x, y)", sp.Eq(yp, f))
    scaled = f.subs({x: t * x, Y: t * Y}, simultaneous=True)
    sol.add("Проверяем однородность: подставляем tx и ty вместо x и y — t сокращается, F(tx, ty) = F(x, y)",
            sp.Eq(sp.Symbol("F(tx, ty)"), sp.simplify(scaled), evaluate=False))

    # φ(u) = F(1, u); при x > 0 корни вида √(x²) упрощаются.
    xp = sp.Symbol(x.name, positive=True)
    phi = sp.simplify(f.subs(x, xp).subs(Y, u * xp))
    if phi.has(xp):
        phi = sp.simplify(f.subs({x: 1, Y: u}))
        sol.notes.append("φ(u) = F(1, u) найдена для x > 0")
    sol.add("Замена y = u·x, тогда y' = u + x·u'", sp.Eq(u + x * up, phi))

    g = sp.simplify(phi - u)
    sol.add("Переносим u: x·u' = φ(u) − u", sp.Eq(x * up, g))
    if g == 0:
        sol.add("Получили u' = 0, значит u = C")
        sol.general = [sp.Eq(y, C * x)]
        sol.add("Обратная замена y = u·x — общее решение", sol.general[0])
        return

    du, dx = differential("u"), differential(x.name)
    sol.add("Разделяем переменные", sp.Eq((1 / g) * du, (1 / x) * dx))
    sol.add("Интегрируем обе части", sp.Eq(sp.Integral(sp.simplify(1 / g), u), sp.Integral(1 / x, x)))
    G, ok = integrate(sp.simplify(1 / g), u)
    if not ok:
        sol.nonelementary = True
        sol.notes.append("интеграл по u не выражается в элементарных функциях")
    lnx = sp.log(x) if x.is_positive else sp.log(sp.Abs(x))
    sol.add("Вычисляем интегралы", sp.Eq(G, lnx + C))

    back = sp.simplify(G.subs(u, Y / x))
    implicit = sp.Eq(back, lnx + C)
    sol.add("Обратная замена u = y/x — общий интеграл", implicit)
    sol.implicit = sp.Eq(implicit.lhs.subs(Y, y), implicit.rhs)

    explicit = solve_for(sp.Eq(G, lnx + C), u) if ok else []
    if explicit:
        branches = [drop_abs(ode, f, sp.simplify(x * e), Y) for e in explicit]
        sol.general = [sp.Eq(y, b) for b in simplify_branches(ode, f, Y, branches)]
        sol.add("Выражаем y = u·x — общее решение", sol.general if len(sol.general) > 1 else sol.general[0])
    else:
        sol.add("y явно не выражается — ответ записываем в виде общего интеграла", implicit)

    roots = real_roots(g, u)
    for r in roots:
        cand = r * x
        c = included_in(cand, [g_.rhs for g_ in sol.general], x)
        text = f"входит в общее решение при C = {sp.sstr(c)}" if c is not None else "особое решение"
        sol.singular.append((sp.Eq(y, cand), text))
        sol.add(f"При φ(u) − u = 0 (u = {sp.sstr(r)}) получаем y = {sp.sstr(cand)}: {text}", sp.Eq(y, cand))


def satisfies(ode, f, Y, expr):
    """Численная проверка: expr — решение y' = f при x разных знаков и разных C."""
    x = ode.x
    residual = sp.diff(expr, x) - f.subs(Y, expr)
    xs = (0.7, 1.9) if x.is_positive else (-1.9, -0.7, 0.7, 1.9)
    checked = 0
    for xv in xs:
        for cv in (-3, 0.5, 3):
            try:
                value = complex(expr.subs({x: xv, C: cv}).evalf())
                z = complex(residual.subs({x: xv, C: cv}).evalf())
            except (TypeError, ValueError):
                continue
            if z != z or abs(value.imag) > 1e-12:
                continue
            if abs(z) > 1e-8:
                return False
            checked += 1
    return checked >= 3


def simplify_branches(ode, f, Y, branches):
    """y = ±x·√((C − x)/x) -> y = ±√(C·x − x²), если так короче и ответ остаётся верным."""
    if len(branches) != 2 or sp.simplify(branches[0] + branches[1]) != 0:
        return branches
    root = sp.sqrt(sp.expand(sp.simplify(branches[1] ** 2)))
    size = lambda e: (sp.count_ops(e), len(str(e)))  # noqa: E731
    if size(root) >= size(branches[1]) or not satisfies(ode, f, Y, root):
        return branches
    return [-root, root]


def drop_abs(ode, f, expr, Y):
    """C/|x| -> C/x: знак модуля вне логарифма поглощается постоянной C (если ответ остаётся верным)."""
    x = ode.x
    if x.is_positive or not expr.has(sp.Abs(x)) or any(l.has(sp.Abs) for l in expr.atoms(sp.log)):
        return expr
    candidate = sp.simplify(expr.xreplace({sp.Abs(x): x}))
    residual = sp.diff(candidate, x) - f.subs(Y, candidate)
    checked = 0
    for xv in (-1.9, -0.7, 0.7, 1.9):
        for cv in (-3, 0.5, 3):
            try:
                z = complex(residual.subs({x: xv, C: cv}).evalf())
            except (TypeError, ValueError):
                continue
            if z != z or abs(complex(candidate.subs({x: xv, C: cv}).evalf()).imag) > 1e-12:
                continue
            if abs(z) > 1e-8:
                return expr
            checked += 1
    return candidate if checked >= 3 else expr
