"""Линейные уравнения первого порядка: y' + p(x)·y = q(x) — метод интегрирующего множителя."""

from __future__ import annotations

import sympy as sp

from solver.classify import linear_parts
from solver.model import C, Solution
from solver.util import abs_logs, integrate, tidy_constant


def solve_linear(sol: Solution, f, Y, coeff):
    ode = sol.ode
    x, y = ode.x, ode.y
    p, q = linear_parts(f, Y)
    yp = sp.Symbol(f"{Y.name}'")

    if coeff != 1:
        sol.add(f"Делим уравнение на коэффициент при y': {sp.sstr(coeff)} (считаем его ≠ 0)",
                sp.Eq(sp.Add(yp, p * Y, evaluate=False), q))
    else:
        sol.add("Записываем уравнение в виде y' + p(x)·y = q(x)", sp.Eq(sp.Add(yp, p * Y, evaluate=False), q))
    sol.add("Коэффициенты", [sp.Eq(sp.Symbol(f"p({x})"), p), sp.Eq(sp.Symbol(f"q({x})"), q)])

    P, _ = integrate(p, x)
    P = abs_logs(P, only_var=x)
    sol.add("Первообразная p(x) (константу не пишем — нужен любой множитель)",
            sp.Eq(sp.Integral(p, x), P))
    mu = sp.simplify(sp.exp(P))
    # Знак модуля для μ не важен: e^(ln|x|) можно взять как x.
    mu_plain = sp.simplify(mu.replace(sp.Abs, lambda a: a))
    note = None
    if mu_plain != mu:
        note = "модуль можно опустить: множитель достаточно найти с точностью до знака"
        mu = mu_plain
    mu_sym = sp.Symbol("μ")
    sol.add("Интегрирующий множитель μ = e^(∫p dx)", sp.Eq(mu_sym, mu), note)

    sol.add("Умножаем уравнение на μ: левая часть становится производной произведения (μ·y)' = μ·q",
            sp.Eq(sp.Derivative(mu * Y, x), sp.simplify(mu * q)))

    integrand = sp.simplify(mu * q)
    I, elementary = integrate(integrand, x)
    note = None
    if not elementary:
        sol.nonelementary = True
        note = "этот интеграл не выражается в элементарных функциях — оставляем его в виде ∫"
        sol.notes.append(note)
    I = abs_logs(I, only_var=x) if elementary else I
    sol.add("Интегрируем: μ·y = ∫ μ·q dx + C", sp.Eq(mu * Y, I + C), note)

    general = sp.expand(sp.simplify((I + C) / mu)) if elementary else (I + C) / mu
    if elementary:
        general = tidy_constant(general)
        factored = sp.collect(general, C)
        general = min([general, factored, sp.factor_terms(general)], key=sp.count_ops)
    sol.general = [sp.Eq(y, general)]
    sol.implicit = None
    sol.add("Делим на μ — общее решение", sol.general[0])
    return mu, integrand, elementary


def apply_ic_nonelementary(sol: Solution, mu, integrand, x0, y0):
    """Задача Коши, когда ∫μq dx не берётся: y = (μ(x0)·y0 + ∫ от x0 до x μq dt) / μ."""
    ode = sol.ode
    x, y = ode.x, ode.y
    t = sp.Symbol("t", real=True)
    sol.ic = (x0, y0)
    sol.add(f"Подставляем начальное условие {ode.func.__name__}({x0}) = {y0}: "
            "неопределённый интеграл заменяем на интеграл с переменным верхним пределом от x0 до x")
    c = sp.simplify(mu.subs(x, x0) * y0)
    sol.add("При x = x0 интеграл равен нулю, поэтому C = μ(x0)·y0", sp.Eq(C, c))
    part = sp.Eq(y, (c + sp.Integral(integrand.subs(x, t), (t, x0, x))) / mu)
    sol.particular = [part]
    sol.add("Частное решение", part)
