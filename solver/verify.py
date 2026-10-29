"""Проверка ответа: подстановка в уравнение и сравнение с sympy.dsolve."""

from __future__ import annotations

import math
import signal
import threading

import sympy as sp

from solver.model import C, Solution, Verification
from solver.util import invariant

SAMPLE_X = [0.7, 1.3, 2.2, 0.45]
SAMPLE_Y = [0.4, 1.1, 1.9, 0.8]


class Timeout(Exception):
    pass


def with_timeout(seconds, func, *args, **kwargs):
    """Выполняет func с ограничением по времени (на Unix); на Windows — без ограничения."""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        return func(*args, **kwargs)

    def handler(signum, frame):
        raise Timeout

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _num(value):
    try:
        z = complex(value.evalf())
    except (TypeError, ValueError, AttributeError):
        return None
    if math.isnan(z.real) or math.isinf(z.real) or abs(z.imag) > 1e-9:
        return None
    return z.real


def _points(x, ode):
    """Точки x для проверки (внутри области определения)."""
    xs = SAMPLE_X if ode.domain is None or x.is_positive else [-v for v in SAMPLE_X]
    return xs


def check_explicit(ode, g: sp.Eq):
    """Подстановка явного решения. Возвращает (ok, способ)."""
    try:
        res = with_timeout(20, sp.checkodesol, ode.eq, g)
        if res[0]:
            return True, "checkodesol"
    except (Timeout, Exception):  # noqa: BLE001 - checkodesol иногда падает на модулях
        pass
    # Численно: невязка в нескольких точках при нескольких C.
    x, y = ode.x, ode.y
    expr = ode.eq.lhs - ode.eq.rhs
    residual = expr.subs(y, g.rhs).doit()
    good = 0
    for xv in _points(x, ode):
        for cv in (0.3, 1.7):
            r = _num(residual.subs({x: xv, C: cv}))
            if r is None:
                continue
            if abs(r) > 1e-7:
                return False, "численная подстановка"
            good += 1
    return (good >= 2), "численная подстановка"


def check_implicit(ode, f, implicit: sp.Eq):
    """Φ(x, y) = C — решение, если Φx + Φy·f ≡ 0."""
    x, y = ode.x, ode.y
    try:
        res = with_timeout(20, sp.checkodesol, ode.eq, implicit, solve_for_func=False)
        if res[0]:
            return True, "checkodesol (неявное дифференцирование)"
    except (Timeout, Exception):  # noqa: BLE001
        pass
    Y = sp.Symbol("Y_", real=True)
    phi = invariant(implicit, y, Y)
    if phi is None:
        return None, ""
    total = sp.diff(phi, x) + sp.diff(phi, Y) * f.subs(sp.Symbol(ode.func.__name__), Y)
    try:
        if with_timeout(15, sp.simplify, total) == 0:
            return True, "дифференцирование общего интеграла"
    except Timeout:
        pass
    good = 0
    for xv in _points(x, ode):
        for yv in SAMPLE_Y:
            r = _num(total.subs({x: xv, Y: yv}))
            if r is None:
                continue
            if abs(r) > 1e-7:
                return False, "дифференцирование общего интеграла"
            good += 1
    return (good >= 2 or None), "дифференцирование общего интеграла (численно)"


def same_family(ode, ours, theirs):
    """Совпадают ли семейства решений с точностью до переобозначения C.

    Сравниваем «первые интегралы» Φ₁(x, y) и Φ₂(x, y): семейства одинаковы,
    если Φ₁ и Φ₂ функционально зависимы, то есть ∂(Φ₁, Φ₂)/∂(x, y) ≡ 0."""
    x, y = ode.x, ode.y
    Y = sp.Symbol("Y_", real=True)
    phi1 = invariant(ours, y, Y)
    C1 = sp.Symbol("C1")
    expr = (theirs.lhs - theirs.rhs).subs(y, Y)
    try:
        cs = sp.solve(expr, C1, check=False)
    except NotImplementedError:
        cs = []
    if phi1 is None or not cs:
        return None
    phi2 = cs[0]
    d1x, d1y = sp.diff(phi1, x).doit(), sp.diff(phi1, Y).doit()
    d2x, d2y = sp.diff(phi2, x).doit(), sp.diff(phi2, Y).doit()
    J = d1x * d2y - d1y * d2x
    scale = sp.Abs(d1x * d2y) + 1
    good = 0
    for xv in _points(x, ode):
        for yv in SAMPLE_Y:
            j = _num(J.subs({x: xv, Y: yv}))
            s = _num(scale.subs({x: xv, Y: yv}))
            if j is None or s is None:
                continue
            if abs(j) > 1e-7 * s:
                return False
            good += 1
    return True if good >= 2 else None


def run_dsolve(eq):
    """sympy.dsolve; если метод по умолчанию падает — пробуем остальные подходящие методы."""
    try:
        return with_timeout(40, sp.dsolve, eq)
    except (Timeout, Exception):  # noqa: BLE001
        pass
    try:
        hints = [h for h in sp.classify_ode(eq)
                 if not h.endswith("_Integral") and h not in ("default", "all", "best", "1st_homogeneous_coeff_best")]
        hints.sort(key=lambda h: h != "1st_homogeneous_coeff_subs_dep_div_indep")
    except Exception:  # noqa: BLE001
        return None
    for hint in hints:
        try:
            return with_timeout(20, sp.dsolve, eq, hint=hint)
        except (Timeout, Exception):  # noqa: BLE001
            continue
    return None


def verify(sol: Solution, compare=True) -> Verification:
    ode = sol.ode
    v = Verification()
    f = sol.classification.f

    results, methods = [], []
    for g in sol.general + sol.particular:
        if g.has(sp.Order):
            results.append(None)
            methods.append("ряд с остаточным членом подстановкой не проверяется")
            continue
        ok, how = check_explicit(ode, g)
        results.append(ok)
        methods.append(how)
    if not sol.general and sol.implicit is not None and f is not None:
        ok, how = check_implicit(ode, f, sol.implicit)
        results.append(ok)
        methods.append(how)
    if results:
        v.ok = all(r is True for r in results) if all(r is not None for r in results) else None
        v.method = ", ".join(dict.fromkeys(m for m in methods if m))

    if not compare:
        return v
    d = run_dsolve(ode.eq)
    if d is None:
        v.note = "sympy.dsolve не справился с уравнением — сравнить не с чем"
        return v
    v.dsolve = d
    d_list = d if isinstance(d, list) else [d]
    ours = sol.general[0] if sol.general else sol.implicit
    if ours is None:
        return v
    verdicts = []
    for theirs in d_list:
        try:
            verdicts.append(with_timeout(30, same_family, ode, ours, theirs))
        except Timeout:
            verdicts.append(None)
    if any(r is True for r in verdicts):
        v.equivalent = True
    elif all(r is False for r in verdicts):
        v.equivalent = False
    return v

