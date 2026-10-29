"""Уравнения, системы и неравенства с ходом решения."""

import sympy as sp

from calc.text import Log, fmt, fmt_number, fmt_set, format_float, sorted_values
from common import ParseError

NUMERIC_RANGE = range(-20, 21)


# ---------------------------------------------------------------- одно уравнение


def solve_equation(lhs, rhs, var=None):
    g = lhs - rhs
    symbols = sorted(g.free_symbols, key=lambda s: s.name)
    log = Log()
    if not symbols:
        log.title(f"Проверяем равенство {fmt(lhs)} = {fmt(rhs)}")
        ok = sp.simplify(g) == 0
        log.answer("верно" if ok else f"неверно ({fmt(sp.simplify(lhs))} ≠ {fmt(sp.simplify(rhs))})")
        return None
    if var is None:
        names = [s.name for s in symbols]
        var = sp.Symbol("x") if "x" in names else symbols[0]
    log.title(f"Решим уравнение {fmt(lhs)} = {fmt(rhs)}" + (f" относительно {var}" if len(symbols) > 1 else ""))

    if len(symbols) > 1:
        return express(g, var, log)

    if rhs != 0:
        log.step(f"Переносим всё в левую часть: {fmt(g)} = 0")
    roots = solve_zero(g, var, log)
    if roots is None:
        return None
    report_roots(roots, var, g, log)
    return roots


def express(g, var, log):
    sols = sp.solve(g, var)
    if not sols:
        log.answer(f"выразить {var} не удалось")
        return None
    log.step(f"Выражаем {var} через остальные переменные")
    log.answer("; ".join(f"{var} = {fmt(s)}" for s in sols))
    return sols


def solve_zero(g, x, log):
    """Решает g(x) = 0 над вещественными числами. Возвращает множество корней."""
    num, den = split_fraction(g, x)
    domain_excluded = sp.S.EmptySet
    if den.has(x):
        log.step(f"Приводим к общему знаменателю: ({fmt(num)}) / ({fmt(den)}) = 0")
        domain_excluded = sp.solveset(den, x, sp.S.Reals)
        if domain_excluded:
            log.step(f"ОДЗ: {fmt(den)} ≠ 0, то есть {x} ∉ {fmt_set(domain_excluded)}")
        log.step(f"Дробь равна нулю, когда числитель равен нулю: {fmt(num)} = 0")
        g = num

    roots = solve_core(sp.expand(g) if g.is_polynomial(x) else g, x, log)
    if roots is None:
        return None
    if domain_excluded and isinstance(roots, sp.FiniteSet):
        bad = roots & domain_excluded
        if bad:
            log.step(f"Корни {fmt_set(bad)} не входят в ОДЗ — это посторонние корни")
            roots = roots - bad
    return roots


def split_fraction(g, x):
    """g = num/den без автоматического сокращения (чтобы не потерять ОДЗ)."""
    dens = [sp.fraction(sp.together(t))[1] for t in sp.Add.make_args(g)]
    den = sp.Integer(1)
    for d in dens:
        den = sp.lcm(den, d) if (den.is_polynomial(x) and d.is_polynomial(x)) else den * d
    if not den.has(x):
        return sp.fraction(sp.together(g))
    num = sp.expand(sp.cancel(sp.expand(g * den)))
    return num, sp.factor(den)


def solve_core(g, x, log):
    if g.is_polynomial(x):
        poly = sp.Poly(g, x)
        deg = poly.degree()
        if deg <= 0:
            if g == 0:
                log.step("Получили верное равенство 0 = 0 — подходит любое число")
                return sp.S.Reals
            log.step(f"Получили неверное равенство {fmt(g)} = 0")
            return sp.S.EmptySet
        if deg == 1:
            return linear(poly, x, log)
        if deg == 2:
            return quadratic(poly, x, log)
        return by_factors(g, x, log)

    factored = sp.factor(g)
    if factored.is_Mul and sum(1 for f in factored.args if f.has(x)) > 1:
        return by_factors(g, x, log)

    roots = sp.solveset(g, x, sp.S.Reals)
    if isinstance(roots, sp.ConditionSet) or roots.has(sp.ConditionSet):
        return numeric(g, x, log)
    explain_transcendental(g, x, log)
    return roots


def linear(poly, x, log):
    a, b = poly.all_coeffs()
    root = sp.simplify(-b / a)
    log.step(f"Линейное уравнение a·{x} + b = 0, где a = {fmt(a)}, b = {fmt(b)}")
    log.step(f"{x} = -b/a = {fmt(-b)}/{paren(a)} = {fmt(root)}" if a != 1 else f"{x} = {fmt(root)}")
    return sp.FiniteSet(root)


def quadratic(poly, x, log):
    a, b, c = poly.all_coeffs()
    log.step(f"Квадратное уравнение a·{x}² + b·{x} + c = 0, где a = {fmt(a)}, b = {fmt(b)}, c = {fmt(c)}")
    D = sp.simplify(b**2 - 4 * a * c)
    log.step(f"Дискриминант D = b² - 4ac = ({fmt(b)})² - 4·{paren(a)}·{paren(c)} = {fmt(D)}")
    if not D.is_number:
        roots = sp.FiniteSet(*[sp.simplify(r) for r in sp.solve(poly.as_expr(), x)])
        log.step(f"{x} = (-b ± √D)/(2a)")
        return roots
    if D > 0:
        x1 = sp.simplify((-b - sp.sqrt(D)) / (2 * a))
        x2 = sp.simplify((-b + sp.sqrt(D)) / (2 * a))
        log.step(f"D > 0 — два корня: {x}₁,₂ = (-b ± √D)/(2a)")
        with log.sub():
            log.step(f"{x}₁ = ({fmt(-b)} - √{fmt(D)})/{paren(2 * a)} = {fmt(x1)}")
            log.step(f"{x}₂ = ({fmt(-b)} + √{fmt(D)})/{paren(2 * a)} = {fmt(x2)}")
        return sp.FiniteSet(x1, x2)
    if D == 0:
        x0 = sp.simplify(-b / (2 * a))
        log.step(f"D = 0 — один корень: {x} = -b/(2a) = {fmt(x0)}")
        return sp.FiniteSet(x0)
    log.step("D < 0 — вещественных корней нет")
    with log.sub():
        z1 = sp.simplify((-b - sp.I * sp.sqrt(-D)) / (2 * a))
        z2 = sp.simplify((-b + sp.I * sp.sqrt(-D)) / (2 * a))
        log.step(f"Комплексные корни: {x} = {fmt(z1)},  {x} = {fmt(z2)}")
    return sp.S.EmptySet


def paren(e):
    return f"({fmt(e)})" if (e.is_number and e < 0) or e.is_Add else fmt(e)


def by_factors(g, x, log):
    factored = sp.factor(g)
    factors = [f for f in sp.Mul.make_args(factored) if f.has(x)]
    if len(factors) <= 1 and not (factors and factors[0].is_Pow):
        if g.is_polynomial(x):
            log.step(f"Многочлен степени {sp.degree(g, x)} не раскладывается на простые множители")
            roots = sp.solveset(g, x, sp.S.Reals)
            if isinstance(roots, sp.FiniteSet) and all(not r.has(sp.CRootOf) for r in roots):
                log.step("Находим корни по общим формулам")
                return roots
            return numeric(g, x, log)
        return numeric(g, x, log)

    log.step(f"Раскладываем на множители: {fmt(factored)} = 0")
    log.step("Произведение равно нулю, когда хотя бы один из множителей равен нулю")
    roots = sp.S.EmptySet
    with log.sub():
        for f in factors:
            base = f.base if f.is_Pow and f.exp.is_positive else f
            log.step(f"{fmt(base)} = 0")
            with log.sub():
                part = solve_core(sp.expand(base) if base.is_polynomial(x) else base, x, log)
            if part is None:
                return None
            roots = roots | part
    return roots


def explain_transcendental(g, x, log):
    const, h = g.as_independent(x, as_Add=True)
    a = -const
    if isinstance(h, (sp.sin, sp.cos, sp.tan, sp.cot)) and a.is_real:
        u = h.args[0]
        U = fmt(u)
        if isinstance(h, sp.sin):
            if abs(a) > 1:
                log.step(f"sin({U}) = {fmt(a)}, но |sin| ≤ 1 — решений нет")
                return
            log.step(f"sin({U}) = {fmt(a)}  ⇒  {U} = arcsin({fmt(a)}) + 2πk  или  {U} = π - arcsin({fmt(a)}) + 2πk")
            log.step(f"arcsin({fmt(a)}) = {fmt(sp.asin(a))}")
        elif isinstance(h, sp.cos):
            if abs(a) > 1:
                log.step(f"cos({U}) = {fmt(a)}, но |cos| ≤ 1 — решений нет")
                return
            log.step(f"cos({U}) = {fmt(a)}  ⇒  {U} = ±arccos({fmt(a)}) + 2πk")
            log.step(f"arccos({fmt(a)}) = {fmt(sp.acos(a))}")
        elif isinstance(h, sp.tan):
            log.step(f"tg({U}) = {fmt(a)}  ⇒  {U} = arctg({fmt(a)}) + πk,  arctg({fmt(a)}) = {fmt(sp.atan(a))}")
        else:
            log.step(f"ctg({U}) = {fmt(a)}  ⇒  {U} = arcctg({fmt(a)}) + πk,  arcctg({fmt(a)}) = {fmt(sp.acot(a))}")
        log.note("k — любое целое число")
        return
    if isinstance(h, sp.log) and a.is_real:
        log.step(f"ln({fmt(h.args[0])}) = {fmt(a)}  ⇒  {fmt(h.args[0])} = e^{paren(a)} (по определению логарифма)")
        return
    if h.is_Pow and not h.base.has(x) and h.exp.has(x) and a.is_positive:
        log.step(f"{fmt(h)} = {fmt(a)}  ⇒  {fmt(h.exp)} = log по основанию {fmt(h.base)} от {fmt(a)} = {fmt(sp.simplify(sp.log(a) / sp.log(h.base)))}")
        return
    if isinstance(h, sp.exp) and a.is_positive:
        log.step(f"{fmt(h)} = {fmt(a)}  ⇒  {fmt(h.args[0])} = ln({fmt(a)})")
        return
    funcs = {type(f) for f in g.atoms(sp.Function)}
    if funcs & {sp.sin, sp.cos, sp.tan, sp.cot}:
        log.step("Тригонометрическое уравнение: используем формулы для sin u = a, cos u = a, tg u = a "
                 "(с учётом периода, k — любое целое)")
    elif g.has(sp.exp) or any(p.is_Pow and p.exp.has(x) for p in g.atoms(sp.Pow)):
        log.step("Показательное уравнение: логарифмируем обе части")
    elif g.has(sp.log):
        log.step("Логарифмическое уравнение: потенцируем, учитывая ОДЗ (аргумент логарифма > 0)")
    elif any(p.is_Pow and not p.exp.is_Integer for p in g.atoms(sp.Pow)):
        log.step("Иррациональное уравнение: уединяем корень и возводим в степень, затем проверяем корни")
    else:
        log.step("Решаем уравнение")


def numeric(g, x, log):
    log.step(f"Точно решить не удаётся — ищем корни численно (метод Ньютона, отрезок [{NUMERIC_RANGE.start}; {NUMERIC_RANGE.stop - 1}])")
    found = []
    for guess in NUMERIC_RANGE:
        try:
            r = sp.nsolve(g, x, guess)
        except Exception:  # noqa: BLE001 - из этой точки метод не сошёлся
            continue
        if r.is_real and all(abs(r - other) > 1e-9 for other in found):
            found.append(sp.Float(r, 15))
    return sp.FiniteSet(*found)


def report_roots(roots, x, g, log):
    if isinstance(roots, sp.FiniteSet) and roots:
        if 0 < len(roots) <= 6:
            checks = []
            for r in sorted_values(roots):
                value = sp.N(g.subs(x, r), 30)
                checks.append(abs(value) < 1e-9)
            if all(checks):
                log.step("Проверка подстановкой: все корни обращают уравнение в верное равенство ✓")
        log.answer("; ".join(f"{x} {fmt_root(r)}" for r in sorted_values(roots)))
    else:
        log.answer(fmt_set(roots, str(x)))


def fmt_root(r):
    if isinstance(r, sp.Float):
        return "≈ " + format_float(r)[:14]
    return "= " + fmt_number(r)


# ---------------------------------------------------------------- системы


def solve_system(equations):
    """equations — список выражений вида «левая - правая» (= 0)."""
    symbols = sorted(set().union(*(e.free_symbols for e in equations)), key=lambda s: s.name)
    if not symbols:
        raise ParseError("в системе нет неизвестных")
    if all(is_linear(e, symbols) for e in equations):
        import linear_solver

        print("\n▶ Линейная система — решаем методом Гаусса")
        linear_solver.solve_system(equations)
        return None

    log = Log()
    log.title("Решим систему: " + ";  ".join(f"{fmt(e)} = 0" for e in equations))
    solutions = substitution(equations, symbols, log)
    if solutions is None:
        log.step("Решаем систему (SymPy)")
        solutions = sp.solve(equations, symbols, dict=True)
    real = [s for s in solutions if all(v.is_real is not False for v in s.values())]
    if len(real) < len(solutions):
        log.note("комплексные решения отброшены")
    if not real:
        log.answer("решений нет")
        return None
    ok = all(all(sp.simplify(e.subs(s)) == 0 for e in equations) for s in real)
    if ok:
        log.step("Проверка подстановкой во все уравнения ✓")
    lines = []
    for s in real:
        lines.append("(" + ", ".join(f"{v} = {fmt_number(s[v]) if v in s else 'любое'}" for v in symbols) + ")")
    log.answer("  ".join(lines))
    return real


def is_linear(e, symbols):
    try:
        return sp.Poly(sp.expand(e), *symbols).total_degree() <= 1
    except sp.PolynomialError:
        return False


def substitution(equations, symbols, log):
    """Метод подстановки для системы из двух уравнений с двумя неизвестными."""
    if len(equations) != 2 or len(symbols) != 2:
        return None
    for i, eq in enumerate(equations):
        for v in symbols:
            coeff = sp.expand(eq).coeff(v, 1)
            rest = sp.expand(eq - coeff * v)
            if coeff == 0 or coeff.has(v) or rest.has(v) or coeff.free_symbols:
                continue
            expr_v = sp.simplify(-rest / coeff)
            other_eq = equations[1 - i]
            other_var = [s for s in symbols if s != v][0]
            log.step(f"Из уравнения {i + 1} выражаем {v} = {fmt(expr_v)}")
            substituted = sp.simplify(other_eq.subs(v, expr_v))
            log.step(f"Подставляем в уравнение {2 - i}: {fmt(substituted)} = 0")
            with log.sub():
                roots = solve_zero(substituted, other_var, log)
            if roots is None or not isinstance(roots, sp.FiniteSet):
                return None
            log.step(f"Находим {v} = {fmt(expr_v)} для каждого найденного {other_var}:")
            solutions = []
            with log.sub():
                for r in sorted_values(roots):
                    value = sp.simplify(expr_v.subs(other_var, r))
                    log.step(f"{other_var} = {fmt(r)}  →  {v} = {fmt(value)}")
                    solutions.append({other_var: r, v: value})
            return solutions
    return None


# ---------------------------------------------------------------- неравенства


def solve_inequality(rel):
    symbols = sorted(rel.free_symbols, key=lambda s: s.name)
    log = Log()
    if len(symbols) != 1:
        raise ParseError("неравенство должно содержать ровно одну переменную")
    x = symbols[0]
    op = rel.rel_op
    g = sp.together(rel.lhs - rel.rhs)
    log.title(f"Решим неравенство {fmt(rel.lhs)} {pretty_op(op)} {fmt(rel.rhs)}")
    truth = sp.solveset(rel, x, sp.S.Reals)
    if rel.rhs != 0:
        log.step(f"Переносим всё в левую часть: {fmt(g)} {pretty_op(op)} 0")

    num, den = sp.fraction(g)
    zeros = sp.solveset(num, x, sp.S.Reals)
    poles = sp.solveset(den, x, sp.S.Reals) if den.has(x) else sp.S.EmptySet
    finite = (sp.FiniteSet, type(sp.S.EmptySet))
    if isinstance(zeros, finite) and isinstance(poles, finite) and 0 < len(zeros | poles) <= 12:
        log.step("Метод интервалов")
        with log.sub():
            log.step(f"Нули числителя: {fmt_set(zeros) if zeros else 'нет'}")
            if den.has(x):
                log.step(f"Нули знаменателя (выколотые точки): {fmt_set(poles) if poles else 'нет'}")
            points = sorted_values(zeros | poles)
            edges = [-sp.oo, *points, sp.oo]
            signs = []
            for lo, hi in zip(edges, edges[1:]):
                test = test_point(lo, hi)
                value = g.subs(x, test)
                sign = "+" if value > 0 else "-"
                signs.append(f"{fmt_set(sp.Interval.open(lo, hi))}: {sign}")
            log.step("Знаки на интервалах:  " + ",  ".join(signs))
    log.answer(f"{x} ∈ {fmt_set(truth)}" if truth not in (sp.S.EmptySet, sp.S.Reals) else fmt_set(truth))
    return truth


def test_point(lo, hi):
    if lo == -sp.oo and hi == sp.oo:
        return sp.Integer(0)
    if lo == -sp.oo:
        return sp.floor(hi) - 1
    if hi == sp.oo:
        return sp.ceiling(lo) + 1
    return (lo + hi) / 2


def pretty_op(op):
    return {"<=": "≤", ">=": "≥", "!=": "≠"}.get(op, op)
