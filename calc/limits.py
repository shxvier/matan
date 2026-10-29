"""Пределы с ходом решения: подстановка, сокращение, деление на старшую степень, Лопиталь."""

import sympy as sp

from calc.text import Log, fmt, fmt_number

MAX_LHOPITAL = 4


def arrow(x, a, direction):
    side = {"+": "+0", "-": "-0"}.get(direction, "")
    point = "+∞" if a == sp.oo else fmt(a)
    return f"{x}→{point}{side}" if side and a.is_finite else f"{x}→{point}"


def lim(expr, x, a, direction):
    return sp.limit(expr, x, a, direction) if a.is_finite else sp.limit(expr, x, a)


def form_of(num_lim, den_lim):
    if num_lim == 0 and den_lim == 0:
        return "0/0"
    if num_lim.is_infinite and den_lim.is_infinite:
        return "∞/∞"
    return None


def limit(f, x, a, direction="+-"):
    a = sp.sympify(a)
    log = Log()
    log.title(f"Найдём lim[{arrow(x, a, direction)}] {fmt(f)}")

    # Двусторонний предел: если односторонние различаются — предела нет.
    if a.is_finite and direction == "+-":
        left, right = sp.limit(f, x, a, "-"), sp.limit(f, x, a, "+")
        if left != right:
            log.step("Односторонние пределы различаются:")
            with log.sub():
                log.step(f"слева  (x→{fmt(a)}-0): {fmt(left)}")
                log.step(f"справа (x→{fmt(a)}+0): {fmt(right)}")
            log.answer("предел не существует")
            return None
        truth = right
    else:
        truth = lim(f, x, a, direction)

    explained = explain(f, x, a, direction, log, truth)
    if not explained:
        log.step(f"Вычисляем предел: {fmt(truth)}")
    log.answer(fmt_number(truth) if truth.is_finite else fmt(truth))
    return truth


def explain(f, x, a, direction, log, truth):
    """Печатает ход решения. Возвращает True, если по шагам пришли к ответу."""
    # 1. Прямая подстановка
    if a.is_finite:
        value = sp.simplify(f.subs(x, a))
        if value.is_finite and value == truth:
            log.step(f"Функция непрерывна в точке {x} = {fmt(a)}, просто подставляем: f({fmt(a)}) = {fmt(value)}")
            return True

    # 1^∞: lim u^v = e^(lim (u - 1)·v)
    if f.is_Pow and f.exp.has(x):
        u, v = f.args
        if lim(u, x, a, direction) == 1 and lim(v, x, a, direction).is_infinite:
            log.step("Неопределённость вида 1^∞ — второй замечательный предел: lim uᵛ = e^(lim (u - 1)·v)")
            inner = sp.simplify((u - 1) * v)
            L = lim(inner, x, a, direction)
            log.step(f"(u - 1)·v = ({fmt(sp.expand(u - 1))})·({fmt(v)}) = {fmt(inner)} → {fmt(L)}")
            log.step(f"Значит, предел равен e^{fmt(L) if L.is_Atom else '(' + fmt(L) + ')'}")
            return sp.exp(L) == truth

    num, den = sp.fraction(sp.together(f))
    if not den.has(x):
        if a.is_infinite and f.is_polynomial(x):
            lead = sp.LT(sp.expand(f), x)
            log.step(f"Многочлен ведёт себя как старший член {fmt(lead)} при {arrow(x, a, direction)}")
            return False
        return False

    for attempt in range(MAX_LHOPITAL + 1):
        ln, ld = lim(num, x, a, direction), lim(den, x, a, direction)
        log.step(f"Числитель {fmt(num)} → {fmt(ln)},  знаменатель {fmt(den)} → {fmt(ld)}")
        form = form_of(ln, ld)
        if form is None:
            if ld == 0 and ln != 0:
                log.step(f"Число {fmt(ln)}, делённое на бесконечно малую, — бесконечность")
            elif ld.is_infinite and ln.is_finite:
                log.step("Ограниченное, делённое на бесконечно большую, стремится к 0")
            elif ln.is_finite and ld.is_finite:
                log.step(f"Предел частного равен частному пределов: {fmt(ln)}/{fmt(ld)} = {fmt(sp.simplify(ln / ld))}")
            return False
        if attempt == MAX_LHOPITAL:
            return False
        log.step(f"Неопределённость вида {form}")

        g = sp.cancel(num / den)
        if num.is_polynomial(x) and den.is_polynomial(x) and a.is_finite and g != num / den:
            log.step(
                f"Раскладываем на множители: {fmt(sp.factor(num))} / ({fmt(sp.factor(den))}) "
                f"и сокращаем: {fmt(g)}"
            )
            value = sp.simplify(g.subs(x, a))
            if value.is_finite:
                log.step(f"Подставляем {x} = {fmt(a)}: {fmt(value)}")
                return value == truth
            num, den = sp.fraction(g)
            continue

        if a.is_infinite and num.is_polynomial(x) and den.is_polynomial(x):
            k = max(sp.degree(num, x), sp.degree(den, x))
            power = x**k
            n2, d2 = sp.expand(num / power), sp.expand(den / power)
            log.step(f"Делим числитель и знаменатель на старшую степень {fmt(power)}: ({fmt(n2)}) / ({fmt(d2)})")
            log.step(f"Слагаемые вида c/{x}ᵏ стремятся к 0, остаётся {fmt(lim(n2, x, a, direction))}/{fmt(lim(d2, x, a, direction))}")
            return True

        dn, dd = sp.diff(num, x), sp.diff(den, x)
        log.step(f"Правило Лопиталя: lim f/g = lim f'/g',  f' = {fmt(dn)},  g' = {fmt(dd)}")
        ratio = sp.simplify(dn / dd)
        if a.is_finite:
            value = sp.simplify(ratio.subs(x, a))
            if value.is_finite and value == truth:
                shown = f"{fmt(dn.subs(x, a))}/{fmt(dd.subs(x, a))}"
                tail = "" if shown == fmt(value) else f" = {fmt(value)}"
                log.step(f"Подставляем {x} = {fmt(a)}: {shown}{tail}")
                return True
        num, den = sp.fraction(sp.together(ratio))
        if not den.has(x):
            value = lim(ratio, x, a, direction)
            log.step(f"lim {fmt(ratio)} = {fmt(value)}")
            return value == truth
    return False
