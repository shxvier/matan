"""Производная с подробным ходом решения (правила дифференцирования)."""

import sympy as sp

from calc.text import Log, fmt

# Таблица производных: функция -> (формула для вывода, f'(u))
TABLE = {
    sp.sin: ("(sin u)' = cos u", lambda u: sp.cos(u)),
    sp.cos: ("(cos u)' = -sin u", lambda u: -sp.sin(u)),
    sp.tan: ("(tg u)' = 1/cos²u", lambda u: 1 / sp.cos(u) ** 2),
    sp.cot: ("(ctg u)' = -1/sin²u", lambda u: -1 / sp.sin(u) ** 2),
    sp.exp: ("(e^u)' = e^u", lambda u: sp.exp(u)),
    sp.log: ("(ln u)' = 1/u", lambda u: 1 / u),
    sp.asin: ("(arcsin u)' = 1/√(1 - u²)", lambda u: 1 / sp.sqrt(1 - u**2)),
    sp.acos: ("(arccos u)' = -1/√(1 - u²)", lambda u: -1 / sp.sqrt(1 - u**2)),
    sp.atan: ("(arctg u)' = 1/(1 + u²)", lambda u: 1 / (1 + u**2)),
    sp.acot: ("(arcctg u)' = -1/(1 + u²)", lambda u: -1 / (1 + u**2)),
    sp.sinh: ("(sh u)' = ch u", lambda u: sp.cosh(u)),
    sp.cosh: ("(ch u)' = sh u", lambda u: sp.sinh(u)),
    sp.tanh: ("(th u)' = 1/ch²u", lambda u: 1 / sp.cosh(u) ** 2),
}


class Differentiator:
    def __init__(self, x, log):
        self.x = x
        self.log = log

    def d(self, f):
        x, log = self.x, self.log
        if not f.has(x):
            return sp.Integer(0)
        if f == x:
            return sp.Integer(1)

        if f.is_Add:
            log.step(f"({fmt(f)})' — производная суммы равна сумме производных")
            with log.sub():
                parts = [self.d(t) for t in f.args]
            return self.done(sp.Add(*parts))

        if f.is_Mul:
            const, rest = f.as_independent(x, as_Add=False)
            if const != 1:
                log.step(f"({fmt(f)})' = {fmt(const)}·({fmt(rest)})' — константу выносим за знак производной")
                with log.sub():
                    inner = self.d(rest)
                return self.done(const * inner)
            num, den = sp.fraction(f)
            if den != 1 and den.has(x):
                log.step(
                    f"({fmt(f)})' — производная частного: (u/v)' = (u'·v - u·v')/v², "
                    f"u = {fmt(num)}, v = {fmt(den)}"
                )
                with log.sub():
                    du, dv = self.d(num), self.d(den)
                    log.step(f"u' = {fmt(du)},  v' = {fmt(dv)}")
                return self.done((du * den - num * dv) / den**2)
            u, v = f.args[0], sp.Mul(*f.args[1:])
            log.step(
                f"({fmt(f)})' — производная произведения: (u·v)' = u'·v + u·v', u = {fmt(u)}, v = {fmt(v)}"
            )
            with log.sub():
                du, dv = self.d(u), self.d(v)
                log.step(f"u' = {fmt(du)},  v' = {fmt(dv)}")
            return self.done(du * v + u * dv)

        if f.is_Pow:
            base, e = f.args
            if not e.has(x):
                result = e * base ** (e - 1)
                if base == x:
                    log.step(f"({fmt(f)})' = {fmt(result)} — степенная функция: (xⁿ)' = n·xⁿ⁻¹")
                    return result
                log.step(f"({fmt(f)})' — сложная функция: (uⁿ)' = n·uⁿ⁻¹·u', u = {fmt(base)}, n = {fmt(e)}")
                with log.sub():
                    du = self.d(base)
                return self.done(result * du)
            if not base.has(x):
                outer = f * sp.log(base)
                if e == x:
                    log.step(f"({fmt(f)})' = {fmt(outer)} — показательная функция: (aˣ)' = aˣ·ln a")
                    return outer
                log.step(f"({fmt(f)})' — сложная функция: (aᵘ)' = aᵘ·ln a·u', u = {fmt(e)}")
                with log.sub():
                    du = self.d(e)
                return self.done(outer * du)
            log.step(
                f"({fmt(f)})' — степенно-показательная функция: (uᵛ)' = uᵛ·(v'·ln u + v·u'/u), "
                f"u = {fmt(base)}, v = {fmt(e)}"
            )
            with log.sub():
                du, dv = self.d(base), self.d(e)
            return self.done(f * (dv * sp.log(base) + e * du / base))

        if isinstance(f, sp.Function) and f.func in TABLE and len(f.args) == 1:
            u = f.args[0]
            formula, rule = TABLE[f.func]
            outer = rule(u)
            if u == x:
                log.step(f"({fmt(f)})' = {fmt(outer)} — табличная производная {formula}")
                return outer
            log.step(f"({fmt(f)})' — сложная функция: {formula}·u', u = {fmt(u)}")
            with log.sub():
                du = self.d(u)
            return self.done(outer * du)

        result = sp.diff(f, x)
        log.step(f"({fmt(f)})' = {fmt(result)}")
        return result

    def done(self, result):
        self.log.line(f"= {fmt(result)}")
        return result


def nice(expr):
    """Самая короткая из нескольких упрощённых форм."""
    candidates = [expr, sp.together(expr), sp.simplify(expr), sp.factor(expr), sp.expand(expr)]
    best = min(candidates, key=sp.count_ops)
    # Переписываем ответ, только если это заметно короче исходного.
    return best if sp.count_ops(best) <= sp.count_ops(expr) - 2 else expr


def derivative(f, x, order=1):
    log = Log()
    title = "Найдём производную" if order == 1 else f"Найдём производную {order}-го порядка"
    log.title(f"{title} функции  f({x}) = {fmt(f)}  по {x}")
    current = f
    for k in range(1, order + 1):
        if order > 1:
            print(f"\n— Производная {k}-го порядка —")
            log.counter = 0
        raw = Differentiator(x, log).d(current)
        truth = sp.diff(current, x)
        if sp.simplify(raw - truth) != 0:
            log.note("результат уточнён с помощью SymPy")
            raw = truth
        result = nice(raw)
        if fmt(result) != fmt(raw):
            log.step(f"Упрощаем: {fmt(result)}")
        current = result

    strokes = "'" * order if order <= 3 else f"^({order})"
    log.answer_block(f"f{strokes}({x}) =", current)
    return current
