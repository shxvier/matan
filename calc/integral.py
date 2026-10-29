"""Интегралы с ходом решения (по дереву правил sympy.manualintegrate)."""

import sympy as sp
from sympy.integrals import manualintegrate as mi

from calc.derivative import nice
from calc.text import Log, fmt, fmt_number

TABLE_FORMULAS = {
    "ExpRule": "∫ aᵘ du = aᵘ/ln a  (∫ eᵘ du = eᵘ)",
    "ReciprocalRule": "∫ du/u = ln|u|",
    "SinRule": "∫ sin u du = -cos u",
    "CosRule": "∫ cos u du = sin u",
    "Sec2Rule": "∫ du/cos²u = tg u",
    "Csc2Rule": "∫ du/sin²u = -ctg u",
    "SecTanRule": "∫ sec u·tg u du = sec u",
    "CscCotRule": "∫ csc u·ctg u du = -csc u",
    "ArctanRule": "∫ du/(a² + u²) = (1/a)·arctg(u/a)",
    "ArcsinRule": "∫ du/√(1 - u²) = arcsin u",
    "ArcsinhRule": "∫ du/√(1 + u²) = ln(u + √(u² + 1))",
    "SinhRule": "∫ sh u du = ch u",
    "CoshRule": "∫ ch u du = sh u",
}


class NoSteps(Exception):
    pass


def integral_text(f, v):
    body = f"({fmt(f)})" if isinstance(f, sp.Add) else fmt(f)
    return f"∫ {body} d{fmt(v)}"


def with_abs(F):
    """ln(u) -> ln|u| в ответе (если u может быть отрицательным)."""
    return F.replace(sp.log, lambda a: sp.log(a if a.is_positive or isinstance(a, sp.Abs) else sp.Abs(a)))


class Renderer:
    def __init__(self, log):
        self.log = log
        self.names = {}  # служебные переменные замены -> t, s, ...

    def render(self, rule):
        log = self.log
        name = type(rule).__name__
        result = rule.eval()
        f, v = rule.integrand, rule.variable
        if f is not None:
            f, v = f.xreplace(self.names), v.xreplace(self.names)
        result = result.xreplace(self.names)

        if name == "DontKnowRule":
            raise NoSteps
        if name == "AlternativeRule":
            return self.render(rule.alternatives[0])
        if name == "ConstantRule":
            log.step(f"{integral_text(f, v)} = {fmt(result)} — интеграл константы: ∫ c dx = c·x")
        elif name == "PowerRule":
            log.step(f"{integral_text(f, v)} = {fmt(result)} — степенная функция: ∫ xⁿ dx = xⁿ⁺¹/(n + 1)")
        elif name == "ConstantTimesRule":
            log.step(f"{integral_text(f, v)} = {fmt(rule.constant)}·{integral_text(rule.other.xreplace(self.names), v)} — выносим константу")
            with log.sub():
                self.render(rule.substep)
            log.line(f"= {fmt(result)}")
        elif name == "AddRule":
            log.step(f"{integral_text(f, v)} — интеграл суммы равен сумме интегралов")
            with log.sub():
                for sub in rule.substeps:
                    self.render(sub)
            log.line(f"= {fmt(result)}")
        elif name == "URule":
            t = sp.Symbol("tsuvw"[len(self.names) % 5])
            self.names[rule.u_var] = t
            u_func = rule.u_func.xreplace(self.names)
            du = sp.diff(u_func, v)
            log.step(
                f"{integral_text(f, v)} — замена переменной: {t} = {fmt(u_func)}, d{t} = {fmt(du)} d{fmt(v)}"
            )
            with log.sub():
                self.render(rule.substep)
                log.step(f"Обратная замена {t} = {fmt(u_func)}")
            log.line(f"= {fmt(result)}")
        elif name == "PartsRule":
            u_part, dv_part = rule.u.xreplace(self.names), rule.dv.xreplace(self.names)
            du = sp.diff(u_part, v)
            vv = rule.v_step.eval().xreplace(self.names)
            log.step(
                f"{integral_text(f, v)} — интегрирование по частям: ∫ u dv = u·v - ∫ v du, "
                f"u = {fmt(u_part)}, dv = {fmt(dv_part)} d{fmt(v)}"
            )
            with log.sub():
                log.step(f"du = {fmt(du)} d{fmt(v)},  v = ∫ dv = {fmt(vv)}")
                if rule.second_step is not None:
                    log.step(f"Остаётся {integral_text(rule.second_step.integrand.xreplace(self.names), v)}")
                    with log.sub():
                        self.render(rule.second_step)
            log.line(f"= {fmt(result)}")
        elif name == "CyclicPartsRule":
            log.step(
                f"{integral_text(f, v)} — дважды по частям; снова получаем исходный интеграл I "
                "и выражаем I из уравнения"
            )
            log.line(f"I = {fmt(result)}")
        elif name == "RewriteRule":
            log.step(f"Преобразуем подынтегральное выражение: {fmt(f)} = {fmt(rule.rewritten.xreplace(self.names))}")
            with log.sub():
                self.render(rule.substep)
        elif name == "CompleteSquareRule":
            log.step(f"Выделяем полный квадрат: {fmt(f)} = {fmt(rule.rewritten.xreplace(self.names))}")
            with log.sub():
                self.render(rule.substep)
        elif name in TABLE_FORMULAS:
            log.step(f"{integral_text(f, v)} = {fmt(result)} — табличный интеграл {TABLE_FORMULAS[name]}")
        else:
            log.step(f"{integral_text(f, v)} = {fmt(result)}")
        return result


def antiderivative(f, x, log):
    """Первообразная с выводом шагов; при неудаче — без шагов через SymPy."""
    try:
        rule = mi.integral_steps(f, x)
        F = Renderer(log).render(rule)
        if F.has(sp.Integral):
            raise NoSteps
    except NoSteps:
        log.note("подробные шаги для этого интеграла недоступны, считаем с помощью SymPy")
        F = sp.integrate(f, x)
        if F.has(sp.Integral):
            return None
        log.step(f"{integral_text(f, x)} = {fmt(F)}")
    return nice(F)


def indefinite(f, x):
    log = Log()
    log.title(f"Найдём неопределённый интеграл  {integral_text(f, x)}")
    F = antiderivative(f, x, log)
    if F is None:
        log.answer("интеграл не выражается в элементарных функциях (или SymPy не смог его найти)")
        return None
    check = sp.simplify(sp.diff(F, x) - f) == 0
    log.step(f"Проверка дифференцированием: ({fmt(F)})' = {fmt(f)}" + ("  ✓" if check else ""))
    log.answer(f"{integral_text(f, x)} = {fmt(with_abs(F))} + C")
    return F


def definite(f, x, a, b):
    a, b = sp.sympify(a), sp.sympify(b)
    log = Log()
    log.title(f"Найдём определённый интеграл  ∫ от {fmt(a)} до {fmt(b)}  {fmt(f)} d{x}")
    truth = sp.integrate(f, (x, a, b))

    lo, hi = (a, b) if (a.is_comparable and b.is_comparable and a <= b) else (b, a)
    try:
        singular = sp.singularities(f, x, sp.Interval(lo, hi))
    except Exception:  # noqa: BLE001
        singular = sp.S.EmptySet
    improper = a.is_infinite or b.is_infinite or (isinstance(singular, sp.FiniteSet) and len(singular) > 0)

    F = antiderivative(f, x, log)
    if F is None:
        log.note("первообразная не выражается в элементарных функциях")
        show_result(log, truth)
        return truth

    inner = [p for p in singular if lo < p < hi] if isinstance(singular, sp.FiniteSet) else []
    if inner:
        log.step(f"Функция разрывна внутри отрезка в точках {', '.join(fmt(p) for p in inner)} — "
                 "интеграл несобственный, разбиваем его на части")
        edges = [lo, *sorted(inner), hi]
        with log.sub():
            Fabs = with_abs(F)
            for left, right in zip(edges, edges[1:]):
                part = sp.simplify(value_at(Fabs, x, right, "-") - value_at(Fabs, x, left, "+"))
                log.step(f"∫ от {fmt(left)} до {fmt(right)} = {fmt(part)}")
                if not part.is_finite:
                    log.step("Эта часть расходится — значит, расходится и весь интеграл")
                    truth = sp.oo if part.is_infinite else sp.nan
                    break
    else:
        Fabs = with_abs(F)
        if improper:
            reasons = []
            if a.is_infinite or b.is_infinite:
                reasons.append("бесконечный предел интегрирования")
            if singular:
                reasons.append(f"функция не ограничена в точке {', '.join(fmt(p) for p in singular)}")
            log.step(f"Интеграл несобственный ({'; '.join(reasons)}) — значения F на концах находим как пределы")
        log.step(f"Формула Ньютона–Лейбница: ∫ₐᵇ f dx = F(b) - F(a),  F({x}) = {fmt(Fabs)}")
        Fb = value_at(Fabs, x, b, "-")
        Fa = value_at(Fabs, x, a, "+")
        with log.sub():
            log.step(f"F({fmt(b)}) = {fmt(Fb)}")
            log.step(f"F({fmt(a)}) = {fmt(Fa)}")
        nl = sp.simplify(Fb - Fa)
        if nl is not sp.nan:
            sub = f"({fmt(Fa)})" if (Fa.is_number and Fa.is_negative) or Fa.is_Add else fmt(Fa)
            log.step(f"{fmt(Fb)} - {sub} = {fmt(nl)}")
    show_result(log, truth)
    return truth


def value_at(F, x, point, direction):
    if point.is_infinite:
        return sp.limit(F, x, point)
    value = F.subs(x, point)
    if value.has(sp.nan, sp.zoo) or not value.is_finite:
        return sp.limit(F, x, point, direction)
    return sp.simplify(value)


def show_result(log, value):
    if value.is_infinite:
        log.answer(f"интеграл расходится ({fmt(value)})")
    elif value is sp.nan or value.has(sp.Integral) or value.has(sp.zoo):
        log.answer("интеграл расходится")
    else:
        log.answer(fmt_number(value))
