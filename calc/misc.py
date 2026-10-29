"""Ряд Тейлора, матрицы, упрощение выражений — с ходом решения."""

import sympy as sp

from calc.text import SUPERSCRIPT, Log, fmt, fmt_number

SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
from common import ParseError, parse, print_block

# ---------------------------------------------------------------- ряд Тейлора


def taylor(f, x, a=0, n=5):
    a = sp.sympify(a)
    log = Log()
    shift = fmt(x) if a == 0 else f"({fmt(x - a)})"
    power = str(n).translate(SUPERSCRIPT)
    log.title(f"Разложим {fmt(f)} в ряд Тейлора в точке {x} = {fmt(a)} до {shift}{power}")
    log.step(f"Формула: f({x}) = Σ f⁽ᵏ⁾({fmt(a)})/k! · {shift}ᵏ")
    terms = []
    current = f
    with log.sub():
        for k in range(n + 1):
            if k > 0:
                current = sp.simplify(sp.diff(current, x))
            value = sp.simplify(current.subs(x, a))
            if value.has(sp.nan, sp.zoo) or not value.is_finite:
                value = sp.limit(current, x, a)
            if not value.is_finite:
                what = "функция" if k == 0 else f"её производная порядка {k}"
                raise ParseError(f"{what} не определена в точке {x} = {fmt(a)} — ряд Тейлора там не строится. "
                                 f"Попробуйте другую точку: ряд ... в точке 1")
            coef = value / sp.factorial(k)
            name = "f" + "'" * k if k <= 3 else f"f⁽{str(k).translate(SUPERSCRIPT)}⁾"
            deriv = f"{name}({x}) = {fmt(current)},  " if k > 0 else ""
            log.step(f"k = {k}: {deriv}{name}({fmt(a)}) = {fmt(value)},  коэффициент {fmt(value)}/{k}! = {fmt(coef)}")
            terms.append(coef * (x - a) ** k)
    poly = sp.Add(*terms)
    # Записываем по возрастанию степеней, как принято для рядов.
    parts = [fmt(t) for t in terms if t != 0]
    text = " + ".join(parts).replace("+ -", "- ") if parts else "0"
    log.answer(f"{fmt(f)} = {text} + o({shift}{power})")
    return poly


# ---------------------------------------------------------------- матрицы


def parse_matrix(text, names=None):
    """'[[1, 2], [3, 4]]' или '1 2; 3 4' -> Matrix."""
    text = text.strip()
    if not text:
        raise ParseError("укажите матрицу, например: определитель 1 2; 3 4  или  [[1, 2], [3, 4]]")
    if text.startswith("["):
        value = parse(text, names)
        rows = value.tolist() if isinstance(value, sp.MatrixBase) else value
    else:
        rows = []
        for row in text.split(";"):
            cells = row.replace(",", " ").split()
            if not cells:
                raise ParseError("пустая строка матрицы (два «;» подряд?)")
            rows.append([parse(c, names) for c in cells])
    if not isinstance(rows, list) or not rows or not all(isinstance(r, list) for r in rows):
        raise ParseError("матрица записывается так: [[1, 2], [3, 4]]  или  1 2; 3 4")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ParseError(f"в строках матрицы разное число элементов: {', '.join(str(len(r)) for r in rows)}")
    return sp.Matrix(rows)


def need_square(A, what):
    if A.rows != A.cols:
        raise ParseError(f"{what} есть только у квадратной матрицы, а у этой размер {A.rows}×{A.cols}")


def mfmt(A):
    return "[" + ", ".join("[" + ", ".join(fmt(v) for v in A.row(i)) + "]" for i in range(A.rows)) + "]"


def determinant(A, log=None, top=True):
    need_square(A, "определитель")
    log = log or Log()
    if top:
        log.title(f"Найдём определитель матрицы {A.rows}×{A.cols}")
        print_block("    A =", A)
    n = A.rows
    if n == 1:
        det = A[0, 0]
        log.step(f"Определитель матрицы 1×1 равен её элементу: {fmt(det)}")
    elif n == 2:
        a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
        det = sp.simplify(a * d - b * c)
        log.step(f"|{mfmt(A)}| = a·d - b·c = {p(a)}·{p(d)} - {p(b)}·{p(c)} = {fmt(det)}")
    else:
        log.step(f"Разложение по первой строке: det A = Σ (-1)^(1+j)·a₁ⱼ·M₁ⱼ")
        total = 0
        with log.sub():
            for j in range(n):
                a = A[0, j]
                if a == 0:
                    log.step(f"j = {j + 1}: a₁{str(j + 1).translate(SUBSCRIPT)} = 0 — слагаемое равно 0")
                    continue
                minor = A.minor_submatrix(0, j)
                sign = (-1) ** j
                idx = str(j + 1).translate(SUBSCRIPT)
                log.step(f"j = {j + 1}: a₁{idx} = {fmt(a)}, минор M₁{idx} = |{mfmt(minor)}|")
                with log.sub():
                    m = determinant(minor, log, top=False) if minor.rows <= 3 else minor.det()
                term = sign * a * m
                log.line(f"слагаемое: {'+' if sign > 0 else '-'}{p(a)}·{p(m)} = {fmt(term)}")
                total += term
        det = sp.simplify(total)
        log.step(f"Сумма: det A = {fmt(det)}")
    if top:
        log.answer(f"det A = {fmt_number(det)}")
    return det


def inverse(A):
    need_square(A, "обратная матрица")
    log = Log()
    log.title(f"Найдём обратную матрицу {A.rows}×{A.cols}")
    print_block("    A =", A)
    det = sp.simplify(A.det())
    log.step(f"det A = {fmt(det)}")
    if det == 0:
        log.answer("det A = 0 — матрица вырожденная, обратной не существует")
        return None
    log.step("A⁻¹ = (1/det A)·adj(A), где adj(A) — транспонированная матрица алгебраических дополнений")
    adj = A.adjugate()
    log.block("adj(A) =", adj)
    inv = sp.simplify(adj / det)
    check = sp.simplify(A * inv) == sp.eye(A.rows)
    log.step("Проверка: A·A⁻¹ = E" + (" ✓" if check else ""))
    print()
    print_block("Ответ: A⁻¹ =", inv)
    return inv


def p(v):
    v = sp.sympify(v)
    return f"({fmt(v)})" if (v.is_number and v.is_negative) or v.is_Add else fmt(v)


# ---------------------------------------------------------------- преобразования


def transform(kind, expr):
    log = Log()
    if kind == "factor" and expr.is_Integer:
        log.title(f"Разложим число {expr} на простые множители")
        if abs(expr) < 2:
            log.answer(f"{expr} не раскладывается на простые множители")
            return expr
        factors = sp.factorint(expr)
        text = "·".join(f"{p_}{'' if k == 1 else str(k).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}"
                        for p_, k in sorted(factors.items()) if p_ != -1)
        sign = "-" if expr < 0 else ""
        if len(factors) == 1 and list(factors.values())[0] == 1:
            log.answer(f"{expr} — простое число")
        else:
            log.answer(f"{expr} = {sign}{text}")
        return expr
    titles = {
        "simplify": ("Упростим выражение", sp.simplify),
        "expand": ("Раскроем скобки", sp.expand),
        "factor": ("Разложим на множители", sp.factor),
        "together": ("Приведём к общему знаменателю", sp.together),
        "apart": ("Разложим на простейшие дроби", sp.apart),
    }
    title, func = titles[kind]
    log.title(f"{title}: {fmt(expr)}")
    if kind == "simplify":
        result = expr
        for name, f in [("раскрываем скобки", sp.expand), ("сокращаем дроби", sp.cancel),
                        ("применяем тригонометрические тождества", sp.trigsimp), ("общее упрощение", sp.simplify)]:
            new = f(result)
            if new != result and sp.count_ops(new) <= sp.count_ops(result):
                log.step(f"{name.capitalize()}: {fmt(new)}")
                result = new
    else:
        result = func(expr)
    if result == expr:
        log.answer(f"{fmt(expr)} (выражение уже в нужном виде)")
    else:
        print()
        print_block("Ответ:", result)
    return result
