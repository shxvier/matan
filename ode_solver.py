"""Консольное приложение: дифференциальные уравнения первого порядка.

Умеет (с выводом хода решения):
  * уравнения с разделяющимися переменными      y' = f(x)·g(y)
  * линейные уравнения                          y' + P(x)·y = Q(x)
  * уравнения Бернулли                          y' + P(x)·y = Q(x)·yⁿ
  * однородные уравнения                        y' = φ(y/x)
  * уравнения в полных дифференциалах           M dx + N dy = 0,  M'_y = N'_x
Остальные уравнения первого порядка решаются через SymPy (dsolve).
Можно задать начальное условие (задача Коши): y(x0) = y0.

Примеры запуска:
    python ode_solver.py                                  # интерактивное меню
    python ode_solver.py "y' = x*y" --ic "y(0) = 2"
    python ode_solver.py "y' + y/x = x^2"
    python ode_solver.py "(2xy + 1)dx + (x^2 + 2y)dy = 0"
    python ode_solver.py --separable "dy/dx = (1 + y^2)/(1 + x^2)"
"""

import argparse
import re

import sympy as sp
from sympy.solvers.ode.ode import constantsimp

from common import ParseError, parse, parse_equation, print_block

x, y = sp.symbols("x y")
u, z = sp.symbols("u z")  # вспомогательные переменные для замен
C = sp.Symbol("C1")
P_SYM = sp.Symbol("p")  # обозначает y'
DX, DY = sp.symbols("Dx Dy")  # обозначают dx и dy в дифференциальной форме

NAMES = {"x": x, "y": y, "C": C, "C1": C, "p": P_SYM, "Dx": DX, "Dy": DY}


class NotApplicable(Exception):
    """Метод не подходит для данного уравнения."""


# ---------------------------------------------------------------- разбор ввода


def parse_ode(text):
    """Возвращает (f, MN), где y' = f(x, y); MN = (M, N) для формы M dx + N dy = 0 или None."""
    text = text.strip()
    if re.search(r"dy\s*/\s*dx", text) is None and re.search(r"(?<![A-Za-z])d[xy](?![A-Za-z0-9])", text):
        text = re.sub(r"(?<![A-Za-z])d([xy])(?![A-Za-z0-9])", lambda m: f" D{m.group(1)} ", text)
        expr = sp.expand(parse_equation(text, NAMES))
        M, N = expr.coeff(DX), expr.coeff(DY)
        if sp.simplify(expr - M * DX - N * DY) != 0 or (M.free_symbols | N.free_symbols) & {DX, DY}:
            raise ParseError("ожидается форма M(x,y) dx + N(x,y) dy = 0")
        check_symbols(M + N)
        if N == 0:
            raise ParseError("коэффициент при dy равен нулю — это не уравнение относительно y(x)")
        return [sp.simplify(-M / N)], (sp.factor(M), sp.factor(N))

    original = text
    text = re.sub(r"dy\s*/\s*dx", " p ", text)
    text = re.sub(r"y\s*['′`]", " p ", text)
    try:
        expr = parse_equation(text, NAMES)
    except ParseError as exc:
        if "=" in str(exc):
            raise
        raise ParseError(f"не удалось разобрать «{original}»") from None
    if not expr.has(P_SYM):
        raise ParseError("в уравнении нет производной (пишите y' или dy/dx)")
    check_symbols(expr.subs(P_SYM, 0))
    roots = sp.solve(expr, P_SYM)
    if not roots:
        raise ParseError("не удалось выразить y' из уравнения")
    return [sp.simplify(r) for r in roots], None


def check_symbols(expr):
    extra = expr.free_symbols - {x, y}
    if extra:
        raise ParseError(f"неизвестные обозначения: {', '.join(map(str, extra))} (допустимы только x и y)")


def parse_ic(text):
    m = re.fullmatch(r"\s*y\s*\((.+)\)\s*=\s*(.+)", text)
    if not m:
        raise ParseError("начальное условие пишите так: y(0) = 1")
    return parse(m.group(1)), parse(m.group(2))


# ---------------------------------------------------------------- вспомогательное


def integral(expr, var):
    result = sp.integrate(expr, var)
    if result.has(sp.Integral):
        raise NotApplicable(f"SymPy не смог взять интеграл ∫ {expr} d{var}")
    return sp.simplify(result)


def tidy(expr):
    """Упрощает выражение с константой: e^(C1 + x) -> C1·e^x и т. п."""
    expr = sp.simplify(expr)
    try:
        return constantsimp(expr, [C]).doit()
    except Exception:  # noqa: BLE001
        return expr


def solve_for_y(lhs, rhs):
    """Пытается выразить y из общего интеграла lhs = rhs."""
    try:
        sols = sp.solve(sp.Eq(lhs, rhs), y)
    except (NotImplementedError, ValueError):
        return []
    return [tidy(s) for s in sols if not s.has(sp.RootOf)]


def result(implicit=None, explicit=(), special=()):
    return {"implicit": implicit, "explicit": list(explicit), "special": list(special)}


# ---------------------------------------------------------------- методы


def separable(f):
    parts = sp.separatevars(sp.factor(f), symbols=[x, y], dict=True)
    if parts is None:
        raise NotApplicable("переменные не разделяются")
    X, Y = sp.simplify(parts["coeff"] * parts[x]), sp.simplify(parts[y])
    print("Тип: уравнение с разделяющимися переменными  y' = X(x)·Y(y)")
    print_block("  X(x) =", X)
    print_block("  Y(y) =", Y)

    if Y == 1:
        print("Правая часть не зависит от y, просто интегрируем: y = ∫ X(x) dx + C")
        sol = integral(X, x) + C
        return result(explicit=[sol])

    print("Разделяем переменные:  dy / Y(y) = X(x) dx")
    print("Интегрируем обе части: ∫ dy / Y(y) = ∫ X(x) dx")
    left, right = integral(1 / Y, y), integral(X, x)
    print_block("  ∫ dy / Y(y) =", left)
    print_block("  ∫ X(x) dx  =", right)
    if left.has(sp.log) or right.has(sp.log):
        print("  (логарифмы понимаются по модулю, знак поглощается константой C)")

    special = [r for r in sp.solve(Y, y) if not r.has(x) and r.is_real is not False]
    implicit = sp.Eq(left, right + C)
    return result(implicit=implicit, explicit=solve_for_y(left, right + C), special=special)


def linear_parts(f):
    """Если f = a(x)·y + b(x), возвращает (a, b)."""
    a = sp.simplify(sp.diff(f, y))
    b = sp.simplify(f - a * y)
    if a.has(y) or b.has(y):
        raise NotApplicable("уравнение не линейно по y")
    return a, b


def solve_linear(P, Q, var):
    """Решает var' + P·var = Q методом интегрирующего множителя, печатая шаги."""
    mu = sp.simplify(sp.exp(integral(P, x)))
    print_block("  Интегрирующий множитель μ(x) = e^(∫P dx) =", mu)
    print(f"  Умножаем на μ:  (μ·{var})' = μ·Q")
    integ = integral(sp.simplify(mu * Q), x)
    print_block("  ∫ μ·Q dx =", integ)
    sol = tidy((integ + C) / mu)
    print_block(f"  {var} = (∫ μ·Q dx + C) / μ =", sol)
    return sol


def linear(f):
    a, b = linear_parts(f)
    if a == 0:
        raise NotApplicable("в уравнении нет слагаемого с y")
    P, Q = -a, b
    print("Тип: линейное уравнение  y' + P(x)·y = Q(x)")
    print_block("  P(x) =", P)
    print_block("  Q(x) =", Q)
    return result(explicit=[solve_linear(P, Q, "y")])


def bernoulli(f):
    groups = {}
    for term in sp.Add.make_args(sp.expand(f)):
        coeff, ypart = term.as_independent(y, as_Add=False)
        if ypart != 1 and not (ypart == y or (ypart.is_Pow and ypart.base == y and not ypart.exp.has(x, y))):
            raise NotApplicable("не уравнение Бернулли")
        groups[ypart] = groups.get(ypart, 0) + coeff
    powers = [k for k in groups if k not in (y, 1)]
    if 1 in groups or len(powers) != 1 or y not in groups:
        raise NotApplicable("не уравнение Бернулли")
    n = powers[0].exp
    P, Q = sp.simplify(-groups[y]), sp.simplify(groups[powers[0]])
    print("Тип: уравнение Бернулли  y' + P(x)·y = Q(x)·yⁿ")
    print_block("  P(x) =", P)
    print_block("  Q(x) =", Q)
    print_block("  n =", n)
    k = 1 - n
    print_block("Замена z = y^(1-n) =", y**k)
    print("Получаем линейное уравнение  z' + (1-n)·P·z = (1-n)·Q")
    zsol = solve_linear(sp.simplify(k * P), sp.simplify(k * Q), "z")
    special = [sp.Integer(0)] if n.is_positive else []
    return result(implicit=sp.Eq(y**k, zsol), explicit=solve_for_y(y**k, zsol), special=special)


def homogeneous(f):
    t = sp.Symbol("t", positive=True)
    if sp.simplify(f.subs({x: t * x, y: t * y}, simultaneous=True) - f) != 0:
        raise NotApplicable("не однородное уравнение")
    xp = sp.Symbol("x", positive=True)  # при x > 0: √(x²) = x
    phi = sp.simplify(f.subs(x, xp).subs(y, u * xp))
    if phi.has(xp):
        raise NotApplicable("не однородное уравнение")
    print("Тип: однородное уравнение  y' = φ(y/x)  (рассматриваем x > 0)")
    print_block("  φ(u) =", phi)
    print("Замена y = u·x, y' = u + x·u'  ⇒  x·u' = φ(u) - u")
    g = sp.simplify(phi - u)
    if g == 0:
        print("φ(u) - u = 0, значит u' = 0, u = C")
        return result(explicit=[C * x])
    print("Разделяем переменные: du / (φ(u) - u) = dx / x")
    left = integral(1 / g, u)
    print_block("  ∫ du / (φ(u) - u) =", left)
    print("  ∫ dx / x = ln|x|")
    special = [r * x for r in sp.solve(g, u) if not r.has(x) and r.is_real is not False]
    left_xy = sp.simplify(left.subs(u, y / x))
    print("Возвращаемся к y = u·x (u = y/x)")
    implicit = sp.Eq(left_xy, sp.log(x) + C)
    return result(implicit=implicit, explicit=solve_for_y(left_xy, sp.log(x) + C), special=special)


def exact(M, N):
    My, Nx = sp.simplify(sp.diff(M, y)), sp.simplify(sp.diff(N, x))
    if sp.simplify(My - Nx) != 0:
        raise NotApplicable("не уравнение в полных дифференциалах")
    print("Тип: уравнение в полных дифференциалах  M dx + N dy = 0")
    print_block("  M =", M)
    print_block("  N =", N)
    print_block("  ∂M/∂y = ∂N/∂x =", My)
    print("Ищем F(x, y): F'_x = M, F'_y = N")
    F = integral(M, x)
    print_block("  F = ∫ M dx + g(y) =", F)
    gprime = sp.simplify(N - sp.diff(F, y))
    print_block("  g'(y) = N - ∂(∫M dx)/∂y =", gprime)
    F = sp.simplify(F + integral(gprime, y))
    print_block("  F(x, y) =", F)
    return result(implicit=sp.Eq(F, C), explicit=solve_for_y(F, C))


def fallback(f):
    Y = sp.Function("y")
    ode = sp.Eq(Y(x).diff(x), f.subs(y, Y(x)))
    hints = [h for h in sp.classify_ode(ode) if not h.endswith("_Integral")]
    if not hints:
        raise NotApplicable("SymPy не нашёл метода решения")
    print(f"Тип не из стандартного списка. Методы SymPy: {', '.join(hints[:4])}")
    for hint in hints:
        try:
            sols = sp.dsolve(ode, hint=hint)
            break
        except Exception:  # noqa: BLE001 - пробуем следующий метод
            continue
    else:
        raise NotApplicable("SymPy не смог решить уравнение")
    print(f"Решено методом SymPy: {hint}")
    if hint == "1st_power_series":
        print("(ответ — приближение степенным рядом в окрестности x = 0)")
    sols = sols if isinstance(sols, list) else [sols]
    explicit, implicit = [], None
    for s in sols:
        s = s.subs(Y(x), y)
        if s.lhs == y and not s.rhs.has(y):
            explicit.append(tidy(s.rhs))
        else:
            implicit = s
    return result(implicit=implicit, explicit=explicit)


# ---------------------------------------------------------------- проверка и вывод


def verify(sol, f):
    """Подставляет явное решение в уравнение y' = f."""
    xp = sp.Symbol("x", positive=True)  # проверяем при x > 0, чтобы не мешали ветви корней
    try:
        residual = sp.diff(sol, x) - f.subs(y, sol)
        residual = residual.subs(x, xp)
        if sp.simplify(residual) == 0:
            return True
        # Символьно не упростилось — проверяем численно в нескольких точках.
        for xv, cv in [(0.7, 0.3), (1.3, 0.9), (2.1, 0.5)]:
            value = complex(residual.subs({xp: xv, C: cv}).evalf())
            if abs(value) > 1e-8:
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def included(special, explicit):
    """Получается ли решение special из общего решения при каком-то C."""
    for sol in explicit:
        try:
            if any(not c.has(x) for c in sp.solve(sp.Eq(sol, special), C)):
                return True
        except NotImplementedError:
            pass
    return False


def apply_ic(res, ic, f):
    x0, y0 = ic
    particular = []
    for sol in res["explicit"]:
        for c in sp.solve(sp.Eq(sol.subs(x, x0), y0), C):
            particular.append(sp.simplify(sol.subs(C, c)))
    if not particular and res["implicit"] is not None:
        eq = res["implicit"]
        for c in sp.solve(eq.subs({x: x0, y: y0}), C):
            spec = eq.subs(C, c)
            for s in solve_for_y(spec.lhs, spec.rhs):
                if sp.simplify(s.subs(x, x0) - y0) == 0:
                    particular.append(s)
            if not particular:
                print_block("Частный интеграл:", spec)
                return
    checked = [s for s in particular if verify(s, f)]
    particular = checked or particular
    particular += [s for s in res["special"] if sp.simplify(s.subs(x, x0) - y0) == 0]
    if not particular:
        print("Не удалось подобрать C под начальное условие.")
        return
    print(f"\nРешение задачи Коши y({x0}) = {y0}:")
    for s in dict.fromkeys(particular):
        print_block("  y =", s)


def report(res, f, ic):
    print("\n--- Ответ ---")
    if res["explicit"]:
        print("Общее решение:")
        for s in res["explicit"]:
            if s.has(sp.Order):
                print_block("  y ≈", s)
                continue
            mark = "✓ проверено подстановкой" if verify(s, f) else "(автопроверка не подтвердила — возможно, верно лишь на части области)"
            print_block(f"  y =   {mark}", s)
    elif res["implicit"] is not None:
        print("Явно выразить y не удалось, ответ — общий интеграл:")
    if res["implicit"] is not None:
        print_block("Общий интеграл:", res["implicit"])
    if res["special"]:
        print("Решения, которые могли потеряться при делении:")
        for s in res["special"]:
            note = "входит в общее решение" if included(s, res["explicit"]) else "особое решение"
            print_block(f"  ({note})  y =", s)
    print("Здесь C₁ — произвольная постоянная.")
    if ic is not None:
        apply_ic(res, ic, f)


def solve(text, ic_text=None, only_separable=False):
    try:
        roots, MN = parse_ode(text)
        ic = parse_ic(ic_text) if ic_text else None
    except ParseError as exc:
        print(f"Ошибка: {exc}")
        return

    for f in roots:
        print_block("\nУравнение в нормальной форме  y' =", f)
        if only_separable:
            methods = [separable]
        else:
            methods = [separable, linear, bernoulli, homogeneous]
            if MN is not None:
                methods.insert(0, lambda _f: exact(*MN))
            methods.append(fallback)
        for method in methods:
            try:
                res = method(f)
            except NotApplicable as exc:
                if only_separable:
                    print(f"Это уравнение нельзя решить разделением переменных: {exc}.")
                    print("Попробуйте пункт «ДУ первого порядка».")
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"Ошибка при решении: {exc}")
                continue
            report(res, f, ic)
            break
        else:
            if not only_separable:
                print("Не удалось решить уравнение.")


# ---------------------------------------------------------------- интерфейс

HELP = """
Как вводить:
  y' = x*y            dy/dx = (1+y^2)/(1+x^2)      x y' + y = x^3
  (2xy + 1)dx + (x^2 + 2y)dy = 0
  Умножение можно не писать: 2xy = 2·x·y. Степень: ^ или **.
  Функции: sin, cos, tg, exp, e^x, ln, sqrt, arctg ...
Начальное условие (необязательно): y(0) = 1
"""


def interactive():
    print("=== Решатель дифференциальных уравнений первого порядка ===")
    while True:
        print("\n1) Уравнение с разделяющимися переменными")
        print("2) ДУ первого порядка (тип определяется автоматически)")
        print("h) Как вводить уравнения")
        print("q) Выход")
        try:
            choice = input("Выбор: ").strip().lower()
            if choice in ("q", "й", "exit", "выход"):
                break
            if choice == "h":
                print(HELP)
                continue
            if choice not in ("1", "2"):
                continue
            text = input("Уравнение: ").strip()
            if not text:
                continue
            ic = input("Начальное условие y(x0)=y0 (Enter — пропустить): ").strip() or None
        except EOFError:
            break
        solve(text, ic, only_separable=(choice == "1"))
    print("До свидания!")


def main():
    parser = argparse.ArgumentParser(description="Решение ДУ первого порядка")
    parser.add_argument("equation", nargs="?", help="уравнение, например \"y' = x*y\"")
    parser.add_argument("--ic", help="начальное условие, например \"y(0) = 1\"")
    parser.add_argument("--separable", action="store_true", help="решать только разделением переменных")
    args = parser.parse_args()
    if args.equation:
        solve(args.equation, args.ic, args.separable)
    else:
        interactive()


if __name__ == "__main__":
    main()
