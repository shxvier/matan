"""Общий консольный калькулятор (арифметика, алгебра, матанализ, уравнения).

Запуск:
    python calculator.py                   # интерактивный режим
    python calculator.py "sin(pi/6) + 2^10"
    python calculator.py "x^2 - 5x + 6 = 0"

Команды внутри калькулятора — введите help.
"""

import re
import sys

import sympy as sp

from common import ParseError, parse, print_block

HELP = """
Арифметика:      2 + 2*3     7/3     2^100     10!     sqrt(8)     0.1 + 0.2
Функции:         sin(pi/6)   ln(e^2)   log(8, 2)   arctg(1)   abs(-3)   exp(1)
Точность:        N(pi, 50)   — 50 знаков
Комплексные:     sqrt(-4)    (1 + I)^2
Алгебра:         expand((x+1)^3)   factor(x^2 - 1)   simplify(sin(x)^2 + cos(x)^2)
Производная:     diff(x^2 sin(x), x)        diff(x^3, x, 2)  — вторая
Интеграл:        int(x^2, x)                int(sin(x), (x, 0, pi))
Предел:          limit(sin(x)/x, x, 0)      limit(1/x, x, 0, "-")   oo — бесконечность
Ряд Тейлора:     series(e^x, x, 0, 5)
Сумма ряда:      summation(1/n^2, (n, 1, oo))
Матрицы:         Matrix([[1, 2], [3, 4]]).det()     Matrix([[1, 2], [3, 4]])^-1
Уравнения:       x^2 - 5x + 6 = 0          cos(x) = x   (численно, если точно нельзя)
Системы:         x + y = 3; x - y = 1      (уравнения через «;»)
Неравенства:     x^2 - 4 < 0
Переменные:      a := 5        f(x) := x^2 + 1   →   f(a)       ans — прошлый результат
Служебные:       vars — переменные, del a — удалить, clear — сбросить всё
Другие решатели: lin — системы линейных уравнений (Гаусс), ode — ДУ первого порядка
Выход:           q
"""

ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\(\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\))?\s*:=\s*(.+)$")
EQUALS_RE = re.compile(r"(?<![<>!=:])=(?!=)")
RESERVED = {"e", "pi", "I", "oo", "ans"} | {name for name in dir(sp) if not name.startswith("_")}

CALC_NAMES = {
    "int": sp.integrate,
    "integral": sp.integrate,
    "lim": sp.limit,
    "d": sp.diff,
    "inf": sp.oo,
    "i": sp.I,
}


def format_float(value):
    """0.300000000000000 -> 0.3; длинные числа (N(pi, 50)) выводятся целиком."""
    if value._prec > 53:
        return str(value)
    return f"{float(value):.15g}"


def needs_approx(value):
    """Нужно ли показывать десятичное приближение (для 7/3, √2, π, но не для 5 или 2+3i)."""
    if not (isinstance(value, sp.Basic) and value.is_number and value.is_finite):
        return False
    if value.atoms(sp.Float):
        return False
    return not all(part.is_Integer for part in value.as_real_imag())


def approx(value):
    re_part, im_part = (format_float(sp.N(p, 15)) for p in value.as_real_imag())
    if im_part in ("0", "-0"):
        return re_part
    sign = "-" if im_part.startswith("-") else "+"
    im_part = im_part.lstrip("-")
    return f"{im_part}i" if re_part in ("0", "-0") and sign == "+" else (
        f"-{im_part}i" if re_part in ("0", "-0") else f"{re_part} {sign} {im_part}i")


class Calculator:
    def __init__(self):
        self.vars = {}

    # ------------------------------------------------------------ разбор

    def parse(self, text):
        return parse(text, {**CALC_NAMES, **self.vars}, decimal_comma=False)

    # ------------------------------------------------------------ вывод

    @staticmethod
    def show_result(value):
        if value is sp.zoo:
            print("= ∞ (деление на ноль)")
            return
        if value is sp.nan:
            print("= не определено")
            return
        if isinstance(value, sp.Basic) and value.is_number and value.has(sp.I):
            value = sp.simplify(sp.expand(value))
        floats = value.atoms(sp.Float) if isinstance(value, sp.Basic) else ()
        if floats and value.is_number and value.is_real and all(f._prec <= 53 for f in floats):
            value = sp.N(value, 15)  # в выражении есть десятичные дроби — считаем приближённо
        if isinstance(value, sp.Float):
            print(f"= {format_float(value)}")
            return
        print_block("=", value)
        if needs_approx(value):
            print(f"≈ {approx(value)}")

    # ------------------------------------------------------------ действия

    def assign(self, match):
        name, args, body = match.groups()
        if name in RESERVED:
            print(f"Имя «{name}» занято, выберите другое.")
            return
        if args:
            params = [sp.Symbol(a.strip()) for a in args.split(",")]
            # Параметры функции не должны подменяться значениями переменных.
            saved = {p.name: self.vars.pop(p.name) for p in params if p.name in self.vars}
            try:
                value = sp.Lambda(tuple(params), self.parse(body))
            finally:
                self.vars.update(saved)
            self.vars[name] = value
            print_block(f"{name}({args}) :=", value.expr)
        else:
            value = self.parse(body)
            self.vars[name] = value
            print_block(f"{name} :=", value)

    def solve_equations(self, text):
        exprs = []
        for part in filter(None, (p.strip() for p in text.split(";"))):
            if len(EQUALS_RE.findall(part)) != 1:
                raise ParseError(f"в «{part}» должен быть ровно один знак «=»")
            left, right = EQUALS_RE.split(part)
            exprs.append(self.parse(left) - self.parse(right))

        symbols = sorted(set().union(*(e.free_symbols for e in exprs)), key=lambda s: s.name)
        if not symbols:
            ok = all(sp.simplify(e) == 0 for e in exprs)
            print("Верно" if ok else "Неверно")
            return

        if len(exprs) == 1 and len(symbols) == 1:
            self.solve_single(exprs[0], symbols[0])
            return

        solutions = sp.solve(exprs, symbols, dict=True)
        if not solutions:
            print("Решений не найдено.")
        for i, sol in enumerate(solutions, 1):
            print(f"Решение {i}:" if len(solutions) > 1 else "Решение:")
            for s in symbols:
                if s in sol:
                    self.print_root(s, sol[s])
                else:
                    print(f"    {s} — любое")

    def solve_single(self, expr, var):
        try:
            roots = sp.solve(expr, var)
        except NotImplementedError:
            roots = None
        if roots:
            for r in roots:
                self.print_root(var, r)
            return
        # Точно решить не удалось — ищем вещественные корни численно.
        numeric = []
        for guess in range(-20, 21):
            try:
                r = sp.nsolve(expr, var, guess)
            except Exception:  # noqa: BLE001 - из этой точки не сошлось
                continue
            if r.is_real and all(abs(r - other) > 1e-9 for other in numeric):
                numeric.append(r)
        if not numeric:
            print("Решений не найдено.")
            return
        print("Точное решение не найдено, численно найдены корни (поиск на [-20; 20]):")
        for r in sorted(numeric):
            print(f"    {var} ≈ {format_float(r)}")

    @staticmethod
    def print_root(var, value):
        value = sp.simplify(value)
        line = f"    {var} = {sp.sstr(value)}"
        if needs_approx(value):
            line += f"   ≈ {approx(value)}"
        print(line.replace("log(", "ln(").replace("**", "^"))

    def solve_inequality(self, rel):
        symbols = list(rel.free_symbols)
        if len(symbols) != 1:
            print("Неравенства решаются только с одной переменной.")
            return
        solution = sp.solve_univariate_inequality(rel, symbols[0], relational=False)
        print_block("Решение:", solution)

    # ------------------------------------------------------------ главный цикл

    def execute(self, line):
        """Выполняет одну строку. Возвращает False, если нужно выйти."""
        line = line.strip()
        cmd = line.lower()
        if not line:
            return True
        if cmd in ("q", "quit", "exit", "выход"):
            return False
        if cmd in ("help", "?", "помощь", "h"):
            print(HELP)
        elif cmd == "vars":
            if not self.vars:
                print("Переменных нет.")
            for name, value in self.vars.items():
                if isinstance(value, sp.Lambda):
                    args = ", ".join(map(str, value.variables))
                    print_block(f"{name}({args}) =", value.expr)
                else:
                    print_block(f"{name} =", value)
        elif cmd == "clear":
            self.vars.clear()
            print("Переменные удалены.")
        elif cmd.startswith("del "):
            for name in line[4:].replace(",", " ").split():
                print(f"{name} удалена." if self.vars.pop(name, None) is not None else f"Нет переменной {name}.")
        elif cmd == "lin":
            import linear_solver

            linear_solver.interactive()
        elif cmd == "ode":
            import ode_solver

            ode_solver.interactive()
        elif (match := ASSIGN_RE.match(line)) is not None:
            self.assign(match)
        elif EQUALS_RE.search(line):
            self.solve_equations(line)
        else:
            value = self.parse(line)
            if isinstance(value, sp.core.relational.Relational):
                self.solve_inequality(value)
                return True
            self.vars["ans"] = value
            self.show_result(value)
        return True

    def run_line(self, line):
        try:
            return self.execute(line)
        except KeyboardInterrupt:
            print("Прервано.")
        except Exception as exc:  # noqa: BLE001 - калькулятор не должен падать
            print(f"Ошибка: {exc}")
        return True


def main():
    calc = Calculator()
    if len(sys.argv) > 1:
        for line in sys.argv[1:]:
            print(f">>> {line}")
            calc.run_line(line)
        return

    print("=== Калькулятор ===  help — справка, q — выход")
    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not calc.run_line(line):
            break
    print("До свидания!")


if __name__ == "__main__":
    main()
