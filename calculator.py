"""Математический калькулятор с ходом решения (на русском языке).

Запуск:
    python calculator.py                          # интерактивный режим, «помощь» — справка
    python calculator.py "производная x^2 sin(x)"
    python calculator.py "x^2 - 5x + 6 = 0" "интеграл x e^x dx"
"""

import difflib
import re
import sys

import sympy as sp

from calc import derivative, equations, integral, limits, misc
from calc.text import approx, choose_variable, fmt, format_float, needs_approx_value, strip_trailing
from common import ParseError, parse, preprocess, print_block, split_equation

HELP = """
══════════════════════ КАЛЬКУЛЯТОР — СПРАВКА ══════════════════════

Вычисления (просто введите выражение):
    2 + 2·3      7/3      2^100      10!      √8      0,1 + 0,2
    sin(pi/6)    ln(e^2)  log(8, 2)  arctg(1)  |−5|   (1 + i)^2

Производная (с ходом решения):
    производная x^2 sin(x)
    производная x^3 по x порядка 2        (x^2 + 1)'      (x^3)''

Интеграл:
    интеграл x e^x dx
    интеграл sin(x) от 0 до pi            интеграл e^(-x) от 0 до ∞

Предел:
    предел sin(x)/x при x -> 0
    предел (1 + 1/x)^x при x -> ∞         предел 1/x при x -> 0+

Уравнения, системы, неравенства (можно без слова «решить»):
    x^2 - 5x + 6 = 0          sin(x) = 1/2          √(x + 3) = x + 1
    x + y = 5; x y = 6        (уравнения системы — через «;»)
    x^2 - 4 < 0               (x - 1)/(x + 2) >= 0
    решить a x + 3 = 0 относительно x

Ряд Тейлора:
    ряд e^x                   ряд ln(x) в точке 1 до 4

Матрицы (строки через «;» или [[1, 2], [3, 4]]):
    определитель 1 2; 3 4     обратная 2 1; 1 1

Преобразования:
    упростить sin(x)^2 + cos(x)^2      раскрыть (x + 1)^3
    разложить x^2 - 1                   разложить 360
    дроби 1/(x^2 - 1)  (на простейшие)  общий знаменатель 1/x + 1/(x+1)

Переменные и функции:
    пусть a = 5               пусть f(x) = x^2 + 1    →    f(a)
    ans — последний результат,  переменные — список,  удалить a,  очистить

Другие решатели:
    ду y' = x y, y(0) = 2     — дифференциальное уравнение 1-го порядка
    ду                        — меню решателя ДУ
    линейные                  — системы линейных уравнений методом Гаусса

    выход — завершить работу

Запись: умножение можно не писать (2x, 3sin(x)), степень ^,
корень √ или sqrt, модуль |x|, π или pi, ∞ или oo, дроби через точку или запятую.
"""

COMMANDS = {
    "производная": "derivative", "произв": "derivative", "diff": "derivative", "d": "derivative",
    "интеграл": "integral", "инт": "integral", "integral": "integral", "int": "integral", "∫": "integral",
    "предел": "limit", "lim": "limit", "limit": "limit",
    "решить": "solve", "реши": "solve", "solve": "solve", "уравнение": "solve",
    "ряд": "series", "тейлор": "series", "series": "series", "taylor": "series",
    "определитель": "det", "det": "det",
    "обратная": "inverse", "inverse": "inverse",
    "упростить": "simplify", "simplify": "simplify",
    "раскрыть": "expand", "expand": "expand",
    "разложить": "factor", "factor": "factor", "множители": "factor",
    "дроби": "apart", "apart": "apart",
    "вычислить": "eval", "посчитать": "eval",
    "пусть": "let", "let": "let",
    "переменные": "vars", "vars": "vars",
    "удалить": "del", "del": "del",
    "очистить": "clear", "clear": "clear",
    "помощь": "help", "справка": "help", "help": "help", "?": "help",
    "выход": "quit", "q": "quit", "quit": "quit", "exit": "quit",
    "ду": "ode", "ode": "ode", "диффур": "ode",
    "линейные": "lin", "lin": "lin", "гаусс": "lin",
}
TWO_WORD_COMMANDS = {
    "общий знаменатель": "together",
    "вторая производная": "derivative2",
    "третья производная": "derivative3",
    "обратная матрица": "inverse",
}
# Команды, после которых может идти не выражение, а «голая» строка.
NO_ARGUMENT_COMMANDS = {"quit", "help", "vars", "clear", "lin"}

INFINITY_WORDS = re.compile(r"\+?\s*(бесконечност[ьи]|беск)", re.IGNORECASE)
RESERVED = {"e", "pi", "i", "oo", "inf", "ans"}


class Calculator:
    def __init__(self):
        self.vars = {}

    # ------------------------------------------------------------ разбор

    def names(self):
        return {"i": sp.I, "I": sp.I, **self.vars}

    def parse(self, text):
        text = INFINITY_WORDS.sub("oo", text)
        return parse(text, self.names())

    def parse_relation(self, text):
        """'a < b', 'a >= b' -> Relational."""
        m = re.search(r"(<=|>=|<|>)", text)
        left, right = text[: m.start()], text[m.end() :]
        if not left.strip() or not right.strip():
            raise ParseError("у неравенства должны быть обе части, например: x^2 - 4 < 0")
        if re.search(r"(<=|>=|<|>)", right):
            raise ParseError("двойные неравенства (a < x < b) пока не поддерживаются — решите их по частям")
        return sp.Rel(self.parse(left), self.parse(right), m.group(1))

    def store(self, value):
        if isinstance(value, (sp.Basic, sp.MatrixBase)) and not isinstance(value, sp.Set):
            self.vars["ans"] = value
        return value

    # ------------------------------------------------------------ команды

    def cmd_eval(self, text):
        if not text.strip():
            raise ParseError("введите выражение")
        value = self.parse(text)
        if isinstance(value, sp.core.relational.Relational):
            return equations.solve_inequality(value)
        self.show_value(value)
        return self.store(value)

    def show_value(self, value):
        if isinstance(value, (list, tuple)):
            print("= " + ", ".join(fmt(v) for v in value))
            return
        if isinstance(value, sp.Basic) and value.is_number:
            if value.has(sp.I):
                value = sp.simplify(sp.expand(value))
            floats = value.atoms(sp.Float)
            if floats and value.is_real and all(f._prec <= 53 for f in floats):
                value = sp.N(value, 15)
            if isinstance(value, sp.Float):
                print(f"= {format_float(value)}")
                return
        print_block("=", value)
        if needs_approx_value(value):
            print(f"≈ {approx(value)}")
        elif isinstance(value, sp.Basic) and not value.is_number and value.free_symbols:
            simpler = sp.simplify(value)
            if sp.count_ops(simpler) < sp.count_ops(value):
                print_block("Упрощённо:", simpler)

    def cmd_derivative(self, text, order=1):
        text, m = strip_trailing(text, r"\s+порядка\s+(\d+)\s*$")
        if m:
            order = int(m.group(1))
        text, var = self.take_variable(text)
        f = self.need_expr(text, "производная", "x^2 sin(x)")
        x = choose_variable(f, var)
        if order < 1 or order > 10:
            raise ParseError("порядок производной должен быть от 1 до 10")
        return self.store(derivative.derivative(f, x, order))

    def cmd_integral(self, text):
        text, bounds = strip_trailing(text, r"\s+от\s+(.+?)\s+до\s+(.+?)\s*$")
        text, var = self.take_variable(text)
        text, dx = strip_trailing(text, r"(?<![A-Za-z])d([A-Za-z])\s*$")
        if dx:
            var = sp.Symbol(dx.group(1))
        if re.search(r"\s(от|до)\s", f" {text} "):
            raise ParseError("пределы интегрирования пишутся так: интеграл sin(x) от 0 до pi")
        f = self.need_expr(text, "интеграл", "x e^x dx")
        x = choose_variable(f, var)
        if bounds:
            a, b = self.parse(bounds.group(1)), self.parse(bounds.group(2))
            for bound in (a, b):
                if isinstance(bound, sp.Basic) and x in bound.free_symbols:
                    raise ParseError("пределы интегрирования не должны зависеть от переменной интегрирования")
            return self.store(integral.definite(f, x, a, b))
        return self.store(integral.indefinite(f, x))

    def cmd_limit(self, text):
        m = re.match(
            r"^(.*?)(?:\s+при\s+|\s*,\s*|\s+)([A-Za-z]\w*)\s*(?:->|→|стремится\s+к)\s*(.+)$", text.strip()
        )
        if not m:
            raise ParseError("укажите, куда стремится переменная, например: предел sin(x)/x при x -> 0")
        f = self.need_expr(m.group(1), "предел", "sin(x)/x при x -> 0")
        x = sp.Symbol(m.group(2))
        point_text = m.group(3).strip()
        direction = "+-"
        side = re.fullmatch(r"(.+?)\s*([+-])\s*0?", point_text)
        if side:
            point_text, direction = side.group(1), side.group(2)
        a = self.parse(point_text)
        if not isinstance(a, sp.Basic) or a.free_symbols:
            raise ParseError(f"точка, к которой стремится {x}, должна быть числом (или ∞)")
        if a.is_infinite:
            direction = "+-"
        return self.store(limits.limit(f, x, a, direction))

    def cmd_solve(self, text):
        text = text.strip()
        text, var_m = strip_trailing(text, r"\s+относительно\s+([A-Za-z]\w*)\s*$")
        var = sp.Symbol(var_m.group(1)) if var_m else None
        if not text:
            raise ParseError("после «решить» напишите уравнение, например: решить x^2 - 4 = 0")
        parts = [p.strip() for p in text.split(";") if p.strip()]
        if len(parts) > 1:
            exprs = []
            for part in parts:
                if re.search(r"(<=|>=|<|>)", part):
                    raise ParseError("системы неравенств пока не поддерживаются")
                left, right = split_equation(part) if "=" in part else (part, "0")
                exprs.append(self.parse(left) - self.parse(right))
            return equations.solve_system(exprs)
        if re.search(r"(<=|>=|<|>)", text):
            return equations.solve_inequality(self.parse_relation(text))
        left, right = split_equation(text) if "=" in text else (text, "0")
        return equations.solve_equation(self.parse(left), self.parse(right), var)

    def cmd_series(self, text):
        text, n_m = strip_trailing(text, r"\s+(?:до|порядка)\s+(\d+)\s*$")
        text, a_m = strip_trailing(text, r"\s+(?:в\s+точке|в|около)\s+(\S+)\s*$")
        if a_m is None:
            text, a_m2 = strip_trailing(text, r"\s+при\s+[A-Za-z]\w*\s*(?:->|→)\s*(\S+)\s*$")
            a_text = a_m2.group(1) if a_m2 else "0"
        else:
            a_text = a_m.group(1)
        f = self.need_expr(text, "ряд", "e^x в точке 0 до 5")
        x = choose_variable(f)
        n = int(n_m.group(1)) if n_m else 5
        if n > 12:
            raise ParseError("порядок ряда — не больше 12")
        a = self.parse(a_text)
        return self.store(misc.taylor(f, x, a, n))

    def cmd_det(self, text):
        return self.store(misc.determinant(misc.parse_matrix(text, self.names())))

    def cmd_inverse(self, text):
        return self.store(misc.inverse(misc.parse_matrix(text, self.names())))

    def cmd_transform(self, kind, text):
        names = {"simplify": "упростить", "expand": "раскрыть", "factor": "разложить",
                 "apart": "дроби", "together": "общий знаменатель"}
        expr = self.need_expr(text, names[kind], "(x + 1)^2")
        return self.store(misc.transform(kind, expr))

    def cmd_let(self, text):
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*(?:\(\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\))?\s*:?=\s*(.+)$", text)
        if not m:
            raise ParseError("пишите так: пусть a = 5   или   пусть f(x) = x^2 + 1")
        name, args, body = m.groups()
        if name in RESERVED or (hasattr(sp, name) and len(name) > 1):
            raise ParseError(f"имя «{name}» занято (это константа или функция), выберите другое")
        if args:
            params = [sp.Symbol(a.strip()) for a in args.split(",")]
            saved = {p.name: self.vars.pop(p.name) for p in params if p.name in self.vars}
            try:
                value = sp.Lambda(tuple(params), self.parse(body))
            finally:
                self.vars.update(saved)
            self.vars[name] = value
            print(f"Функция {name}({args}) = {fmt(value.expr)}")
        else:
            value = self.parse(body)
            self.vars[name] = value
            print(f"{name} = {fmt(value)}")

    def cmd_vars(self, _):
        if not self.vars:
            print("Переменных пока нет. Пример: пусть a = 5")
        for name, value in self.vars.items():
            if isinstance(value, sp.Lambda):
                print(f"  {name}({', '.join(map(str, value.variables))}) = {fmt(value.expr)}")
            else:
                print(f"  {name} = {fmt(value)}")

    def cmd_del(self, text):
        names = text.replace(",", " ").split()
        if not names:
            raise ParseError("укажите, какую переменную удалить: удалить a")
        for name in names:
            print(f"{name} удалена." if self.vars.pop(name, None) is not None else f"Переменной «{name}» нет.")

    def cmd_ode(self, text):
        import ode_solver

        if not text.strip():
            ode_solver.interactive()
            return
        ic = None
        m = re.search(r"[,;]\s*(y\s*\(.+?\)\s*=.+)$", text)
        if m:
            ic, text = m.group(1), text[: m.start()]
        ode_solver.solve(text, ic)

    # ------------------------------------------------------------ помощники

    def take_variable(self, text):
        text, m = strip_trailing(text, r"\s+по\s+([A-Za-z]\w*)\s*$")
        return text, (sp.Symbol(m.group(1)) if m else None)

    def need_expr(self, text, command, example):
        if not text.strip():
            raise ParseError(f"после команды «{command}» укажите выражение, например: {command} {example}")
        if re.search(r"(?<![<>!=:])=(?!=)|<|>", text):
            raise ParseError(f"для команды «{command}» нужно выражение, а не уравнение или неравенство")
        value = self.parse(text)
        if not isinstance(value, sp.Basic) or isinstance(value, sp.MatrixBase):
            raise ParseError("здесь нужно математическое выражение")
        return value

    # ------------------------------------------------------------ главный цикл

    def execute(self, line):
        """Выполняет одну строку. Возвращает False, если нужно выйти."""
        line = line.strip()
        if not line:
            return True
        # «∞», «≤», «−» и т. п. — заменяем, но русские слова-команды не трогаем.
        line = preprocess(line, fix_cyrillic=False)

        low = line.lower()
        for phrase, cmd in TWO_WORD_COMMANDS.items():
            if low == phrase or low.startswith(phrase + " "):
                return self.dispatch(cmd, line[len(phrase) :].strip())

        m = re.match(r"^([^\W\d_]+|∫|\?)(?:\s+(.*))?$", line)
        if m and m.group(1).lower() in COMMANDS:
            cmd = COMMANDS[m.group(1).lower()]
            # «d», «int» и т. п. могут быть и обычными именами: d + 1
            if not (len(m.group(1)) <= 3 and m.group(2) and re.match(r"^[+\-*/^=)]", m.group(2))):
                return self.dispatch(cmd, (m.group(2) or "").strip())
        if m and re.fullmatch(r"[а-яёА-ЯЁ]+", m.group(1)):
            word = m.group(1).lower()
            guess = difflib.get_close_matches(word, list(COMMANDS) + list(TWO_WORD_COMMANDS), n=1, cutoff=0.6)
            hint = f" Возможно, вы имели в виду «{guess[0]}»?" if guess else " Введите «помощь» — список команд."
            raise ParseError(f"неизвестная команда «{m.group(1)}».{hint}")

        # (f)' или (f)'' — производная
        m = re.fullmatch(r"\((.+)\)\s*('+)", line)
        if m:
            return self.dispatch("derivative", m.group(1), order=len(m.group(2)))
        if "'" in line:
            raise ParseError("производную пишите так: (x^2 + 1)'  или  производная x^2 + 1. "
                             "Для дифференциального уравнения: ду y' = x y")
        if re.match(r"^\s*[A-Za-z_]\w*\s*(\([^()]*\))?\s*:=", line):
            return self.dispatch("let", line)
        if ";" in line or re.search(r"(?<![<>!=:])=(?!=)", line) or re.search(r"(<=|>=|<|>)", line):
            return self.dispatch("solve", line)
        return self.dispatch("eval", line)

    def dispatch(self, cmd, text, order=1):
        if cmd in NO_ARGUMENT_COMMANDS and text and cmd != "lin":
            raise ParseError("эта команда пишется без продолжения")
        if cmd == "quit":
            return False
        if cmd == "help":
            print(HELP)
        elif cmd == "derivative":
            self.cmd_derivative(text, order)
        elif cmd in ("derivative2", "derivative3"):
            self.cmd_derivative(text, int(cmd[-1]))
        elif cmd in ("simplify", "expand", "factor", "apart", "together"):
            self.cmd_transform(cmd, text)
        elif cmd == "clear":
            self.vars.clear()
            print("Все переменные удалены.")
        elif cmd == "lin":
            import linear_solver

            if text:
                linear_solver.run([t for t in text.split(";") if t.strip()])
            else:
                linear_solver.interactive()
        else:
            getattr(self, f"cmd_{cmd}")(text)
        return True

    def run_line(self, line):
        try:
            return self.execute(line)
        except ParseError as exc:
            print(f"Ошибка ввода: {exc}")
        except KeyboardInterrupt:
            print("Прервано.")
        except RecursionError:
            print("Ошибка: выражение слишком сложное.")
        except Exception as exc:  # noqa: BLE001 - калькулятор не должен падать
            print(f"Не удалось вычислить: {exc}")
        return True


def main():
    calc = Calculator()
    if len(sys.argv) > 1:
        for line in sys.argv[1:]:
            print(f"\n>>> {line}")
            calc.run_line(line)
        return

    print("=== Калькулятор с ходом решения ===   «помощь» — справка и примеры, «выход» — завершить")
    while True:
        try:
            line = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not calc.run_line(line):
            break
    print("До свидания!")


if __name__ == "__main__":
    main()
