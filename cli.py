"""Консольный интерфейс решателя ОДУ: пошаговые решения «как в тетради».

Примеры:
    python cli.py                                  # интерактивное меню
    python cli.py "y' = x/y" --ic "y(0)=1"
    python cli.py --task Б5
    python cli.py --all
    python cli.py --task Г4 --ivp "theta(0)=1,theta'(0)=0" --interval 0 10 --plot g4.png
"""

from __future__ import annotations

import argparse
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # Windows: иначе Unicode-формулы ломаются
    except (AttributeError, ValueError):
        pass

import sympy as sp  # noqa: E402
from sympy.printing.pretty.pretty import PrettyPrinter  # noqa: E402
from sympy.printing.pretty.stringpict import prettyForm  # noqa: E402
from sympy.printing.str import StrPrinter  # noqa: E402
from rich import box  # noqa: E402
from rich.console import Console, Group  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.prompt import Prompt  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from solver import ParseError, parse_ode, solve, solve_numeric  # noqa: E402
from solver.model import TYPE_NAMES, Solution  # noqa: E402
from solver.problems import PROBLEMS, find  # noqa: E402
from solver.systems import deriv_name  # noqa: E402

console = Console(highlight=False)


# ---------------------------------------------------------------- оформление формул


def to_display(expr, ode):
    """y(x) -> y, y'(x) -> y', log -> ln — запись как в тетради."""
    if isinstance(expr, (list, tuple)):
        return [to_display(e, ode) for e in expr]
    if not isinstance(expr, (sp.Basic, sp.MatrixBase)):
        return expr
    x, y, name = ode.x, ode.y, ode.func.__name__
    # некоммутативные символы печатаются последними: x·y', а не y'·x
    mapping = {y.diff(x, k): sp.Symbol(deriv_name(name, k), commutative=False) for k in range(ode.order, 0, -1)}
    mapping[y] = sp.Symbol(deriv_name(name, 0))
    expr = expr.xreplace(mapping)
    if isinstance(expr, sp.Basic):
        expr = expr.replace(sp.log, sp.Function("ln"))
    return expr


def _derivative_order(term, ode):
    names = {deriv_name(ode.func.__name__, k): k for k in range(1, ode.order + 1)}
    return max((names.get(s.name, 0) for s in term.free_symbols), default=0)


def pretty(expr, ode) -> str:
    """Как sympy.pretty, но в левой части уравнения производные идут первыми: y' + p·y = q."""
    expr = to_display(expr, ode)
    if isinstance(expr, sp.Eq) and isinstance(expr.lhs, sp.Add) and _derivative_order(expr.lhs, ode):
        printer = PrettyPrinter({"use_unicode": True})
        terms = sorted(expr.lhs.as_ordered_terms(), key=lambda t: -_derivative_order(t, ode))
        form = printer._print(terms[0])
        for t in terms[1:]:
            sign, t = (" - ", -t) if t.could_extract_minus_sign() else (" + ", t)
            form = prettyForm(*form.right(sign, printer._print(t)))
        return str(prettyForm(*form.right(" = ", printer._print(expr.rhs))))
    return sp.pretty(expr, use_unicode=True)


class LinePrinter(StrPrinter):
    """Формула в одну строку: y = C·e^(x), |x|, ∫ f dx."""

    def _print_Equality(self, expr):
        return f"{self._print(expr.lhs)} = {self._print(expr.rhs)}"

    def _print_Abs(self, expr):
        return f"|{self._print(expr.args[0])}|"

    def _print_exp(self, expr):
        arg = self._print(expr.args[0])
        return f"e^{arg}" if expr.args[0].is_Atom else f"e^({arg})"

    def _print_Integral(self, expr):
        var = expr.limits[0][0]
        return f"∫ {self._print(expr.function)} d{var}"

    def _print_Exp1(self, expr):
        return "e"


def one_line(expr, ode) -> str:
    text = LinePrinter().doprint(to_display(expr, ode))
    return text.replace("**", "^").replace("*", "·")


def formula_block(expr, ode, indent="    "):
    items = expr if isinstance(expr, (list, tuple)) else [expr]
    lines = []
    for item in items:
        for line in pretty(item, ode).splitlines():
            lines.append(indent + line)
        if len(items) > 1:
            lines.append("")
    return Text("\n".join(lines).rstrip("\n"))


def latex(expr, ode):
    return sp.latex(to_display(expr, ode), ln_notation=True)


# ---------------------------------------------------------------- вывод решения


def print_classification(sol: Solution):
    cls = sol.classification
    body = Text()
    body.append("Тип: ", style="bold")
    body.append(TYPE_NAMES[sol.kind] + "\n", style="bold yellow")
    body.append("Признак: ", style="bold")
    body.append(cls.reasons.get(sol.kind, ""))
    if cls.others:
        body.append("\nТакже подходит: ", style="bold")
        body.append("; ".join(f"{TYPE_NAMES[t]} ({cls.reasons[t]})" for t in cls.others))
        body.append("\nВыбран метод: ", style="bold")
        body.append(TYPE_NAMES[sol.kind])
    console.print(Panel(body, title="Классификация уравнения", border_style="yellow", expand=False))


def print_steps(sol: Solution):
    for i, step in enumerate(sol.steps, 1):
        console.print(Text(f"Шаг {i}. {step.title}", style="bold cyan"))
        if step.expr is not None:
            console.print(formula_block(step.expr, sol.ode))
        if step.note:
            console.print(Text(f"    ⓘ {step.note}", style="italic dim"))
        console.print()


def answer_parts(sol: Solution):
    ode = sol.ode
    parts = []

    def add(title, expr):
        parts.append(Text(title, style="bold"))
        parts.append(formula_block(expr, ode))

    if sol.system is not None:
        add("Нормальная система первого порядка:", sol.system.equations)
        if sol.system.linear:
            kind = "с постоянными коэффициентами" if sol.system.constant_coeffs else "с переменными коэффициентами"
            parts.append(Text(f"Система линейная {kind}; матрица A:"))
            parts.append(formula_block(sol.system.A, ode))
        else:
            parts.append(Text("Система нелинейная — решается численно (--ivp)."))
        if sol.general:
            add("Общее решение исходного уравнения:", sol.general)
        if sol.particular:
            add("Частное решение:", sol.particular)
        return parts

    if sol.general:
        add("Общее решение:", sol.general)
        if sol.nonelementary:
            parts.append(Text("∫ в ответе не выражается в элементарных функциях.", style="italic"))
    elif sol.implicit is not None:
        add("Общий интеграл (y явно не выражается):", sol.implicit)
    else:
        parts.append(Text("Решение не найдено.", style="red"))
    for s, text in sol.singular:
        add(f"Решение g(y) = 0 ({text}):", s)
    if sol.particular:
        x0, y0 = sol.ic
        add(f"Частное решение при {ode.func.__name__}({x0}) = {y0}:", sol.particular)
    elif sol.particular_implicit is not None:
        x0, y0 = sol.ic
        add(f"Частный интеграл при {ode.func.__name__}({x0}) = {y0}:", sol.particular_implicit)
    return parts


def print_answer(sol: Solution):
    console.print(Panel(Group(*answer_parts(sol)), title="Ответ", border_style="green", box=box.DOUBLE, expand=False))


def mark(value):
    return {True: ("✓", "green"), False: ("✗", "red")}.get(value, ("?", "yellow"))


def print_verification(sol: Solution):
    v = sol.verification
    if v is None:
        return
    sign, style = mark(v.ok)
    line = Text("Проверка: ", style="bold")
    line.append(sign, style=f"bold {style}")
    if v.method:
        line.append(f"  ({v.method})", style="dim")
    console.print(line)
    if sol.system is not None:
        if v.note:
            console.print(Text(f"           {v.note}", style="dim"))
        return
    if v.dsolve is not None:
        sign, style = mark(v.equivalent)
        text = {True: "совпадает с точностью до переобозначения C",
                False: "формы различаются", None: "эквивалентность установить не удалось"}[v.equivalent]
        line = Text("Сравнение с sympy.dsolve: ", style="bold")
        line.append(f"{sign} {text}", style=style)
        console.print(line)
        d = v.dsolve if isinstance(v.dsolve, list) else [v.dsolve]
        block = formula_block(d, sol.ode, indent="    ")
        block.stylize("dim")
        console.print(block)
    elif v.note:
        console.print(Text(f"Сравнение с sympy.dsolve: {v.note}", style="yellow"))


def print_latex(sol: Solution):
    ode = sol.ode
    items = []
    if sol.system is not None:
        items += sol.system.equations
    items += sol.general or ([sol.implicit] if sol.implicit is not None else [])
    items += sol.particular
    if sol.particular_implicit is not None:
        items.append(sol.particular_implicit)
    text = "\n".join(latex(e, ode) for e in items)
    console.print(Panel(Text(text), title="LaTeX", border_style="magenta", expand=False))


def print_numeric(sol: Solution):
    num = sol.numeric
    table = Table(title="Численное решение (RK45)", box=box.SIMPLE_HEAVY)
    table.add_column(sol.ode.x.name, justify="right")
    for name in num.names:
        table.add_column(name, justify="right")
    for i, xv in enumerate(num.xs):
        table.add_row(f"{xv:.4g}", *(f"{num.values[k][i]: .6f}" for k in range(len(num.names))))
    console.print(table)
    if not num.success:
        console.print(Text(f"Внимание: интегратор сообщил: {num.message}", style="red"))


def show_solution(sol: Solution, steps=True, show_latex=False):
    console.print(Panel(Text(pretty(sol.ode.eq, sol.ode)), title=f"Уравнение: {sol.ode.text}",
                        border_style="blue", expand=False))
    if steps:
        print_classification(sol)
        print_steps(sol)
    else:
        console.print(Text(f"Тип: {TYPE_NAMES[sol.kind]}", style="bold yellow"))
    print_answer(sol)
    print_verification(sol)
    for note in sol.notes:
        console.print(Text(f"ⓘ {note}", style="italic dim"))
    if sol.numeric is not None:
        print_numeric(sol)
    if show_latex:
        print_latex(sol)


# ---------------------------------------------------------------- сценарии


def run_one(text, *, prefer=None, ic=None, ivp=None, interval=None, plot=None, steps=True,
            show_latex=False, default_ivp=None, compare=True):
    ode = parse_ode(text)
    if ode.order > 1:
        conditions = ivp or ic
        if conditions is None and plot:
            conditions = default_ivp
            if conditions:
                console.print(Text(f"Для графика взяты начальные условия по умолчанию: {conditions}", style="dim"))
        with console.status("Решаю…"):
            sol = solve(ode, ic=conditions, prefer=prefer, compare=compare)
            if conditions:
                solve_numeric(sol, conditions, interval)
    else:
        with console.status("Решаю и проверяю…"):
            sol = solve(ode, ic=ic, prefer=prefer, compare=compare)
            if ivp:
                solve_numeric(sol, ivp, interval)
    show_solution(sol, steps=steps, show_latex=show_latex)
    if plot:
        import plots

        if ode.order > 1 or (ivp and sol.numeric is not None and ode.order == 1 and not sol.general):
            if sol.numeric is None:
                console.print(Text("Для графика системы нужны начальные условия: --ivp \"y(0)=1,y'(0)=0\"", style="red"))
                return sol
            plots.plot_numeric(sol, plot)
        else:
            plots.plot_first_order(sol, plot, interval)
        console.print(Text(f"График сохранён в файл: {plot}", style="bold green"))
    return sol


def run_all(compare=True):
    table = Table(title="Все задачи", box=box.ROUNDED, show_lines=True)
    for col in ("Задача", "Уравнение", "Тип", "Ответ", "Подстановка", "dsolve"):
        table.add_column(col, overflow="fold")
    failures = 0
    start = time.time()
    for p in PROBLEMS:
        with console.status(f"Решаю {p.code}: {p.text}"):
            try:
                sol = solve(p.text, prefer=p.kind, compare=compare)
            except Exception as exc:  # noqa: BLE001 - отчёт должен дойти до конца
                failures += 1
                table.add_row(p.code, p.text, "—", f"ошибка: {exc}", "✗", "")
                continue
        ode = sol.ode
        if sol.system is not None:
            n = sol.system.n
            kind = "линейная" if sol.system.linear else "нелинейная"
            answer = f"система {n}×{n} ({kind}): " + "; ".join(one_line(e, ode) for e in sol.system.equations)
            cmp = "—"
        else:
            items = sol.general or [sol.implicit]
            answer = ";  ".join(one_line(e, ode) for e in items)
            v = sol.verification
            cmp = mark(v.equivalent)[0] if v.dsolve is not None else "—"
        ok = sol.verification.ok if sol.verification else None
        if ok is not True:
            failures += 1
        s, style = mark(ok)
        table.add_row(p.code, p.text, TYPE_NAMES[sol.kind], answer, Text(s, style=style), cmp)
    console.print(table)
    console.print(Text(f"Решено задач: {len(PROBLEMS)}, проблем с проверкой: {failures}, "
                       f"время: {time.time() - start:.1f} с", style="bold"))
    return failures


def problems_table():
    table = Table(title="Набор задач", box=box.SIMPLE)
    table.add_column("Код")
    table.add_column("Блок")
    table.add_column("Уравнение")
    for p in PROBLEMS:
        table.add_row(p.code, p.block_title, p.text)
    console.print(table)


# ---------------------------------------------------------------- интерактивное меню


def ask(prompt, default=None):
    return Prompt.ask(prompt, default=default, console=console, show_default=default not in (None, ""))


def interactive_solve(text, prefer=None, default_ivp=None):
    ode = parse_ode(text)  # ошибки ввода — до остальных вопросов
    ic = ivp = interval = None
    if ode.order == 1:
        ic = ask("Начальное условие, например y(0)=1 (Enter — без него)", "") or None
    else:
        hint = f" (Enter — {default_ivp})" if default_ivp else " (Enter — без численного решения)"
        ivp = ask(f"Начальные условия для численного решения{hint}", "") or default_ivp
        if ivp:
            text_iv = ask("Отрезок a b", "0 10")
            try:
                interval = tuple(float(v) for v in text_iv.split())
                if len(interval) != 2:
                    raise ValueError
            except ValueError:
                raise ParseError("отрезок вводится двумя числами через пробел, например: 0 10") from None
    plot = ask("Сохранить график в файл (например plot.png, Enter — не сохранять)", "") or None
    run_one(text, prefer=prefer, ic=ic, ivp=ivp, interval=interval, plot=plot, default_ivp=default_ivp)


def menu():
    console.print(Panel(Text("Решатель ОДУ первого порядка и сведение ОДУ высших порядков к системам"),
                        border_style="blue"))
    while True:
        console.print(Panel(Text("1 — выбрать задачу из списка (А1…Г4)\n2 — ввести своё уравнение\n"
                                 "3 — решить все задачи подряд (краткий отчёт)\n0 — выход"),
                            title="Меню", expand=False))
        try:
            choice = ask("Ваш выбор").strip()
            if choice == "0":
                break
            if choice == "1":
                problems_table()
                p = find(ask("Номер задачи (например А1 или A1)"))
                interactive_solve(p.text, prefer=p.kind, default_ivp=p.ivp)
            elif choice == "2":
                console.print(Text("Примеры: y' = x/y   x*y' + y = x^2   y'' + 4y = 0   "
                                   "theta'' + sin(theta) = 0   y' = (y/x)*ln(x), x>0", style="dim"))
                interactive_solve(ask("Уравнение"))
            elif choice == "3":
                run_all()
            else:
                console.print(Text("Нет такого пункта — введите 0, 1, 2 или 3.", style="red"))
        except ParseError as exc:
            console.print(Text(f"Ошибка ввода: {exc}", style="bold red"))
        except KeyError as exc:
            console.print(Text(f"Ошибка: {exc.args[0]}", style="bold red"))
        except KeyboardInterrupt:
            console.print(Text("\nПрервано — возвращаемся в меню.", style="yellow"))
        except EOFError:
            break
        except Exception as exc:  # noqa: BLE001 - меню не должно падать
            console.print(Text(f"Не удалось решить: {type(exc).__name__}: {exc}", style="bold red"))
    console.print("До свидания!")


# ---------------------------------------------------------------- аргументы командной строки


def build_parser():
    p = argparse.ArgumentParser(
        description="Пошаговое решение ОДУ первого порядка и сведение ОДУ высших порядков к системам.",
        epilog="Без аргументов запускается интерактивное меню.",
    )
    p.add_argument("equation", nargs="?", help="уравнение, например \"y' = x/y\"")
    p.add_argument("--task", help="номер задачи из набора: А1…Г4 (можно латиницей: A1, B5)")
    p.add_argument("--all", action="store_true", help="решить все задачи и вывести краткий отчёт")
    p.add_argument("--ic", help="начальное условие, например \"y(0)=1\"")
    p.add_argument("--ivp", help="начальные условия для численного решения: \"y(0)=1,y'(0)=0\"")
    p.add_argument("--interval", nargs=2, type=float, metavar=("A", "B"), help="отрезок для численного решения")
    p.add_argument("--plot", metavar="ФАЙЛ.png", help="сохранить график в файл")
    p.add_argument("--no-steps", action="store_true", help="выводить только ответ")
    p.add_argument("--latex", action="store_true", help="дополнительно вывести ответ в LaTeX")
    p.add_argument("--no-compare", action="store_true", help="не сравнивать с sympy.dsolve (быстрее)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    compare = not args.no_compare
    try:
        if args.all:
            return 1 if run_all(compare=compare) else 0
        if args.task:
            p = find(args.task)
            run_one(p.text, prefer=p.kind, ic=args.ic, ivp=args.ivp, interval=args.interval, plot=args.plot,
                    steps=not args.no_steps, show_latex=args.latex, default_ivp=p.ivp, compare=compare)
            return 0
        if args.equation:
            run_one(args.equation, ic=args.ic, ivp=args.ivp, interval=args.interval, plot=args.plot,
                    steps=not args.no_steps, show_latex=args.latex, compare=compare)
            return 0
    except ParseError as exc:
        console.print(Text(f"Ошибка ввода: {exc}", style="bold red"))
        return 2
    except KeyError as exc:
        console.print(Text(f"Ошибка: {exc.args[0]}", style="bold red"))
        return 2
    menu()
    return 0


if __name__ == "__main__":
    sys.exit(main())
