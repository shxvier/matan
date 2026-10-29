"""Консольное приложение: решение линейных уравнений и систем линейных уравнений.

Примеры запуска:
    python linear_solver.py                       # интерактивный режим
    python linear_solver.py "3x - 7 = 2"          # одно уравнение
    python linear_solver.py "2x + y = 5" "x - y = 1"
"""

import sys

import sympy as sp

from common import ParseError, parse_equation, print_block


def check_linear(exprs):
    """Проверяет, что все уравнения линейны, и возвращает список неизвестных."""
    symbols = sorted(set().union(*(e.free_symbols for e in exprs)), key=lambda s: s.name)
    for e in exprs:
        if not symbols:
            break
        try:
            poly = sp.Poly(sp.expand(e), *symbols)
        except sp.PolynomialError:
            raise ValueError(f"уравнение «{e} = 0» не является линейным") from None
        if poly.total_degree() > 1:
            raise ValueError(f"уравнение «{e} = 0» не является линейным")
    return symbols


def solve_system(exprs):
    symbols = check_linear(exprs)
    if not symbols:
        # Уравнения без неизвестных: либо тождество, либо противоречие.
        if all(sp.simplify(e) == 0 for e in exprs):
            print("Каждое уравнение — верное равенство без неизвестных.")
        else:
            print("Есть неверное числовое равенство — решений нет.")
        return

    A, b = sp.linear_eq_to_matrix(exprs, symbols)
    augmented = A.row_join(b)
    print(f"\nНеизвестные: {', '.join(s.name for s in symbols)}")
    print_block("Расширенная матрица системы (A | b):", augmented)

    rref, pivots = augmented.rref()
    print_block("После метода Гаусса (ступенчатый вид):", rref)

    rank_a, rank_ab = A.rank(), augmented.rank()
    print(f"rank A = {rank_a}, rank (A|b) = {rank_ab}, неизвестных: {len(symbols)}")

    if rank_a < rank_ab:
        print("\nРанги различаются (теорема Кронекера–Капелли) — система несовместна, решений нет.")
        return

    solution = sp.linsolve((A, b), *symbols)
    (values,) = solution
    if rank_a == len(symbols):
        print("\nЕдинственное решение:")
    else:
        free = [s.name for i, s in enumerate(symbols) if i not in pivots]
        print(f"\nБесконечно много решений. Свободные неизвестные: {', '.join(free)}")
        print("Общее решение:")
    for s, v in zip(symbols, values):
        print(f"    {s} = {sp.nsimplify(sp.simplify(v))}")
        if v.is_number and not v.is_Integer:
            print(f"       ≈ {sp.N(v, 8)}")


def read_interactive():
    print("Введите уравнения по одному в строке (например: 2x + 3y = 7).")
    print("Пустая строка — решить, «q» — выход.")
    equations = []
    while True:
        try:
            line = input(f"  {len(equations) + 1}) ").strip()
        except EOFError:
            return None if not equations else equations
        if line.lower() in ("q", "quit", "exit", "выход"):
            return None
        if not line:
            return equations
        equations.append(line)


def run(lines):
    try:
        # Десятичные дроби переводим в обыкновенные, чтобы считать точно.
        exprs = [sp.nsimplify(parse_equation(line), rational=True) for line in lines]
        solve_system(exprs)
    except (ParseError, ValueError) as exc:
        print(f"Ошибка: {exc}")


def main():
    if len(sys.argv) > 1:
        run(sys.argv[1:])
        return
    interactive()


def interactive():
    print("=== Решатель линейных уравнений и систем ===")
    while True:
        lines = read_interactive()
        if lines is None:
            print("До свидания!")
            return
        if lines:
            run(lines)
        print()


if __name__ == "__main__":
    main()
