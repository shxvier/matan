"""Оформление хода решения: формулы в строку, шаги, множества."""

import re
from contextlib import contextmanager

import sympy as sp
from sympy.printing.str import StrPrinter

from common import print_block, russify

SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


class RuPrinter(StrPrinter):
    """Печать формул в одну строку в привычной записи: x², e^(2x), ln, tg, √, π, ∞."""

    def _print_Exp1(self, expr):
        return "e"

    def _print_Pi(self, expr):
        return "π"

    def _print_Infinity(self, expr):
        return "∞"

    def _print_NegativeInfinity(self, expr):
        return "-∞"

    def _print_ComplexInfinity(self, expr):
        return "∞"

    def _print_ImaginaryUnit(self, expr):
        return "i"

    def _print_exp(self, expr):
        arg = self._print(expr.args[0])
        return f"e^{arg}" if expr.args[0].is_Atom and not str(arg).startswith("-") else f"e^({arg})"

    def _print_Function(self, expr):
        if expr.func.__name__ == "ln" and isinstance(expr.args[0], sp.Abs):
            return "ln" + self._print(expr.args[0])
        return super()._print_Function(expr)

    def _print_Abs(self, expr):
        return f"|{self._print(expr.args[0])}|"

    def _print_Pow(self, expr, rational=False):
        base, exp = expr.args
        if exp == sp.Rational(1, 2):
            return f"√({self._print(base)})" if not base.is_Atom else f"√{self._print(base)}"
        if exp == -sp.Rational(1, 2):
            inner = f"√({self._print(base)})" if not base.is_Atom else f"√{self._print(base)}"
            return f"1/{inner}"
        if exp.is_Integer and exp < 0:
            return "1/" + self.parenthesize(base ** (-exp), sp.printing.precedence.PRECEDENCE["Mul"] + 1)
        if exp.is_Integer and exp > 1:
            b = self.parenthesize(base, sp.printing.precedence.PRECEDENCE["Pow"])
            return b + str(exp).translate(SUPERSCRIPT)
        return super()._print_Pow(expr, rational)

    def _print_Symbol(self, expr):
        return "t" if expr.name in ("_u", "_t") else super()._print_Symbol(expr)

    _print_Dummy = _print_Symbol


def fmt(expr):
    """Формула в одну строку."""
    if isinstance(expr, (list, tuple)):
        return ", ".join(fmt(e) for e in expr)
    if not isinstance(expr, sp.Basic):
        return str(expr)
    text = RuPrinter().doprint(russify(expr))
    text = text.replace("**", "^").replace("*", "·")
    return text


def needs_approx_value(value):
    """Нужно ли десятичное приближение: для 7/3, √2, π — да, для 5, 2 + 3i, 0.5 — нет."""
    if not (isinstance(value, sp.Basic) and value.is_number and value.is_finite):
        return False
    if value.atoms(sp.Float):
        return False
    return not all(part.is_Integer for part in value.as_real_imag())


def fmt_number(value):
    """Число: точно, а если нужно — и приближённо."""
    text = fmt(value)
    if needs_approx_value(value):
        text += f" ≈ {approx(value)}"
    return text


def format_float(value):
    """0.300000000000000 -> 0.3; длинные числа (N(pi, 50)) выводятся целиком."""
    if value._prec > 53:
        return str(value)
    return f"{float(value):.15g}"


def approx(value):
    re_part, im_part = (sp.N(p, 15) for p in value.as_real_imag())
    re_s, im_s = format_float(re_part), format_float(abs(im_part))
    if im_part == 0:
        return re_s
    sign = "-" if im_part < 0 else "+"
    if re_part == 0:
        return f"{'-' if sign == '-' else ''}{im_s}i"
    return f"{re_s} {sign} {im_s}i"


# ---------------------------------------------------------------- множества


def fmt_interval(iv):
    left = "(" if iv.left_open or iv.start == -sp.oo else "["
    right = ")" if iv.right_open or iv.end == sp.oo else "]"
    start = "-∞" if iv.start == -sp.oo else fmt(iv.start)
    end = "+∞" if iv.end == sp.oo else fmt(iv.end)
    return f"{left}{start}; {end}{right}"


def fmt_set(s, var="x"):
    """Множество решений в школьной записи."""
    if s is sp.S.EmptySet:
        return "решений нет"
    if s == sp.S.Reals:
        return "(-∞; +∞) — любое число"
    if isinstance(s, sp.Interval):
        return fmt_interval(s)
    if isinstance(s, sp.FiniteSet):
        return "{" + "; ".join(fmt(e) for e in sorted_values(s)) + "}"
    if isinstance(s, sp.Union):
        if all(isinstance(a, sp.ImageSet) for a in s.args):
            return ";  ".join(fmt_set(a, var) for a in s.args)
        return " ∪ ".join(fmt_set(a, var) for a in s.args)
    if isinstance(s, sp.ImageSet) and s.base_sets[0] == sp.S.Integers:
        n = s.lamda.variables[0]
        k = sp.Symbol("k")
        return f"{var} = {fmt(s.lamda.expr.subs(n, k))}, k ∈ ℤ"
    if isinstance(s, sp.Complement) and isinstance(s.args[1], sp.FiniteSet):
        return f"{fmt_set(s.args[0], var)}, кроме {fmt_set(s.args[1], var)}"
    return fmt(s)


def sorted_values(values):
    def key(v):
        try:
            return (0, float(v))
        except TypeError:
            return (1, str(v))

    return sorted(values, key=key)


# ---------------------------------------------------------------- шаги решения


class Log:
    """Печать хода решения с нумерацией и вложенными подшагами."""

    def __init__(self):
        self.level = 0
        self.counter = 0

    @property
    def pad(self):
        return "   " * self.level

    def title(self, text):
        print(f"\n▶ {text}")

    def step(self, text):
        if self.level == 0:
            self.counter += 1
            print(f"Шаг {self.counter}. {text}")
        else:
            print(f"{self.pad}• {text}")

    def line(self, text):
        print(f"{self.pad}    {text}")

    def block(self, title, expr):
        print_block(f"{self.pad}    {title}", expr, indent=self.pad + "        ")

    def note(self, text):
        print(f"{self.pad}  ({text})")

    def answer(self, text):
        print(f"\nОтвет: {text}")

    def answer_block(self, title, expr):
        print()
        print_block(f"Ответ: {title}", expr)

    @contextmanager
    def sub(self):
        self.level += 1
        try:
            yield
        finally:
            self.level -= 1


def choose_variable(expr, given=None):
    """Переменная по умолчанию: указанная, иначе x, иначе единственная."""
    if given is not None:
        return given
    symbols = sorted(expr.free_symbols, key=lambda s: s.name) if isinstance(expr, sp.Basic) else []
    names = [s.name for s in symbols]
    if "x" in names or not symbols:
        return sp.Symbol("x")
    if len(symbols) == 1:
        return symbols[0]
    from common import ParseError

    raise ParseError(f"в выражении несколько переменных ({', '.join(names)}) — укажите нужную: «... по {names[0]}»")


def strip_trailing(text, pattern):
    """Отрезает от конца строки фрагмент по шаблону и возвращает (остаток, совпадение)."""
    m = re.search(pattern, text)
    if not m:
        return text, None
    return (text[: m.start()] + text[m.end() :]).strip(), m
