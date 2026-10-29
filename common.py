"""Общие функции: проверка и разбор математических выражений, введённых пользователем.

Все ошибки ввода превращаются в ParseError с понятным русским сообщением,
а где возможно — с указанием места ошибки:

    Ошибка ввода: не закрыта скобка «(»
        2*(x + 1
          ^
"""

import difflib
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Имена, которые пользователь может написать «по-школьному».
BASE_NAMES = {
    "e": sp.E,
    "pi": sp.pi,
    "oo": sp.oo,
    "inf": sp.oo,
    "ln": sp.log,
    "lg": lambda t: sp.log(t, 10),
    "tg": sp.tan,
    "ctg": sp.cot,
    "arctg": sp.atan,
    "arcctg": sp.acot,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "sh": sp.sinh,
    "ch": sp.cosh,
    "th": sp.tanh,
    "cth": sp.coth,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}

# Функции, которые чаще всего пишут без скобок: sinx, ln2, sqrtx.
MATH_FUNCTIONS = [
    "arcsin", "arccos", "arctg", "arcctg", "asin", "acos", "atan", "acot",
    "sinh", "cosh", "tanh", "coth", "sin", "cos", "tan", "cot", "tg", "ctg",
    "sqrt", "exp", "log", "ln", "lg", "abs", "sh", "ch", "th", "cth",
]

# Похожие русские буквы, которые часто попадают в формулы вместо латинских.
CYRILLIC_LOOKALIKES = {
    "х": "x", "у": "y", "а": "a", "с": "c", "е": "e", "о": "o", "р": "p",
    "к": "k", "т": "t", "м": "m", "в": "b", "н": "h",
}

UNICODE_REPLACEMENTS = {
    "−": "-", "–": "-", "—": "-", "×": "*", "·": "*", "∙": "*", "÷": "/",
    "π": "pi", "∞": "oo", "²": "^2", "³": "^3", "≤": "<=", "≥": ">=", "≠": "!=",
}

ALLOWED_CHARS = re.compile(r"[A-Za-z0-9_\s+\-*/^().,!=<>'\[\]\"]")


class ParseError(ValueError):
    """Ошибка во вводе пользователя."""


def error_at(message, text, pos):
    """Сообщение об ошибке со стрелкой под нужным символом."""
    return ParseError(f"{message}\n    {text}\n    {' ' * pos}^")


# ---------------------------------------------------------------- подготовка текста


def preprocess(text, warnings=None, fix_cyrillic=True):
    """Заменяет «красивые» символы на обычные и исправляет русские буквы-двойники."""
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"√\s*\(", "sqrt(", text)
    text = re.sub(r"√\s*([A-Za-z_]\w*|\d+(?:\.\d+)?)", r"sqrt(\1)", text)
    # |x - 1| -> Abs(x - 1) (без вложенных модулей).
    text = re.sub(r"\|([^|]+)\|", r"Abs(\1)", text)

    def fix_letter(m):
        letter = m.group(0)
        latin = CYRILLIC_LOOKALIKES[letter.lower()]
        if warnings is not None:
            warnings.append(f"русская буква «{letter}» заменена на латинскую «{latin}»")
        return latin

    if not fix_cyrillic:
        return text
    # Только одиночные буквы: «х» в «2х+1», но не в слове «при».
    pattern = "(?<![а-яёА-ЯЁ])[" + "".join(CYRILLIC_LOOKALIKES) + "](?![а-яёА-ЯЁ])"
    text = re.sub(pattern, fix_letter, text, flags=re.IGNORECASE)
    return text


def convert_commas(text):
    """Запятая внутри вызова функции — разделитель аргументов, иначе — десятичная."""
    out, stack = [], []
    for i, ch in enumerate(text):
        if ch in "([":
            before = text[:i].rstrip()
            is_call = ch == "[" or (before and (before[-1].isalnum() or before[-1] == "_"))
            stack.append(is_call)
        elif ch in ")]" and stack:
            stack.pop()
        elif ch == "," and not any(stack):
            prev, nxt = text[i - 1 : i], text[i + 1 : i + 2]
            if prev.isdigit() and nxt.isdigit():
                ch = "."
            else:
                raise error_at(
                    "лишняя запятая. Десятичные дроби пишите так: 0.5 или 0,5 (без пробелов)", text, i
                )
        out.append(ch)
    return "".join(out)


def check_syntax(text, known_names):
    """Ищет типичные ошибки до разбора, чтобы объяснить их понятно."""
    if not text.strip():
        raise ParseError("пустое выражение")

    for i, ch in enumerate(text):
        if not ALLOWED_CHARS.fullmatch(ch):
            if re.fullmatch(r"[а-яёА-ЯЁ]", ch):
                word = re.search(r"[а-яёА-ЯЁ]+", text[i:]).group(0)
                raise error_at(
                    f"непонятное слово «{word}». Переменные и функции пишутся латиницей "
                    "(x, y, sin, ln), а команды — в начале строки (help — список команд)",
                    text,
                    i,
                )
            raise error_at(f"недопустимый символ «{ch}»", text, i)

    # Скобки
    stack = []
    pairs = {")": "(", "]": "["}
    for i, ch in enumerate(text):
        if ch in "([":
            stack.append((ch, i))
        elif ch in ")]":
            if not stack:
                raise error_at(f"лишняя закрывающая скобка «{ch}»", text, i)
            opened, pos = stack.pop()
            if opened != pairs[ch]:
                raise error_at(f"скобка «{opened}» закрыта скобкой «{ch}»", text, i)
    if stack:
        opened, pos = stack[-1]
        raise error_at(f"не закрыта скобка «{opened}»", text, pos)

    checks = [
        (r"\(\s*\)", "пустые скобки — внутри должно быть выражение"),
        (r"^\s*[*/^,]", "выражение начинается со знака «{0}»"),
        (r"[+\-*/^,]\s*$", "выражение обрывается на знаке «{0}» — после него не хватает числа или переменной"),
        (r"\(\s*[*/^,]", "после «(» сразу стоит знак «{0}»"),
        (r"[+\-*/^,](?=\s*\))", "перед «)» стоит знак «{0}» — не хватает числа или переменной"),
        (r"\d+\.\d*\.", "в числе две десятичные точки"),
    ]
    for pattern, message in checks:
        m = re.search(pattern, text)
        if m:
            op = re.search(r"[+\-*/^,]", m.group(0))
            raise error_at(message.format(op.group(0) if op else ""), text, m.start())

    # Два знака подряд: 2+*3, 2^/3. «**» — это степень, «*-», «^-» — минус у числа.
    ops = re.sub(r"\*\*", "^ ", text)  # длина строки сохраняется
    m = re.search(r"[+\-*/^]\s*[*/^]", ops)
    if m:
        raise error_at(f"два знака операции подряд: «{text[m.start():m.end()].replace(' ', '')}»", text, m.start())

    # Имена функций: sinx, sinn(x), cos^2(x)
    for m in re.finditer(r"[A-Za-z_]\w*", text):
        name = m.group(0)
        if text[: m.start()].rstrip().endswith(".") or name in known_names:
            continue
        after = text[m.end() :].lstrip()
        if after.startswith("(") and len(name) > 1 and not is_symbol_product(name):
            guess = difflib.get_close_matches(name, list(known_names), n=1, cutoff=0.6)
            hint = f" Возможно, вы имели в виду «{guess[0]}»?" if guess else ""
            raise error_at(f"неизвестная функция «{name}».{hint}", text, m.start())
        for func in MATH_FUNCTIONS:
            if func in ("sh", "ch", "th", "cth"):
                continue  # слишком короткие: theta, chi — не ошибки
            if name.startswith(func) and len(name) > len(func) and name not in known_names:
                rest = name[len(func) :]
                raise error_at(
                    f"«{name}»: аргумент функции пишется в скобках — {func}({rest}) или {func} {rest}",
                    text,
                    m.start(),
                )


def is_symbol_product(name):
    """'xy', 'ab' — произведение однобуквенных переменных, а не функция."""
    return len(name) <= 3 and not any(name.startswith(f) for f in MATH_FUNCTIONS) and name.isalpha()


def known_function_names(extra):
    names = set(BASE_NAMES) | set(extra or {})
    names |= {n for n in dir(sp) if not n.startswith("_") and callable(getattr(sp, n))}
    return names


# ---------------------------------------------------------------- разбор


def parse(text, extra_names=None):
    """Разбирает выражение вида '2x + 3sin(x)', 'x^2', 'e^(3x)', '0,5' в объект SymPy."""
    warnings = []
    text = preprocess(text.strip(), warnings)
    for w in dict.fromkeys(warnings):
        print(f"(Замечание: {w})")
    names = dict(BASE_NAMES)
    if extra_names:
        names.update(extra_names)
    check_syntax(text, known_function_names(names))
    text = convert_commas(text)
    # Имена вида x1, x2, a10 — отдельные переменные, а не x*1.
    for name in re.findall(r"(?<![A-Za-z_])[A-Za-z]\d+", text):
        names.setdefault(name, sp.Symbol(name))
    # cos^2(x) -> cos(x)^2
    funcs = "|".join(MATH_FUNCTIONS)
    text = re.sub(rf"\b({funcs})\s*\^\s*(\d+)\s*\(([^()]*)\)", r"\1(\3)^\2", text)

    try:
        value = parse_expr(text, local_dict=names, transformations=TRANSFORMATIONS)
    except TypeError as exc:
        msg = str(exc)
        if "argument" in msg:
            raise ParseError(
                f"неверное число аргументов функции в «{text}». "
                "Если это десятичная дробь, пишите через точку: sin(0.5)"
            ) from None
        raise ParseError(f"не удалось разобрать «{text}»: проверьте знаки операций") from None
    except ZeroDivisionError:
        raise ParseError("деление на ноль") from None
    except Exception:  # noqa: BLE001 - любые ошибки парсера сводим к одной
        raise ParseError(f"не удалось разобрать «{text}». Проверьте знаки операций и скобки") from None

    if isinstance(value, sp.Basic):
        if value.has(sp.zoo):
            raise ParseError("деление на ноль")
        if value is sp.nan:
            raise ParseError("неопределённость (например, 0/0 или ∞ - ∞)")
    return value


def split_equation(text):
    """Делит 'левая = правая' на две части с проверкой знака «=»."""
    parts = re.split(r"(?<![<>!=:])=(?!=)", text)
    if len(parts) != 2:
        raise ParseError("в уравнении должен быть ровно один знак «=»")
    left, right = (p.strip() for p in parts)
    if not left:
        raise ParseError("пустая левая часть уравнения (перед «=»)")
    if not right:
        raise ParseError("пустая правая часть уравнения (после «=»)")
    return left, right


def parse_equation(text, extra_names=None):
    """Разбирает уравнение 'левая = правая' и возвращает выражение 'левая - правая'."""
    left, right = split_equation(text)
    return parse(left, extra_names) - parse(right, extra_names)


# ---------------------------------------------------------------- вывод

RU_FUNCTIONS = {
    sp.log: "ln", sp.tan: "tg", sp.cot: "ctg", sp.atan: "arctg", sp.acot: "arcctg",
    sp.asin: "arcsin", sp.acos: "arccos",
}


def russify(expr):
    """Заменяет имена функций на привычные: log -> ln, tan -> tg, atan -> arctg."""
    if not isinstance(expr, sp.Basic):
        return expr
    for func, name in RU_FUNCTIONS.items():
        expr = expr.replace(func, sp.Function(name))
    return expr


def show(expr):
    """Красивый многострочный вывод выражения."""
    return sp.pretty(russify(expr), use_unicode=True)


def print_block(title, expr, indent="    "):
    lines = show(expr).splitlines()
    if len(lines) == 1:
        print(f"{title} {lines[0]}")
        return
    print(title)
    for line in lines:
        print(indent + line)
