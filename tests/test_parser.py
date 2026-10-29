import pytest
import sympy as sp

from solver.parser import ParseError, parse_conditions, parse_ode


def rhs_of(text):
    ode = parse_ode(text)
    return ode, sp.simplify(ode.eq.lhs - ode.eq.rhs)


def test_implicit_multiplication():
    ode, expr = rhs_of("y' = 2xy")
    x, y = ode.x, ode.y
    assert sp.simplify(expr - (y.diff(x) - 2 * x * y)) == 0


def test_number_times_function():
    ode, expr = rhs_of("y'' + 4y = 0")
    x, y = ode.x, ode.y
    assert ode.order == 2
    assert sp.simplify(expr - (y.diff(x, 2) + 4 * y)) == 0


def test_power_with_caret_and_parentheses_after_function():
    ode, expr = rhs_of("y' = x^2/(y(1+x^3))")
    x, y = ode.x, ode.y
    assert sp.simplify(expr - (y.diff(x) - x**2 / (y * (1 + x**3)))) == 0


def test_e_power_and_ln():
    ode, expr = rhs_of("y' = e^x * ln(x) + e^(2x)")
    x, y = ode.x, ode.y
    assert sp.simplify(expr - (y.diff(x) - sp.exp(x) * sp.log(x) - sp.exp(2 * x))) == 0


def test_coefficient_before_derivative():
    ode, expr = rhs_of("x*y' + y = x^2")
    x, y = ode.x, ode.y
    assert sp.simplify(expr - (x * y.diff(x) + y - x**2)) == 0


def test_greek_function_name():
    ode, expr = rhs_of("theta'' + sin(theta) = 0")
    assert ode.func.__name__ == "theta"
    assert ode.order == 2


def test_domain_makes_x_positive():
    ode = parse_ode("y' = (y/x)*ln(x), x>0")
    assert ode.x.is_positive
    assert ode.domain == (ode.x > 0)


def test_tan_and_dy_dx():
    ode, expr = rhs_of("dy/dx + y*tan(x) = sin(2x)")
    x, y = ode.x, ode.y
    assert sp.simplify(expr - (y.diff(x) + y * sp.tan(x) - sp.sin(2 * x))) == 0


@pytest.mark.parametrize("bad", ["y = x", "y' = x +", "y' = a*y", "y' = sinn(x)", "y' = (x", "y' = x = 1", ""])
def test_bad_input_gives_parse_error(bad):
    with pytest.raises(ParseError):
        parse_ode(bad)


def test_conditions():
    ode = parse_ode("theta'' + sin(theta) = 0")
    conds = parse_conditions("theta(0)=1, theta'(0)=0", ode)
    assert conds == {0: (0, 1), 1: (0, 0)}
    with pytest.raises(ParseError):
        parse_conditions("theta(0)=1", ode)
