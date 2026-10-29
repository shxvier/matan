import numpy as np
import pytest
import sympy as sp

from solver import solve, solve_numeric, solve_problem
from solver.classify import classify
from solver.parser import parse_ode
from solver.problems import PROBLEMS, find

FIRST_ORDER = [p for p in PROBLEMS if p.kind != "system"]
SYSTEMS = [p for p in PROBLEMS if p.kind == "system"]


def test_there_are_26_problems():
    assert len(PROBLEMS) == 26


@pytest.mark.parametrize("problem", PROBLEMS, ids=[p.code for p in PROBLEMS])
def test_type_is_detected(problem):
    cls = classify(parse_ode(problem.text))
    assert problem.kind in cls.types


@pytest.mark.parametrize("problem", FIRST_ORDER, ids=[p.code for p in FIRST_ORDER])
def test_solution_passes_checkodesol(problem):
    sol = solve_problem(problem.code, compare=False)
    assert sol.kind == problem.kind
    assert sol.steps, "должен быть ход решения"
    eq = sol.ode.eq
    if sol.general:
        for g in sol.general:
            assert sp.checkodesol(eq, g)[0], f"{problem.code}: {g}"
    else:
        assert sol.implicit is not None
        assert sp.checkodesol(eq, sol.implicit, solve_for_func=False)[0]
    assert sol.verification.ok is True


@pytest.mark.parametrize("problem", SYSTEMS, ids=[p.code for p in SYSTEMS])
def test_reduction_to_system(problem):
    sol = solve_problem(problem.code)
    res = sol.system
    order = sol.ode.order
    assert res.n == order and len(res.equations) == order and len(res.rhs) == order
    # y_k' = y_{k+1} для всех, кроме последнего уравнения
    for k in range(order - 1):
        assert res.rhs[k] == res.symbols[k + 1]
    assert sol.verification.ok is True


def test_system_matrices():
    g1 = solve_problem("Г1").system
    assert g1.linear and g1.constant_coeffs
    assert g1.A == sp.Matrix([[0, 1], [-4, 0]])
    g2 = solve_problem("Г2").system
    assert g2.A == sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    g3 = solve_problem("Г3").system
    x = solve_problem("Г3").ode.x
    assert g3.A == sp.Matrix([[0, 1], [-5, -2]])
    assert g3.F == sp.Matrix([0, sp.sin(x)])
    g4 = solve_problem("Г4").system
    assert not g4.linear


def test_numeric_matches_exact_solution():
    sol = solve_problem("Г1")
    num = solve_numeric(sol, "y(0)=1, y'(0)=0", (0, 5))
    assert np.allclose(num.values[0], np.cos(2 * num.xs), atol=1e-6)
    assert np.allclose(num.values[1], -2 * np.sin(2 * num.xs), atol=1e-6)


def test_pendulum_energy_is_conserved():
    sol = solve_problem("Г4")
    num = solve_numeric(sol, "theta(0)=1, theta'(0)=0", (0, 10))
    theta, omega = num.dense_values
    energy = omega**2 / 2 - np.cos(theta)
    assert np.ptp(energy) < 1e-6


def test_initial_condition_separable():
    sol = solve_problem("А1", ic="y(0)=1")
    x = sol.ode.x
    assert len(sol.particular) == 1
    assert sp.simplify(sol.particular[0].rhs - sp.sqrt(x**2 + 1)) == 0


def test_initial_condition_linear_nonelementary():
    sol = solve_problem("Б6", ic="y(0)=1")
    assert sol.nonelementary
    part = sol.particular[0]
    assert part.rhs.has(sp.Integral)
    assert sp.checkodesol(sol.ode.eq, part)[0]
    assert part.rhs.subs(sol.ode.x, 0).doit() == 1


def test_implicit_answer_for_v2():
    sol = solve_problem("В2", compare=False)
    assert not sol.general and sol.implicit is not None


def test_singular_solution_reported():
    sol = solve("y' = y^2 - 1", compare=False)
    values = sorted(s.rhs for s, _ in sol.singular)
    assert values == [-1, 1]


def test_dsolve_comparison():
    sol = solve_problem("А3")
    assert sol.verification.equivalent is True


def test_other_types_listed_and_simplest_chosen():
    sol = solve("y' = y/x", compare=False)
    assert sol.kind == "separable"
    assert set(sol.classification.others) == {"linear", "homogeneous"}
    assert solve_problem("В1", compare=False).kind == "homogeneous"


@pytest.mark.parametrize("code,expected", [("B5", "Б5"), ("A1", "А1"), ("б5", "Б5"), ("g4", "Г4"), ("В 2", "В2")])
def test_task_codes_latin_and_cyrillic(code, expected):
    assert find(code).code == expected


def test_unknown_task():
    with pytest.raises(KeyError):
        find("Д1")
