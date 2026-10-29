"""Графики: семейство интегральных кривых и численные решения. Только сохранение в файл."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
from scipy.integrate import cumulative_trapezoid  # noqa: E402

from solver.model import C  # noqa: E402
from solver.util import invariant  # noqa: E402

C_VALUES = np.linspace(-2, 2, 9)
Y_LIMIT = 6.0


def _x_range(sol, interval):
    x = sol.ode.x
    if interval is not None:
        a, b = map(float, interval)
    elif x.is_positive:
        a, b = 0.05, 5.0
    elif x.is_negative:
        a, b = -5.0, -0.05
    else:
        a, b = -5.0, 5.0
    if sol.ic is not None and sol.ode.order == 1:
        x0 = float(sol.ic[0])
        a, b = min(a, x0 - 1), max(b, x0 + 1)
        if x.is_positive:
            a = max(a, 0.01)
    return a, b


def _numeric_function(expr, x, xs):
    """expr(x, C) -> функция от C на сетке xs; неэлементарные ∫ считаются численно."""
    integrals = list(expr.atoms(sp.Integral))
    extra_syms, extra_vals = [], []
    for k, integral in enumerate(integrals):
        J = sp.Symbol(f"J{k}_")
        integrand = integral.function
        if len(integral.limits[0]) == 1:  # неопределённый: первообразная от левого конца
            f = sp.lambdify(x, integrand, "numpy")
            vals = cumulative_trapezoid(np.broadcast_to(f(xs), xs.shape), xs, initial=0.0)
        else:  # ∫ от x0 до x
            t, lo, _ = integral.limits[0]
            f = sp.lambdify(t, integrand, "numpy")
            base = float(lo)
            full = cumulative_trapezoid(np.broadcast_to(f(xs), xs.shape), xs, initial=0.0)
            vals = full - np.interp(base, xs, full)
        expr = expr.xreplace({integral: J})
        extra_syms.append(J)
        extra_vals.append(vals)
    f = sp.lambdify((x, C, *extra_syms), expr, "numpy")

    def evaluate(c):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = f(xs, c, *extra_vals)
        y = np.array(np.broadcast_to(y, xs.shape), dtype=complex)
        y = np.where(np.abs(y.imag) < 1e-9, y.real, np.nan).astype(float)
        y[np.abs(y) > 50 * Y_LIMIT] = np.nan
        # разрываем линию на скачках (асимптоты tg и т. п.)
        jumps = np.abs(np.diff(y)) > Y_LIMIT
        y[1:][jumps] = np.nan
        return y

    return evaluate


def _mask_jumps(Z, threshold=1.0):
    """Убираем разрывы (например arctg(y/x) при x = 0), чтобы contour не рисовал ложных линий."""
    Z = np.array(Z, dtype=float)
    bad = np.zeros_like(Z, dtype=bool)
    jx = np.abs(np.diff(Z, axis=1)) > threshold
    jy = np.abs(np.diff(Z, axis=0)) > threshold
    bad[:, 1:] |= jx
    bad[:, :-1] |= jx
    bad[1:, :] |= jy
    bad[:-1, :] |= jy
    Z[bad] = np.nan
    return Z


def plot_first_order(sol, path, interval=None):
    ode = sol.ode
    x = ode.x
    a, b = _x_range(sol, interval)
    xs = np.linspace(a, b, 800)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")

    if sol.general:
        for branch in sol.general:
            evaluate = _numeric_function(branch.rhs, x, xs)
            for i, c in enumerate(C_VALUES):
                ax.plot(xs, evaluate(c), color=cmap(i / (len(C_VALUES) - 1)), lw=1.2,
                        label=f"C = {c:g}" if branch is sol.general[0] else None)
    elif sol.implicit is not None:
        Y = sp.Symbol("Y_", real=True)
        phi = invariant(sol.implicit, ode.y, Y)
        if phi is not None:
            f = sp.lambdify((x, Y), phi, "numpy")
            ys = np.linspace(-Y_LIMIT, Y_LIMIT, 600)
            XX, YY = np.meshgrid(xs, ys)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ZZ = _mask_jumps(np.broadcast_to(f(XX, YY), XX.shape))
            ax.contour(XX, YY, ZZ, levels=np.sort(C_VALUES), cmap="viridis", linewidths=1.2)
            ax.plot([], [], color=cmap(0.5), label="Φ(x, y) = C, C ∈ [−2; 2]")

    for s, _ in sol.singular:
        f = sp.lambdify(x, s.rhs, "numpy")
        ax.plot(xs, np.broadcast_to(f(xs), xs.shape), "k--", lw=1.5, label=f"особое: y = {sp.sstr(s.rhs)}")

    if sol.ic is not None:
        x0, y0 = float(sol.ic[0]), float(sol.ic[1])
        for k, p in enumerate(sol.particular):
            ys = _numeric_function(p.rhs, x, xs)(0)
            ax.plot(xs, ys, color="crimson", lw=2.8, label="частное решение" if k == 0 else None)
        if sol.particular_implicit is not None:
            Y = sp.Symbol("Y_", real=True)
            expr = (sol.particular_implicit.lhs - sol.particular_implicit.rhs).subs(ode.y, Y)
            f = sp.lambdify((x, Y), expr, "numpy")
            ys = np.linspace(-Y_LIMIT, Y_LIMIT, 600)
            XX, YY = np.meshgrid(xs, ys)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ZZ = _mask_jumps(np.broadcast_to(f(XX, YY), XX.shape))
            if x0 != 0:  # частный интеграл рисуем только в той полуплоскости, где задано НУ
                ZZ[np.sign(XX) != np.sign(x0)] = np.nan
            ax.contour(XX, YY, ZZ, levels=[0], colors="crimson", linewidths=2.8)
            ax.plot([], [], color="crimson", lw=2.8, label="частный интеграл")
        ax.plot([x0], [y0], "o", color="crimson", ms=7, zorder=5)

    ax.set_xlim(a, b)
    ax.set_ylim(-Y_LIMIT, Y_LIMIT)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(x.name)
    ax.set_ylabel(ode.func.__name__)
    ax.set_title(f"Интегральные кривые: {ode.text}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_numeric(sol, path):
    num, ode, res = sol.numeric, sol.ode, sol.system
    n = ode.order
    xname = ode.x.name
    two = n == 2
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(13 if two else 8, 5.5))
    ax = axes[0] if two else axes
    for k in range(n):
        ax.plot(num.dense_x, num.dense_values[k], lw=2 if k == 0 else 1.2,
                label=f"{num.names[k]}({xname})", ls="-" if k == 0 else "--")
    ax.plot(num.xs, num.values[0], "o", ms=4, color="C0")
    ax.set_xlabel(xname)
    ax.set_title(f"Численное решение: {ode.text}")
    ax.grid(alpha=0.3)
    ax.legend()

    if two:
        ph = axes[1]
        y1, y2 = num.dense_values
        ph.plot(y1, y2, color="crimson", lw=2, label="траектория")
        ph.plot([y1[0]], [y2[0]], "o", color="crimson")
        autonomous = not any(r.has(ode.x) for r in res.rhs)
        if autonomous:
            pad1 = max(1.0, 0.3 * (y1.max() - y1.min()))
            pad2 = max(1.0, 0.3 * (y2.max() - y2.min()))
            g1 = np.linspace(y1.min() - pad1, y1.max() + pad1, 30)
            g2 = np.linspace(y2.min() - pad2, y2.max() + pad2, 30)
            G1, G2 = np.meshgrid(g1, g2)
            f = sp.lambdify(res.symbols, res.rhs, "numpy")
            U, V = f(G1, G2)
            U = np.broadcast_to(U, G1.shape)
            V = np.broadcast_to(V, G1.shape)
            ph.streamplot(G1, G2, U, V, color="lightsteelblue", density=1.1, linewidth=0.8)
            ph.set_title("Фазовый портрет (поле направлений системы)")
        else:
            ph.set_title("Фазовая траектория (система неавтономная)")
        ph.set_xlabel(num.names[0])
        ph.set_ylabel(num.names[1])
        ph.grid(alpha=0.3)
        ph.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
