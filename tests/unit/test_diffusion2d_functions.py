"""
Tests for functions in class SolveDiffusion2D
"""
import numpy as np
import pytest

from diffusion2d import SolveDiffusion2D


def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    w = 1.0
    h = 2.0
    dx = 0.19
    dy = 0.09
    solver.initialize_domain(w, h, dx, dy)

    assert (solver.w == 1.0)
    assert (solver.h == 2.0)
    assert (solver.dx == 0.19)
    assert (solver.dy == 0.09)
    assert (solver.nx == 5)
    assert (solver.ny == 22)


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    d = 8.
    T_hot = 1000.
    T_cold = 100.

    solver.dx = 0.1
    solver.dy = 0.1
    solver.initialize_physical_parameters(d, T_cold, T_hot)

    assert (solver.D == 8.)
    assert (solver.T_hot == 1000.)
    assert (solver.T_cold == 100.)
    assert (solver.dt == pytest.approx(0.0003125))


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    solver.T_hot = 1000.
    solver.T_cold = 100.
    solver.dx = 0.1
    solver.dy = 0.1
    solver.nx = 20
    solver.ny = 20
    solver.h = 10.
    solver.w = 10.

    u = solver.T_cold * np.ones((solver.nx, solver.ny))
    r = min(solver.h, solver.w) / 4.0
    cx = solver.w / 2.0
    cy = solver.h / 2.0
    r2 = r ** 2
    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                u[i, j] = solver.T_hot

    u_solv = solver.set_initial_condition()
    assert (np.allclose(u, u_solv))
