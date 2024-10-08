"""
Tests for functions in class SolveDiffusion2D
"""
import unittest

import numpy as np
import pytest

from diffusion2d import SolveDiffusion2D


class TestDiffusion2D(unittest.TestCase):
    def setUp(self):
        self.w = 1.0
        self.h = 2.0
        self.dx = 0.19
        self.dy = 0.09
        self.d = 8.
        self.T_hot = 1000.
        self.T_cold = 100.
        self.solver = SolveDiffusion2D()

    def test_initialize_domain(self):
        """
        Check function SolveDiffusion2D.initialize_domain
        """

        self.solver.initialize_domain(self.w, self.h, self.dx, self.dy)

        assert (self.solver.w == 1.0)
        assert (self.solver.h == 2.0)
        assert (self.solver.dx == 0.19)
        assert (self.solver.dy == 0.09)
        assert (self.solver.nx == 5)
        assert (self.solver.ny == 22)

    def test_initialize_physical_parameters(self):
        """
        Checks function SolveDiffusion2D.initialize_domain
        """
        self.solver = SolveDiffusion2D()
        self.solver.dx = self.dx
        self.solver.dy = self.dy
        self.solver.initialize_physical_parameters(self.d, self.T_cold, self.T_hot)

        assert (self.solver.D == 8.)
        assert (self.solver.T_hot == 1000.)
        assert (self.solver.T_cold == 100.)
        assert (self.solver.dt == pytest.approx(0.0003125))

    def test_set_initial_condition(self):
        """
        Checks function SolveDiffusion2D.get_initial_function
        """
        self.solver = SolveDiffusion2D()
        self.solver.T_hot = self.T_hot
        self.solver.T_cold = self.T_cold
        self.solver.dx = self.dx
        self.solver.dy = self.dy
        self.solver.nx = int(self.w / self.dx)
        self.solver.ny = int(self.h / self.dy)
        self.solver.h = self.h
        self.solver.w = self.w

        u = self.solver.T_cold * np.ones((self.solver.nx, self.solver.ny))
        r = min(self.solver.h, self.solver.w) / 4.0
        cx = self.solver.w / 2.0
        cy = self.solver.h / 2.0
        r2 = r ** 2
        for i in range(self.solver.nx):
            for j in range(self.solver.ny):
                p2 = (i * self.solver.dx - cx) ** 2 + (j * self.solver.dy - cy) ** 2
                if p2 < r2:
                    u[i, j] = self.solver.T_hot

        u_solv = self.solver.set_initial_condition()
        assert (np.allclose(u, u_solv))
