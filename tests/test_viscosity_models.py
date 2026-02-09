"""
Tests for viscosity models.
"""

import pytest
import numpy as np
from src.models.viscosity import (
    walther_equation, andrade_equation, beggs_robinson_dead_oil,
    fit_walther_parameters, fit_andrade_parameters,
    WaltherViscosity, AndradeViscosity, BeggsRobinsonViscosity
)


class TestWaltherEquation:
    """Test Walther viscosity model."""

    def test_walther_basic(self):
        """Test basic Walther equation calculation."""
        # Typical values for medium crude
        A, B = 10.0, 3.5
        T = 313.15  # 40°C
        
        nu = walther_equation(T, A, B)
        assert nu > 0
        assert isinstance(nu, (float, np.floating))

    def test_walther_temperature_dependence(self):
        """Test that viscosity decreases with temperature."""
        A, B = 10.0, 3.5
        T1 = 293.15  # 20°C
        T2 = 353.15  # 80°C
        
        nu1 = walther_equation(T1, A, B)
        nu2 = walther_equation(T2, A, B)
        
        assert nu2 < nu1, "Viscosity should decrease with temperature"

    def test_walther_parameter_fitting(self):
        """Test parameter fitting from two points."""
        T1, nu1 = 293.15, 50e-6  # 20°C, 50 cSt
        T2, nu2 = 373.15, 10e-6  # 100°C, 10 cSt
        
        A, B = fit_walther_parameters(T1, nu1, T2, nu2)
        
        # Check that fitted equation reproduces original points
        nu1_calc = walther_equation(T1, A, B)
        nu2_calc = walther_equation(T2, A, B)
        
        np.testing.assert_allclose(nu1, nu1_calc, rtol=0.01)
        np.testing.assert_allclose(nu2, nu2_calc, rtol=0.01)


class TestAndradeEquation:
    """Test Andrade viscosity model."""

    def test_andrade_basic(self):
        """Test basic Andrade equation calculation."""
        A = 1e-4  # Pa·s
        B = 1000  # K
        T = 313.15  # 40°C
        
        mu = andrade_equation(T, A, B)
        assert mu > 0
        assert isinstance(mu, (float, np.floating))

    def test_andrade_temperature_dependence(self):
        """Test that viscosity decreases with temperature."""
        A, B = 1e-4, 1000
        T1 = 293.15
        T2 = 353.15
        
        mu1 = andrade_equation(T1, A, B)
        mu2 = andrade_equation(T2, A, B)
        
        assert mu2 < mu1, "Viscosity should decrease with temperature"

    def test_andrade_parameter_fitting(self):
        """Test parameter fitting from two points."""
        T1, mu1 = 293.15, 0.050  # 20°C, 50 cP
        T2, mu2 = 373.15, 0.010  # 100°C, 10 cP
        
        A, B = fit_andrade_parameters(T1, mu1, T2, mu2)
        
        # Check that fitted equation reproduces original points
        mu1_calc = andrade_equation(T1, A, B)
        mu2_calc = andrade_equation(T2, A, B)
        
        np.testing.assert_allclose(mu1, mu1_calc, rtol=0.01)
        np.testing.assert_allclose(mu2, mu2_calc, rtol=0.01)


class TestBeggsRobinson:
    """Test Beggs-Robinson correlation."""

    def test_beggs_robinson_basic(self):
        """Test basic Beggs-Robinson calculation."""
        api = 30.0  # Medium crude
        T = 313.15  # 40°C (104°F)
        
        mu = beggs_robinson_dead_oil(T, api)
        assert mu > 0
        assert isinstance(mu, (float, np.floating))

    def test_beggs_robinson_api_dependence(self):
        """Test that heavier oils have higher viscosity."""
        T = 313.15
        api_light = 35.0
        api_heavy = 20.0
        
        mu_light = beggs_robinson_dead_oil(T, api_light)
        mu_heavy = beggs_robinson_dead_oil(T, api_heavy)
        
        assert mu_heavy > mu_light, "Heavier oil should have higher viscosity"

    def test_beggs_robinson_temperature_dependence(self):
        """Test that viscosity decreases with temperature."""
        api = 25.0
        T1 = 293.15  # 20°C
        T2 = 353.15  # 80°C
        
        mu1 = beggs_robinson_dead_oil(T1, api)
        mu2 = beggs_robinson_dead_oil(T2, api)
        
        assert mu2 < mu1, "Viscosity should decrease with temperature"


class TestViscosityModelClasses:
    """Test viscosity model classes."""

    def test_walther_viscosity_class(self):
        """Test WaltherViscosity class."""
        A, B = 10.0, 3.5
        model = WaltherViscosity(A, B)
        
        assert model.model_type == "walther"
        
        nu = model.calculate(313.15)
        assert nu > 0

    def test_andrade_viscosity_class(self):
        """Test AndradeViscosity class."""
        A, B = 1e-4, 1000
        model = AndradeViscosity(A, B)
        
        assert model.model_type == "andrade"
        
        mu = model.calculate(313.15)
        assert mu > 0

    def test_beggs_robinson_viscosity_class(self):
        """Test BeggsRobinsonViscosity class."""
        api = 27.0
        model = BeggsRobinsonViscosity(api)
        
        assert model.model_type == "beggs-robinson"
        assert model.api_gravity == api
        
        mu = model.calculate(313.15)
        assert mu > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
