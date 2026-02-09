"""
Tests for oil properties.
"""

import pytest
import numpy as np
from src.models.oil_properties import (
    OilFluid, OilType,
    create_light_oil, create_medium_oil, create_heavy_oil, create_extra_heavy_oil
)


class TestOilFluid:
    """Test OilFluid class."""

    def test_oil_creation(self):
        """Test basic oil creation."""
        oil = OilFluid(api_gravity=30.0)
        
        assert oil.api_gravity == 30.0
        assert oil.oil_type == OilType.MEDIUM
        assert oil.name == "crude_oil"

    def test_api_gravity_validation(self):
        """Test API gravity validation."""
        with pytest.raises(ValueError):
            OilFluid(api_gravity=0)
        
        with pytest.raises(ValueError):
            OilFluid(api_gravity=-10)

    def test_oil_classification(self):
        """Test oil type classification."""
        light = OilFluid(api_gravity=35.0)
        medium = OilFluid(api_gravity=27.0)
        heavy = OilFluid(api_gravity=15.0)
        extra_heavy = OilFluid(api_gravity=8.0)
        
        assert light.oil_type == OilType.LIGHT
        assert medium.oil_type == OilType.MEDIUM
        assert heavy.oil_type == OilType.HEAVY
        assert extra_heavy.oil_type == OilType.EXTRA_HEAVY

    def test_density_calculation(self):
        """Test density calculation from API gravity."""
        oil = OilFluid(api_gravity=30.0)
        
        # Density should be around 876 kg/m³ for API 30
        rho = oil.density()
        assert 850 < rho < 900

    def test_density_temperature_dependence(self):
        """Test that density decreases with temperature."""
        oil = OilFluid(api_gravity=30.0)
        
        rho1 = oil.density(293.15)  # 20°C
        rho2 = oil.density(353.15)  # 80°C
        
        assert rho2 < rho1, "Density should decrease with temperature"

    def test_viscosity_calculation(self):
        """Test viscosity calculation."""
        oil = OilFluid(api_gravity=30.0)
        
        mu = oil.dynamic_viscosity(313.15)  # 40°C
        assert mu > 0

    def test_viscosity_temperature_dependence(self):
        """Test that viscosity decreases with temperature."""
        oil = OilFluid(api_gravity=30.0)
        
        mu1 = oil.dynamic_viscosity(293.15)
        mu2 = oil.dynamic_viscosity(353.15)
        
        assert mu2 < mu1, "Viscosity should decrease with temperature"

    def test_kinematic_viscosity(self):
        """Test kinematic viscosity calculation."""
        oil = OilFluid(api_gravity=30.0)
        
        nu = oil.kinematic_viscosity(313.15)
        mu = oil.dynamic_viscosity(313.15)
        rho = oil.density(313.15)
        
        # nu = mu / rho
        np.testing.assert_allclose(nu, mu / rho, rtol=0.01)

    def test_reynolds_number(self):
        """Test Reynolds number calculation."""
        oil = OilFluid(api_gravity=30.0)
        
        Re = oil.reynolds_number(
            velocity=1.0,
            diameter=0.2,
            temperature=313.15
        )
        
        assert Re > 0
        assert isinstance(Re, (float, np.floating))

    def test_prandtl_number(self):
        """Test Prandtl number calculation."""
        oil = OilFluid(api_gravity=30.0)
        
        Pr = oil.prandtl_number(313.15)
        assert Pr > 0

    def test_thermal_properties(self):
        """Test thermal conductivity and specific heat."""
        oil = OilFluid(api_gravity=30.0)
        
        k = oil.thermal_conductivity()
        cp = oil.specific_heat()
        
        assert k > 0
        assert cp > 0


class TestPreConfiguredOils:
    """Test pre-configured oil types."""

    def test_create_light_oil(self):
        """Test light oil creation."""
        oil = create_light_oil()
        assert oil.oil_type == OilType.LIGHT
        assert oil.api_gravity > 31.1

    def test_create_medium_oil(self):
        """Test medium oil creation."""
        oil = create_medium_oil()
        assert oil.oil_type == OilType.MEDIUM
        assert 22.3 < oil.api_gravity < 31.1

    def test_create_heavy_oil(self):
        """Test heavy oil creation."""
        oil = create_heavy_oil()
        assert oil.oil_type == OilType.HEAVY
        assert 10 < oil.api_gravity < 22.3

    def test_create_extra_heavy_oil(self):
        """Test extra heavy oil creation."""
        oil = create_extra_heavy_oil()
        assert oil.oil_type == OilType.EXTRA_HEAVY
        assert oil.api_gravity < 10

    def test_oil_ordering_by_density(self):
        """Test that oils are ordered correctly by density."""
        light = create_light_oil()
        medium = create_medium_oil()
        heavy = create_heavy_oil()
        extra_heavy = create_extra_heavy_oil()
        
        # Lighter oils should have lower density
        assert light.density() < medium.density()
        assert medium.density() < heavy.density()
        assert heavy.density() < extra_heavy.density()

    def test_oil_ordering_by_viscosity(self):
        """Test that oils are ordered correctly by viscosity."""
        T = 313.15  # 40°C
        
        light = create_light_oil()
        medium = create_medium_oil()
        heavy = create_heavy_oil()
        extra_heavy = create_extra_heavy_oil()
        
        # Lighter oils should have lower viscosity
        assert light.dynamic_viscosity(T) < medium.dynamic_viscosity(T)
        assert medium.dynamic_viscosity(T) < heavy.dynamic_viscosity(T)
        assert heavy.dynamic_viscosity(T) < extra_heavy.dynamic_viscosity(T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
