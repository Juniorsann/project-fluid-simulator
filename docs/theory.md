# CFD Industrial Simulator - Theoretical Foundation

## Overview

This document provides the theoretical foundation for the CFD simulator, covering the fundamental equations, numerical methods, and physical models implemented.

## Governing Equations

### 1. Continuity Equation (Mass Conservation)

For incompressible flow:

```
∇·u = 0
```

Where:
- u: velocity vector [m/s]

### 2. Navier-Stokes Equations (Momentum Conservation)

```
ρ(∂u/∂t + u·∇u) = -∇p + ∇·(μ∇u) + ρg
```

Where:
- ρ: density [kg/m³]
- p: pressure [Pa]
- μ: dynamic viscosity [Pa·s]
- g: gravitational acceleration [m/s²]

For steady-state, fully developed pipe flow:
```
dp/dz = (1/r)·d/dr(r·μ·du/dr)
```

### 3. Energy Equation (Heat Transfer)

```
ρcp(∂T/∂t + u·∇T) = ∇·(k∇T) + Φ
```

Where:
- T: temperature [K]
- cp: specific heat [J/(kg·K)]
- k: thermal conductivity [W/(m·K)]
- Φ: viscous dissipation [W/m³]

## Analytical Solutions

### Poiseuille Flow (Laminar Pipe Flow)

For steady, fully developed laminar flow in a circular pipe:

**Velocity Profile:**
```
u(r) = uₘₐₓ(1 - r²/R²)
```

Where uₘₐₓ = 2u_mean

**Pressure Drop (Hagen-Poiseuille):**
```
Δp = 32μLu_mean/D²
```

Where:
- L: pipe length [m]
- D: pipe diameter [m]
- u_mean: mean velocity [m/s]

### Darcy-Weisbach Equation

For general pipe flow (laminar or turbulent):

```
Δp = f·(L/D)·(ρu²/2)
```

Where f is the Darcy friction factor:

**Laminar (Re < 2300):**
```
f = 64/Re
```

**Turbulent (Re > 4000):**
Colebrook-White equation (solved iteratively):
```
1/√f = -2·log₁₀(ε/(3.7D) + 2.51/(Re√f))
```

Where:
- ε: absolute roughness [m]
- Re: Reynolds number

## Dimensionless Numbers

### Reynolds Number
```
Re = ρuD/μ
```

Determines flow regime:
- Re < 2300: Laminar
- 2300 < Re < 4000: Transitional
- Re > 4000: Turbulent

### Prandtl Number
```
Pr = μcp/k = ν/α
```

Ratio of momentum diffusivity to thermal diffusivity.

### Nusselt Number
```
Nu = hD/k
```

Dimensionless heat transfer coefficient.

Correlations:
- Laminar, constant wall temperature: Nu = 3.66
- Laminar, constant heat flux: Nu = 4.36
- Turbulent (Dittus-Boelter): Nu = 0.023·Re⁰·⁸·Pr⁰·⁴

## Viscosity Models

### 1. Walther Equation (ASTM D341)

Standard for petroleum products:

```
log₁₀(log₁₀(ν + 0.7)) = A - B·log₁₀(T)
```

Where:
- ν: kinematic viscosity [cSt]
- T: absolute temperature [K]
- A, B: empirical constants

### 2. Andrade Equation

Exponential temperature dependence:

```
μ = A·exp(B/T)
```

Where:
- μ: dynamic viscosity [Pa·s]
- A, B: constants fitted from data

### 3. Beggs-Robinson Correlation

Specific for dead crude oil:

```
μ_oil = 10^X - 1  [cP]
X = y·T^(-1.163)
y = 10^z
z = 3.0324 - 0.02023·API
```

Where:
- T: temperature [°F]
- API: API gravity [°API]

Valid range: 70°F < T < 295°F, 16 < API < 58

## Oil Characterization

### API Gravity

Definition:
```
API = 141.5/SG₆₀ - 131.5
```

Where SG₆₀ is specific gravity at 60°F.

**Density Calculation:**
```
ρ = 141.5/(131.5 + API)·999  [kg/m³]
```

**Classification:**
- Light crude: API > 31.1°
- Medium crude: 22.3° < API < 31.1°
- Heavy crude: 10° < API < 22.3°
- Extra heavy crude: API < 10°

## Heat Transfer

### Convective Heat Transfer

Heat transfer rate:
```
Q = h·A·(Twall - Tbulk)
```

Overall heat transfer coefficient with pipe wall and insulation:
```
1/U = 1/hi + Rwall + Rinsulation + 1/ho
```

Where:
```
Rwall = ln(ro/ri)/(2πkwall)
Rinsulation = ln(rins/ro)/(2πkins)
```

### Temperature Profile with Constant Wall Temperature

Exponential approach to wall temperature:
```
T(z) = Twall - (Twall - Tin)·exp(-z/τ)
```

Where τ is the thermal time constant:
```
τ = ρcp·A·u/(h·P)
```

- A: cross-sectional area
- P: wetted perimeter

## Numerical Methods

### Finite Difference Method (FDM)

For 1D problems, discretize derivatives:
```
∂f/∂x ≈ (fi+1 - fi-1)/(2Δx)  [central difference]
∂²f/∂x² ≈ (fi+1 - 2fi + fi-1)/Δx²
```

### Grid Generation

**1D Uniform:**
- Cell centers: xi = (i + 0.5)·Δx
- Cell faces: xi+1/2 = i·Δx

**2D Axisymmetric with Wall Refinement:**
- Geometric progression for wall-normal direction
- Uniform spacing in axial direction

## References

1. White, F.M. (2011). "Fluid Mechanics", 7th Edition, McGraw-Hill.

2. Versteeg, H.K. & Malalasekera, W. (2007). "An Introduction to Computational Fluid Dynamics: The Finite Volume Method", 2nd Edition.

3. Beggs, H.D. and Robinson, J.R. (1975). "Estimating the Viscosity of Crude Oil Systems", Journal of Petroleum Technology.

4. ASTM D341 - Standard Practice for Viscosity-Temperature Charts for Liquid Petroleum Products.

5. API Technical Data Book - Petroleum Refining, 8th Edition.

6. Bird, R.B., Stewart, W.E., and Lightfoot, E.N. (2007). "Transport Phenomena", 2nd Edition.
