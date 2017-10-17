# GPU Driven Finite Difference WENO Scheme for Real Time Solution of the Shallow Water Equations
## P. Parna, K. Meyer, R. Falconer
Accompanying source code for the PIFWENO3/C paper.

# Abstract
The shallow water equations are applicable to many common engineering problems involving modelling of waves dominated by motions in the horizontal directions
(e.g. tsunami propagation, dam breaks). As such events pose substantial economic costs, as well as potential loss of life, accurate real-time simulation and
visualization methods are of great importance. For this purpose, we propose a new finite difference scheme for the 2D shallow water equations that is specifically
formulated to take advantage of modern GPUs. The new scheme is based on the so-called Picard integral formulation of conservation laws combined with Weighted Essentially
Non-Oscillatory reconstruction. The emphasis of the work is on third order in space and second order in time solutions (in both single and double precision). Further, the
scheme is well-balanced for bathymetry functions that are not surface piercing and can handle wetting and drying in a GPU-friendly manner without resorting to long and specific
case-by-case procedures. We also present a fast single kernel GPU implementation with a novel boundary condition application technique that allows for simultaneous real-time
visualization and single precision simulations even on large (>2000x2000) grids on consumer-level hardware - the full kernel source codes are also provided online at [https://github.com/pparna/swe_pifweno3](https://github.com/pparna/swe_pifweno3).

## Full text
[Computers & Fluids: GPU Driven Finite Difference WENO Scheme for Real Time Solution of the Shallow Water Equations](http://www.google.com)

# Demo Video
Available [here](http://www.phys-gfx.net/swe_pifweno3).