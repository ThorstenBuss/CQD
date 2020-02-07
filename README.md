# CQD - The Gross-Pitaevskii equation (GPE): Dynamics of solitons and vortices
The GPE is a mean field equation that has been incredibly successful in describing the dynamics
of Bose Einstein condensates (BECs). The equation features a non-linear term and thus allows for
stable soliton solutions in 1D and additional topological defects such as vortices in 2D. The goal of
this project is to study these phenomena using the split-step Fourier method.

Literature: Sebastian Erne Diplomarbeit (provided), M. Karl and T. Gasenzer New. J. Phys. 19:093014 (2017)Methods: Mean field approximation, discretization, non-linear partial differential equations, split-step Fourier method

Steps:
* Simulate the evolution of dark solitons in a homogeneous (no trapping potential) 1D Bose gas. Use single and multiple grey solitons and study their propagation.
* Study the dynamics of solitons in a homogeneous 2D Bose gas.
* In 2D other topological defects, called vortices can be present. Use the given routine to initialize a 2D gas containing vortices (and anti-vortices) and visualize and analyze their dynamics. What happens if you initially put some singly quantized vortices (and no anti-vortices)? What happens to vortices with winding number higher than 1? What happens if you start with a vortex-anti-vortex pair or several of them?
_Hint_ on spatial and temporal resolution: It will be important to choose the right spatial resolution
such that the healing length is well resolved. Since you don’t want to go to huge grid sizes due to
computational limitations, this will limit the number of vortices you can sensibly use.
Hint on initial conditions in 2D: The idea in the provided code is that the initial condition respects
the periodic boundary conditions (which the FFT method imposes), which they unfortunately do
not. (Is it actually possible topologically possible to have vortices and still maintain periodic
boundaries?) This causes artefacts at the boundaries, which, however, for large enough grids
should not overlay the vortex dynamics completely.