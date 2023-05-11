# Test FMM matvec vs full matrix
include("../src/julia/laplace.jl")
using bieps2d
import LinearMaps
using LinearAlgebra

@testset "Matvec" begin
    # Discretize
    numpanels = 20
    panelorder = 16
    curve = AnalyticDomains.starfish(n_arms = 3, amplitude=0.2)
    dcurve = CurveDiscretization.discretize(curve, numpanels, panelorder)
    N = dcurve.numpoints
    # Compute
    x = rand(N)
		flhs = system_matvec(dcurve)
    LHS = LinearMaps.LinearMap(flhs, dcurve.numpoints)
		yfmm = LHS(x)
    A = system_matrix(dcurve)
    ydir = A*x
    erel = norm(yfmm-ydir, Inf) / norm(ydir, Inf)
    @test erel < 5e-14
end
