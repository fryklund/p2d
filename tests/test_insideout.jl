# Test code for identifiying inside/outside points

using ReTest

#push!(LOAD_PATH, string(pwd(),"/src"))
using Plots
using bieps2d
import .AnalyticDomains
import .CurveDiscretization

@testset "InsideOutside" begin
    # Discretize
    numpanels = 10
    panelorder = 16
    curve = AnalyticDomains.starfish(n_arms = 7, amplitude=0.3)
    dcurve = discretize(curve, numpanels, panelorder)
    # Easy test
    rgrid = range(0.01, 0.7-1e-8, length = 50)
    tgrid = range(0, 2*pi, length = 500)
#		R = rgrid' .* ones(500)
#		T = tgrid' .* ones(500)
    R, T = ndgrid(rgrid, tgrid)
    X = R.*cos.(T)
    Y = R.*sin.(T)
    zt = copy([X[:] Y[:]]')
    interior = interior_points(dcurve, zt)[1]
    exterior = .!interior
    @test all(interior)

    # clf()
    # plot(zt[1,interior], zt[2,interior],".b")
    # plot(zt[1,exterior], zt[2,exterior],".r")
    # plot(dcurve.points[1,:], dcurve.points[2,:], "-k")
    # axis("image")
	

    # Hard test
    rgrid = range(0, 2*pi, length = 1000)
    igrid = range(-5e-2, 5e-2, length = 100)
#		R = rgrid' .* ones(1000)
#		I = igrid' .* ones(100)
    R, I = ndgrid(rgrid, igrid)
    Z = @. curve.tau(R + 1im*I)
    X = real(Z)
    Y = imag(Z)

    zt = copy([X[:] Y[:]]')
    interior = interior_points(dcurve, zt)[1]
    exterior = .!interior

    ref_interior = I[:] .> 0
    ref_exterior = I[:] .< 0
		
    # clf()
    # plot(zt[1,interior], zt[2,interior],".b")
    # plot(zt[1,exterior], zt[2,exterior],".r")
    # plot(dcurve.points[1,:], dcurve.points[2,:], "-k")
    # axis("image")


    @test !any(ref_interior .& exterior) 
    @test !any(ref_exterior .& interior)
end
