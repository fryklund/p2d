using LinearAlgebra
using bieps2d.AnalyticDomains, bieps2d.CurveDiscretization
#using InlineTest

@testset "Curve discretization" begin
	numpanels = 100
	panelorder = 16
	curve = AnalyticDomains.starfish(n_arms = 5, amplitude=0.3)
	dcurve = CurveDiscretization.discretize(curve, numpanels, panelorder);
	# Test equal arclength discretization
	h = sum(reshape(dcurve.dS, panelorder, :), dims=1)
	err = norm(h .- h[1], Inf)
	@show err
@test err < 1e-13
end
