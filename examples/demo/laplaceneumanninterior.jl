using bieps2d
using FMMLIB2D
using LinearAlgebra
import LinearMaps
import IterativeSolvers
using Plots
using Random
include("../src/julia/laplace.jl")
include("../src/julia/helsingquad_laplace.jl")

# Parameters
numpanels = 100
panelorder = 16
solve_inteq = true

# Discretize
#curve = AnalyticDomains.starfish(radius = 1, n_arms = 0, amplitude = 0)
curve = AnalyticDomains.starfish(n_arms = 5, amplitude = 0.3)
println(" Discretizing ")
dcurve = bieps2d.discretize(curve, numpanels, panelorder, equal_arclength=false)

N = dcurve.numpoints

#if solve_inteq
		# Setup a problem
		z0 = 3.0 + 1im*3.0
		f(zr,zi) = @. real(1 / (zr+1im*zi - z0))
		df(zr,zi) = @. conj(-1 / (zr+1im*zi-z0)^2) 
#		rhs1 = df(dcurve.points[1,:], dcurve.points[2,:]);
		complexnormals = dcurve.normals[1,:] .+ 1im*dcurve.normals[2,:];
		rhs = real.(df(dcurve.points[1,:], dcurve.points[2,:]) .* conj(complexnormals))  
		A = system_matrix_neumann(dcurve)
		sol = IterativeSolvers.gmres(A, rhs; reltol=eps())
		gmresidual = norm(A*sol-rhs, Inf)
		@show gmresidual

# Eval near center
zt1 = [[0.1,0.1] [-0.1,0.1] [-0.1,0.1] [-0.1,-0.1]]
#ref1 = f(zt1[1],zt1[2])
ref1 = f(zt1[1,:],zt1[2,:])
slp1 = layer_potential_direct(dcurve, sol, zt1; slp=1, dlp=0)
#relerr1 = min(sqrt.((ref1 .- slp1).^2)/ abs(ref1),sqrt.((ref1 .+ slp1).^2) / abs(ref1))
println(" * Direct eval")
@show sum(sol.*dcurve.dS)/(2*pi);
@show sum(rhs.*dcurve.dS);
diff = slp1 .- ref1
@show diff
C = maximum(diff) # assumes positive diff. Fix.
err = (abs(maximum(diff)) - abs(minimum(diff))) / norm(ref1, Inf)
@show err

ndS = dcurve.normals .* dcurve.dS'
slp1fmm = layer_potential_fmm(dcurve, sol, zt1, BitArray(undef, 4); slp=1, dlp=0)
#slp1fmm = slp1fmm ./ (-2*pi)
diff1 = slp1fmm .- ref1
err1 = (abs(maximum(diff1)) - abs(minimum(diff1))) / norm(ref1, Inf)
@show err1

# Eval near bdry
t1 = dcurve.t_edges[1,1]
t2 = dcurve.t_edges[2,1]
h1 = t2 - t1
ztc = curve.tau(t1+h1/2 + 1im*h1*0.1)
zt = [real(ztc), imag(ztc)]
println(" * Near eval")
@show zt
ref = f(zt[1], zt[2])
slp = layer_potential(dcurve, sol, zt; slp=1, dlp=0)
relerr = (ref .- slp .- C) / ref
@show relerr


# Eval on grid
M = 50
x = range(-1.3, 1.3, length = M)
y = x
X, Y = bieps2d.ndgrid(x, y)
zt = copy([vec(X) vec(Y)]')
interior, interior_near = bieps2d.interior_points(dcurve, zt)
zt = zt[:,interior]
ref = f(zt[1,:],zt[2,:])

ndS = dcurve.normals .* dcurve.dS'
slpfmm = layer_potential_fmm(dcurve, sol, zt, interior_near; slp=1, dlp=0)

slpquad = layer_potential(dcurve, sol, zt; slp=1, dlp=0)

@show norm(slpquad .- slpfmm)

slp = slpquad
diff = slp .- ref
err = maximum(abs.(diff .- C))  / norm(ref, Inf)
@show err

E = zeros(M^2)
E[interior] = abs.(diff .- C)
E = reshape(E,(M,M))

println(" Max relative pointwise error")
@show maximum(log10.(E[interior]))

# TODO: only interior points should be plotted
heatmap(x,y,log10.(E .+ eps()),clim = (-16.,-10.), aspect_ratio =:equal)
plot!(dcurve.points[1,:], dcurve.points[2,:])
