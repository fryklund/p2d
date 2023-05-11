using bieps2d
using FMMLIB2D
using LinearAlgebra
import LinearMaps
import IterativeSolvers
using Plots
using Random
include("../../src/julia/laplace.jl")
include("../../src/julia/helsingquad_laplace.jl")

# Parameters
numpanels1 = 100
numpanels2 = 50
numpanels3 = 70
panelorder = 16
solve_inteq = true

# Discretize
L = 4.0
#curve = AnalyticDomains.starfish(n_arms = 5, amplitude = 0.3)
curve1 = AnalyticDomains.starfish(n_arms = 5, amplitude = 0.1,radius = 0.4,center = [1.0,1.0], exterior=true)
curve2 = AnalyticDomains.starfish(n_arms = 5, amplitude = 0.1,radius=0.3,center=[-1.0,-1.0],exterior=true)
curve3 = AnalyticDomains.starfish(n_arms = 4, amplitude = 0.2,radius=0.3,center=[0.0,0.0],exterior=true)


#curve1 = AnalyticDomains.harmonicext(0.0,0.0,0.25,[0.0,0.0,0.0,0.0,0.02,0.01,0.0,0.01,0.0,0.01],[0.0,0.0,0.01,0.0,0.0,0.0,0.0,0.0,0.0,0.0],1.0,false)

#curve2 = AnalyticDomains.harmonicext(0.0,0.0,0.05,[0.0,0.005,0.0,0.0,0.005,0.0,0.005,0.0,0.0,0.0],[0.0,0.0,0.005,0.0,0.0,0.0,0.0,0.0,0.0,0.0],1.0,true)

curve = [curve1,curve2,curve3]
numcurve = length(curve)
numpanelsv = [numpanels1,numpanels2,numpanels3]

S = zeros(Float64,2,0)
if numcurve > 1
	 for i = 1:numcurve
		  global S = hcat(S,curve[i].center)
	 end
end

println(" Discretizing ")
dcurve = CurveDiscretization.discretize(curve, numpanelsv,panelorder, equal_arclength=true)

N = dcurve.numpoints
#M = numcurve - 1
#if solve_inteq
# Setup a problem
#z01 = 0.0 + 1im*0.0
z01 = 1.0 + 1im*1.0
z02 = -1.0 - 1im*1.0
z03 = 0.0 + 1im*0.0
#z03 = -0.2 + 1im*3.0*0.0
f(zr,zi) = @. real(1 / (zr+1im*zi - z01)) .+ real(1 / (zr+1im*zi - z02)) + real(1 / (zr+1im*zi - z03))
#		f(zr,zi) = @. log(abs(zr+1im*zi - z01)^2)
#rhs = vcat(f.(dcurve.points[1,:], dcurve.points[2,:]),zeros(Float64,numcurve + (numcurve-1 > 0)))
rhs = vcat(f.(dcurve.points[1,:], dcurve.points[2,:]),zeros(Float64,(numcurve-1>0) * (numcurve)))
#rhs = f.(dcurve.points[1,:], dcurve.points[2,:])

A = system_matrix(dcurve,S,exterior=true) # Todo: assemble with FMM

#Arhs = A * rhs
iprec = 5
source = dcurve.points
ifcharge = 1
charge = zeros(dcurve.numpoints)
ifdipole = 1
dipstr = f.(dcurve.points[1,:], dcurve.points[2,:])
dipvec = dcurve.normals .* dcurve.dS'
ifpot = 1
ifgrad = 0
ifhess = 0

#pot, grad, hess = CurveDiscretization.rfmm2dpartself(iprec, source, ifcharge, charge, ifdipole, dipstr, dipvec, ifpot, ifgrad, ifhess)
#ArhsFMM = -pot / (2*pi) + rhs.*dcurve.dS.*dcurve.curvature / (4*pi) - rhs/2
flhs = system_matvec(dcurve,S,exterior = true)

LHS = LinearMaps.LinearMap(flhs, dcurve.numpoints+(numcurve)*(numcurve>1))
sol, gmlog = IterativeSolvers.gmres(LHS, rhs; reltol=eps(), log=true)
#sol = sol[1:end-M+1-(numcurve>1)]
solmat = IterativeSolvers.gmres(A, rhs; reltol=eps())
@show norm(solmat - sol)
density = sol
#gmresidual = norm(A*sol-rhs, Inf)
#gmresidualFMM = norm(LHS*sol[1:end-M]-rhs[1:end-M], Inf)
#@show norm(sol - solmat)
#@show gmresidualFMM

# Eval near center
zt1 = dcurve.points[:,1] * 1.1
ref1 = f(zt1[1], zt1[2])
dlp1 = layer_potential_direct(dcurve, density, zt1; slp=0, dlp = 1,S,exterior=true)[1] 
relerr1 = (ref1 .- dlp1) / ref1
println(" * Direct eval")
@show zt1
@show dlp1
@show ref1
@show relerr1

ndS = dcurve.normals .* dcurve.dS'
#U = CurveDiscretization.rfmm2d(source=dcurve.points, target=zt1, dipstr=solmatvec, dipvec=ndS,ifpot=false, ifpottarg=true)
U = CurveDiscretization.rfmm2d(source=dcurve.points, target=zt1, dipstr=density[1:end-numcurve*(numcurve>1)], dipvec=ndS,ifpot=false, ifpottarg=true)

dlp2 = U.pottarg ./ (-2*pi) .+ sum(density[1:end-numcurve*(numcurve>1)] .* dcurve.dS) / 2 / pi
if numcurve-1 > 0
	for k = 1:numcurve
		dlp2 .+= density[end-numcurve + k] .* log.(sqrt.((zt1[1] .- S[1,k]).^2 .+ (zt1[2] .- S[2,k]).^2))
	end
end	
relerrfmm1 = (ref1 .- dlp2) / ref1
@show relerrfmm1


# Eval near bdry
t1 = dcurve.t_edges[1,end]
t2 = dcurve.t_edges[1,end-1]
h1 = t2 - t1
ztc = curve[1].tau(t1-h1/1000 - 1im*h1*0.0001)
zt = [real(ztc), imag(ztc)]
println(" * Near eval")
@show zt
ref = f(zt[1], zt[2])
dlp1 = layer_potential(dcurve, density, zt; slp=0, dlp=1,S,exterior=true)  #.+ sum(density[1:end-numcurve] .* dcurve.dS) / 2/ pi
relerrNEAR = (ref .- dlp1) / ref
@show relerrNEAR


# Eval on grid
x = range(-L/2, L/2, length = 50)
y = x
X, Y = bieps2d.ndgrid(x, y)
zt = copy([vec(X) vec(Y)]')
ref = f(zt[1,:], zt[2,:])

interior, interior_near = bieps2d.interior_points(dcurve, zt; exterior = true)

#interior_near[:] .= false

dlp = zeros(size(ref))
@time dlp[interior] = layer_potential(dcurve, density, zt[:, interior]; slp=0, dlp=1,S,exterior = true) 


dlpfmm = zeros(size(ref))

@time dlpfmm[interior] = layer_potential_fmm(dcurve, density, zt[:, interior], interior_near,S; slp=0, dlp=1, exterior = true)  #.+ sum(density[1:end-numcurve] .* dcurve.dS) / 2/ pi


temp1 = reshape(abs.(dlp .- ref), size(X)) / norm(ref, Inf)
E1 = zeros(size(temp1))
E1[interior] = temp1[interior]
temp2 = reshape(abs.(dlpfmm .- ref), size(X)) / norm(ref, Inf)
E2 = zeros(size(temp2))
E2[interior] = temp2[interior]

println(" Max relative pointwise error")
@show maximum(log10.(E1[interior]))
println(" Max relative pointwise error FMM")
@show maximum(log10.(E2[interior]))

# TODO: only interior points should be plotted
#heatmap(x,y,log10.(E1 .+ eps()),clim = (-16.,-10.), aspect_ratio =:equal)
#plot!(dcurve.points[1,:], dcurve.points[2,:])

