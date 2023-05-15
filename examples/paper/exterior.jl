using bieps2d
using FMMLIB2D
using LinearAlgebra
import LinearMaps
import IterativeSolvers
using SpecialFunctions
include("../../src/julia/laplace.jl")
include("../../src/julia/helsingquad_laplace.jl")
include("../../src/julia/solve_volumepot.jl")


function exactfgaussn(targ,dpars,Crad,x0,y0)
	pot = 0.0 #+ eps()
   ng=4

	for i = 1:ng
		idp = (i-1)*3

		dx = targ[1] - dpars[idp+1]
		dy = targ[2] - dpars[idp+2]
		r2 = dx*dx + dy*dy

		sigma = dpars[idp+3]
		pot += exp(-r2 / sigma)
	end
	pot += log((targ[1]-x0).^2 .+ (targ[2]-y0).^2)*Crad
	return pot
end


function fgaussn(xy,dpars)
#=
c
c       compute four gaussians, their
c       centers are given in dpars(1:3*nd), and their 
c       variances in dpars(3*nd+1:4*nd)
c
=#

	ng = 4
	f = 0.0
	for i = 1:ng
		idp = (i-1)*3
		dx = xy[1] - dpars[idp+1]
		dy = xy[2] - dpars[idp+2]
		r2 = dx*dx + dy*dy
		sigma = dpars[idp+3]
		f = f+exp(-r2/sigma)*(r2/sigma-1)*4/sigma + eps()
	end
	return f
end

function solveexterior(minlevitr,tol,uniform)

boxlen = 1.0

Ngrid = 100

rsig = 1.0/1000.0	
x0 = -0.2
y0 = 0.0
sigma = 1/40
Crad = -10
#     Gaussian source centers and variance
		dpars = Vector{Float64}(undef,15)
      dpars[1] = 0.1
      dpars[2] = 0.07

      dpars[3] = rsig
      
      dpars[4] = 0.09
      dpars[5] = -0.25

      dpars[6] = rsig/2.1

      dpars[7] = -0.21
      dpars[8] = -0.25

      dpars[9] = rsig/4.5 
      
      dpars[10] = 10.076
      dpars[11] = -0.051

      dpars[12] = rsig/2.5
		  
      dpars[13] = 0.09
      dpars[14] = 0.25

      dpars[15] = rsig/3.3

ufunc(x,y) = exactfgaussn([x,y],dpars,Crad,x0,y0)
ffunc(x,y) = fgaussn([x,y],dpars)
# Discretize
numpanels1 = 200
numpanels2 = 200
numpanels3 = 200
numpanelsv = [numpanels1,numpanels2,numpanels3]
panelorder = 16
c0 = [0.0,0.0]
curve1 = AnalyticDomains.starfish(n_arms = 5, amplitude = 0.3,radius = 0.12,center = [0.186,-0.15], exterior=true)
curve2 = AnalyticDomains.starfish(n_arms = 4, amplitude = 0.3,radius=0.17,center=[-0.21,-0.03],exterior=true)
curve3 = AnalyticDomains.starfish(n_arms = 3, amplitude = 0.2,radius=0.2,center=[0.2,0.25],exterior=true)

curve = [curve1,curve2,curve3]
numcurve = length(curve)
S = zeros(Float64,2,0)
for i = 1:numcurve
	 S = hcat(S,curve[i].center)
end



dcurve = CurveDiscretization.discretize(curve, numpanelsv, panelorder)
arclength = sum(dcurve.dS)

# Volume grid
xgrid = range(-boxlen/2, boxlen/2, length=Ngrid)
ygrid = range(-boxlen/2, boxlen/2, length=Ngrid)
X, Y = ndgrid(xgrid, ygrid)
xe = [vec(X) vec(Y)]
xet = copy(xe')
interior, interior_near = CurveDiscretization.interior_points(dcurve, xet; exterior=true)

polygon = hcat(dcurve.points,dcurve.points[:,1])
polygon = StaticArrays.SVector.(polygon[1,:],polygon[2,:])

checkinside(x) = checkoutside_multconnected(x,curve)
checkcut(c,bs) = checkcut_polar(c,bs,curve)
    # Setup input and reference in grid
    UREF = ufunc.(X, Y)

    xt, yt = dcurve.points[1,:], dcurve.points[2,:]
    xbdry = vcat(copy(xt'),copy(yt'))
	 xdomain = xet

    # Poisson solve free-space
	println(" * Extension and VFMM")
	up, up_bdry,minlev,FEXT,itree, iptr, fvals, centers, boxsize, nboxes, ipou, potmat, uplim  = solve_volumepot(ffunc,boxlen,xbdry,xdomain,checkinside,curve,uniform,tol,checkcut,minlev=minlevitr)

    # Laplace solve
    bdry_cond = ufunc.(xt, yt)
    mod_bdry_cond = bdry_cond - (up_bdry  - (uplim .- Crad*(numcurve==1)) .* log.((xt .- S[1,1]).^2 .+ (yt .- S[2,1]).^2))
    @show norm(mod_bdry_cond)

    println(" * Integral equation solve")
flhs = system_matvec(dcurve,S,exterior = true)
 
LHS = LinearMaps.LinearMap(flhs, dcurve.numpoints+(numcurve)*(numcurve>1))
if numcurve == 1
	 rhs = mod_bdry_cond
else
	rhs = vcat(mod_bdry_cond,zeros(Float64,(numcurve-1>0) * (numcurve-1)),Crad*2)
end

	@time sol, gmlog = IterativeSolvers.gmres(LHS, rhs; reltol=eps(), log=true)
    gmresidual = norm(LHS*sol-rhs, Inf)

    @show gmresidual

density = sol
println(" * Layer pot eval")

	@time uh = layer_potential_fmm(dcurve, density, xet[:, interior],interior_near,S; slp = 0, dlp = 1,exterior=true)


	 println(" * Compute error")
	 UREF = ufunc.(X, Y)
	 Uh = zeros(size(X))
    Uh[interior] = uh
	 Up = zeros(size(X))
    Up = reshape(up,size(X))
    U = Uh + Up - (uplim .- Crad*(numcurve == 1)) .* log.((X .- S[1,1]).^2 .+ (Y .- S[2,1]).^2) 
    idxnan = findall(isnan.(U[:]))
	 U[idxnan] .= eps()
    E = U - UREF
	
    E[.!interior] .= 0
    U[.!interior] .= 0
	 UREF[.!interior] .= 0    
    Einf = norm(E, Inf)/norm(UREF,Inf)
	 EL1 = sum(abs.(E))/sum(abs.(UREF))
    h = X[2,1]-X[1,1]
    EL2 = sqrt(sum(E.^2)*h^2)/sqrt(sum(UREF.^2)*h^2)
    @show Einf
    @show EL1
    @show EL2

return [Einf EL1 EL2]'
end


