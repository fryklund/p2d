using bieps2d
using FMMLIB2D
using LinearAlgebra
import LinearMaps
import IterativeSolvers
using SpecialFunctions
include("../../src/julia/laplace.jl")
include("../../src/julia/helsingquad_laplace.jl")
include("../../src/julia//solve_volumepot.jl")


function solveinterior(minlev,tol,uniform)

boxlen = 1.0
Ngrid = 100


uharmext(x,y) = sin.(10*(x .+ y)) .+ x.^2  .- 3*y + 8 + exp(-500*x^2)
fharmext(x,y) = 1000 * exp.(-500 .* x.^2) .* (1000 * x^2 .- 1) .- 200 * sin.(10 * (x .+ y)) .+ 2
ufunc(x,y) = uharmext(x,y)
ffunc(x,y) = fharmext(x,y)

# Discretize

numpanels1 = 4000
numpanelsv =[numpanels1]
panelorder = 16
c0 = [0.0,0.0]

curve1 = AnalyticDomains.saw(radius = 0.85 * 0.2,exterior=false)

curve = [curve1]
numcurve = length(curve)
S = zeros(Float64,2,length(curve) - 1) #Put centers here



dcurve = CurveDiscretization.discretize(curve, numpanelsv, panelorder)
arclength = sum(dcurve.dS)

# Volume grid
xgrid = range(-boxlen/2, boxlen/2, length=Ngrid)
ygrid = range(-boxlen/2, boxlen/2, length=Ngrid)
X, Y = ndgrid(xgrid, ygrid)
xe = [vec(X) vec(Y)]
xet = copy(xe')
interior, interior_near = CurveDiscretization.interior_points(dcurve, xet)

polygon = hcat(dcurve.points,dcurve.points[:,1])
polygon = StaticArrays.SVector.(polygon[1,:],polygon[2,:])
checkinside(x) = findall([PolygonOps.inpolygon(p, polygon; in=true, on=false, out=false) for p in StaticArrays.SVector.(x[1,:],x[2,:])] .== 1)
checkcut(c,bs) = checkcut_search(c,bs,curve)
# Setup input and reference in grid
UREF = ufunc.(X, Y)

xt, yt = dcurve.points[1,:], dcurve.points[2,:]
xbdry = vcat(copy(xt'),copy(yt'))
xdomain = xet

# Poisson solve free-space
println(" * Extension and VFMM")
Up, Up_bdry,minlev,FEXT,itree, iptr, fvals, centers, boxsize, nboxes, ipou, potmat, uplim = solve_volumepot(ffunc,boxlen,xbdry,xdomain,checkinside,curve,uniform,tol,checkcut,minlev=minlev)
# Laplace solve
bdry_cond = ufunc.(xt, yt)
mod_bdry_cond = bdry_cond - Up_bdry
@show norm(mod_bdry_cond)

println(" * Integral equation solve")
flhs = system_matvec(dcurve,S)
LHS = LinearMaps.LinearMap(flhs, dcurve.numpoints+numcurve-1)
rhs = vcat(mod_bdry_cond,zeros(Float64,numcurve-1))
@time sol, gmlog = IterativeSolvers.gmres(LHS, rhs; reltol=eps(), log=true)
gmresidual = norm(LHS*sol-rhs, Inf)
@show gmresidual

density = sol
println(" * Layer pot eval")
@time uh = layer_potential_fmm(dcurve, density, xet[:, interior],interior_near,S; slp = 0, dlp = 1)


println(" * Compute error")
UREF = ufunc.(X, Y)
Uh = zeros(size(X))
Uh[interior] = uh
U = Uh + reshape(Up,(Ngrid,Ngrid))
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
