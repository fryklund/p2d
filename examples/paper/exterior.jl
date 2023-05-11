using bieps2d
using FMMLIB2D
using LinearAlgebra
import LinearMaps
import IterativeSolvers
using SpecialFunctions
include("../../src/julia/laplace.jl")
include("../../src/julia/helsingquad_laplace.jl")
include("../../dev/plotvfmm.jl")
include("../../dev/solve_volumepot.jl")

function exacttrig(targ,dpars)
	Cx = 0.5*pi
	Cy = Cx
	pot = sin(Cx*targ[1])*sin(Cy*targ[2])
	return pot
end


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



function fcos(xy,dpars,zpars,ipars)
	L = 0.5#dpars[1]
	x = xy[1]
	y = xy[2]
	f = -(4*pi^2*(-3 - 4*cos((2*pi*(x + y))/L) + cos((4*pi*(x + y))/L)))/(L^2*(2 + cos((2*pi*(x + y))/L))^3)
	return f
end

function fsinsin(xy,dpars,zpars,ipars)
	 k = 2 * pi
	 f = 0.0
	 f = - sin(k*xy[1])*sin(k*xy[2])
	return f
end

function fgaussn(xy,dpars)
#=
c
c       compute three gaussians, their
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

function solveexterior(minlevitr)
#minlev = 8
#eulergamma = Base.MathConstants.eulergamma

boxlen = 1.0
#sigma = 0.25
Ngrid = 100
    
		rsig = 1.0/1000.0

#     Gaussian source centers and variance
		dpars = Vector{Float64}(undef,15)
#=
      dpars[1] = 0.0
      dpars[2] = 0.0

      dpars[3] = 1/300
      
      dpars[4] = 0.13
      dpars[5] = -0.3

      dpars[6] = rsig/2.1

      dpars[7] = -0.26
      dpars[8] = -0.22

      dpars[9] = rsig/4.5 
      
      dpars[10] = -0.25
      dpars[11] = 0.21

      dpars[12] = rsig/1.2
		  
      dpars[13] = 0.09
      dpars[14] = 0.25

      dpars[15] = rsig/3.3
=#

      dpars[1] = 0.1
      dpars[2] = 0.07

      dpars[3] = rsig
      
      dpars[4] = 0.09
      dpars[5] = -0.25

      dpars[6] = rsig/2.1

      dpars[7] = -0.21
      dpars[8] = -0.25#-0.16

      dpars[9] = rsig/4.5 
      
      dpars[10] = 10.076
      dpars[11] = -0.051

      dpars[12] = rsig/2.5
		  
      dpars[13] = 0.09
      dpars[14] = 0.25

      dpars[15] = rsig/3.3
#	 ufunc(x,y) = exp(-((x-dpars[1])^2 + (y-dpars[2])^2)/dpars[3]) + exp(-((x-dpars[4])^2 + (y-dpars[5])^2)/dpars[6]) + exp(-((x-dpars[7])^2 + (y-dpars[8])^2)/dpars[9]) + exp(-((x-dpars[10])^2 + (y-dpars[11])^2)/dpars[12])
#ku = 40*pi
#ufunc(x,y) = sin(ku*x)*sin(ku*y) / (2 * ku^2)
#ffunc(x,y) = -sin(ku*x)*sin(ku*y)
#L = 1
#ufunc(x,y) = 1/(cos((2*pi*(x + y))/L) + 2)
#ffunc(x,y) = -(4*pi^2*(-3 - 4*cos((2*pi*(x + y))/L) + cos((4*pi*(x + y))/L)))/(L^2*(2 + cos((2*pi*(x + y))/L))^3)


#eulergamma = Base.MathConstants.eulergamma
#
x0 = -0.2
y0 = 0.0
sigma = 1/40
Crad = -10
r(x,y) = sqrt( (x-x0)^2 + (y-y0)^2 )
ufuncr(r) = Crad*(expint(r^2/(2.0*sigma^2)) + log(r^2))
#u(r) = Crad*(expint(r^2/(2.0*sigma^2)) + log(r^2))
#ulim = (eulergamma + log(1/(2*sigma^2)))/(4*pi)
ffuncr(r) = 2*Crad*exp(-r^2/(2*sigma^2))/(sigma^2)    
ufunc(x,y) = ufuncr(r(x,y))
ffunc(x,y) =  ffuncr(r(x,y))
#C0 = 0.0
#x0 = 0.0
#y0 = 0.0
ufunc(x,y) = exactfgaussn([x,y],dpars,Crad,x0,y0)
ffunc(x,y) = fgaussn([x,y],dpars)
#=
rad(x,y) = sqrt(x.^2 .+ y.^2) / (0.4)
Wen = setrbfWen(8)
Wenf(x,y) = rad(x,y) <= 1 ? (Wen(rad(x,y)) .+ eps()) : eps()
ufunc(x,y) = Wenf(x,y)
ffunctemp(x,y) = .-(1 ./sqrt.(x.^2 .+ y.^2)) .* 26 .* (.-1 .+ sqrt.(x.^2 .+ y.^2)).^8 .* (.-6162 .* x.^4 .- 6162 .* y.^4 .+ 2 * sqrt.(x.^2 .+ y.^2) .+ y.^2 .* (.-281 .+ 2976 .* sqrt.(x.^2 .+ y.^2)) .+ x.^2 .* (.-281 .- 12324 .* y.^2 .+ 2976 .* sqrt.(x.^2 .+ y.^2)))
Wenftemp(x,y) = rad(x/0.4,y/0.4) <= 1 ? (ffunctemp(x/0.4,y/0.4) .+ eps()) : eps()
ffunc(x,y) = Wenftemp(x,y)
=#
#z01 = 0.0 + 1im*0.0
#ufunc(zr,zi) = @. real(1 / (zr+1im*zi - z01))# .+ real(1 / (zr+1im*zi - z02)) + real(1 / (zr+1im*zi - z03))
#ffunc(x,y) = x*0.0 .+ y*0.0 .+ 0.0
#ufunc(x,y) = exacttrig([x,y],dpars)
#ufunc(x,y) = sin(10*(x + y)) + 64*x.^7 .- 112*x.^5 .+ 56*x.^3 .- 7*x - 3*y + 8 + exp(-500*x^2)
#ffunc(x,y) = 1000 * exp(-500 * x^2) * (1000 * x^2 - 1) - 200 * sin(10 * (x + y)) + 64*7*6*x.^5 .- 112*5*4*x.^3 .+ 56*3*2*x.^1
#ufunc(x,y) = (x^2 + y^2)/4
#ffunc(x,y) = x*0.0 .+ y*0.0 .+ 1.0
#	ffunc(x,y) = exp(-((x-dpars[1])^2 + (y-dpars[2])^2)/dpars[3])*(((x-dpars[1])^2 + (y-dpars[2])^2)/dpars[3] - 1)*4/dpars[3] + exp(-((x-dpars[4])^2 + (y-dpars[5])^2)/dpars[6])*(((x-dpars[4])^2 + (y-dpars[5])^2)/dpars[6] - 1)*4/dpars[6] + exp(-((x-dpars[7])^2 + (y-dpars[8])^2)/dpars[9])*(((x-dpars[7])^2 + (y-dpars[8])^2)/dpars[9] - 1)*4/dpars[9] + exp(-((x-dpars[10])^2 + (y-dpars[11])^2)/dpars[12])*(((x-dpars[10])^2 + (y-dpars[11])^2)/dpars[12] - 1)*4/dpars[12]
#frhs = ffunc
# Discretize

# Value at infinity: log(r)u0lim
#ulim = 0.0



numpanels1 = 200#4000#100
numpanels2 = 200
numpanels3 = 200
numpanelsv = [numpanels1,numpanels2,numpanels3]
panelorder = 16
c0 = [0.0,0.0]
curve1 = AnalyticDomains.starfish(n_arms = 5, amplitude = 0.3,radius = 0.12,center = [0.186,-0.15], exterior=true)

#	curve1 = AnalyticDomains.saw(radius = 0.85 * 0.2,exter0.186,-0.150.186,-0.15 ior=false)
curve2 = AnalyticDomains.starfish(n_arms = 4, amplitude = 0.3,radius=0.17,center=[-0.21,-0.03],exterior=true)
curve3 = AnalyticDomains.starfish(n_arms = 3, amplitude = 0.2,radius=0.2,center=[0.2,0.25],exterior=true)
#	curve = AnalyticDomains.starfish(n_arms = 10, amplitude = 0.7, radius = 0.22)
#curve1 = AnalyticDomains.harmonicext(0.0,0.0,0.25,[0.0,0.0,0.0,0.0,0.02,0.01,0.0,0.01,0.0,0.01],[0.0,0.0,0.01,0.0,0.0,0.0,0.0,0.0,0.0,0.0],1.0,false)
#a_ellipse = 0.02
#b_ellipse = 0.1
#curve2 = AnalyticDomains.ellipse(a = a_ellipse, b = b_ellipse, center=(0,0), interior = true)

#curve2 = AnalyticDomains.harmonicext(0.0,0.0,0.05,[0.0,0.005,0.0,0.0,0.005,0.0,0.005,0.0,0.0,0.0],[0.0,0.0,0.005,0.0,0.0,0.0,0.0,0.0,0.0,0.0],1.0,true)
curve = [curve1,curve2,curve3]
numcurve = length(curve)
S = zeros(Float64,2,0)
for i = 1:numcurve
#global
	 S = hcat(S,curve[i].center)
end

#curve = [curve]

dcurve = CurveDiscretization.discretize(curve, numpanelsv, panelorder)
arclength = sum(dcurve.dS)

# Volume grid
xgrid = range(-boxlen/2, boxlen/2, length=Ngrid)
ygrid = range(-boxlen/2, boxlen/2, length=Ngrid)
#xgrid = xgrid[1:end-1]
#ygrid = ygrid[1:end-1]
X, Y = PUX.ndgrid(xgrid, ygrid)
xe = [vec(X) vec(Y)]
xet = copy(xe')
interior, interior_near = CurveDiscretization.interior_points(dcurve, xet; exterior=true)

#checkinside(x) = findall(CurveDiscretization.interior_points(dcurve, x)[1])
polygon = hcat(dcurve.points,dcurve.points[:,1])
polygon = StaticArrays.SVector.(polygon[1,:],polygon[2,:])
#checkinside(x) = findall([PolygonOps.inpolygon(p, polygon; in=true, on=false, out=false) for p in StaticArrays.SVector.(x[1,:],x[2,:])] .== 1)
#checkinside(x) = checkinside_param(x,c0[1],c0[2],t -> hcat(real.(curve[1].tau.(t)),imag.(curve[1].tau.(t))))
#	checkinside1(x) = checkinside_param(x,c0[1],c0[2],t -> 10*hcat(real.(curve[1].tau.(t)),imag.(curve[1].tau.(t))))

#	checkinside2(x) = checkinside_ellipse(x,c0[1],c0[2],a_ellipse,b_ellipse)
#	checkinside2(x) = checkinside_param(x,c0[1],c0[2],t -> hcat(real.(curve[1].tau.(-t)),imag.(curve[1].tau.(-t))))
#	checkinside(x) = checkinside_param2(x,checkinside1,checkinside2)

checkinside(x) = checkoutside_multconnected(x,curve)
    # Setup input and reference in grid
    UREF = ufunc.(X, Y)

    xt, yt = dcurve.points[1,:], dcurve.points[2,:]
    xbdry = vcat(copy(xt'),copy(yt'))
	 xdomain = xet

    # Poisson solve free-space
	println(" * Extension and VFMM")
	up, up_bdry,minlev,FEXT,itree, iptr, fvals, centers, boxsize, nboxes, ipou, potmat, uplim  = solve_volumepot(ffunc,boxlen,xbdry,xdomain,checkinside,minlev=minlevitr)
#uplim = 0.0
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
#A = system_matrix(dcurve,S,exterior=true)
    gmresidual = norm(LHS*sol-rhs, Inf)
#soldirect = A\rhs
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
 #Uh = zeros(size(X))
 #   Uh[interior] = uh
	
#	E = ufunc.(X, Y) .- (reshape(up,size(X)) .+ (uplim/4/pi .+ Crad) .* log.((X .- 0.0).^2 .+ (Y .- 0.0).^2).^2) .- Uh


