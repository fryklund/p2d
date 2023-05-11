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


function exactfgaussn(targ,dpars)
	pot = 0.0 #+ eps()
   ng=4

	for i = 1:4
		idp = (i-1)*3

		dx = targ[1] - dpars[idp+1]
		dy = targ[2] - dpars[idp+2]
		r2 = dx*dx + dy*dy

		sigma = dpars[idp+3]
		pot += exp(-r2 / sigma)
	end
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
	for i = 1:4
		idp = (i-1)*3
		dx = xy[1] - dpars[idp+1]
		dy = xy[2] - dpars[idp+2]
		r2 = dx*dx + dy*dy
		sigma = dpars[idp+3]
		f = f+exp(-r2/sigma)*(r2/sigma-1)*4/sigma + eps()
	end
	return f
end

function solveinterior(minlev)
#eulergamma = Base.MathConstants.eulergamma

boxlen = 1.0
#sigma = 0.25
Ngrid = 100
    
		rsig = 1.0/1000.0

#     Gaussian source centers and variance
		dpars = Vector{Float64}(undef,15)

      dpars[1] = 0.275
      dpars[2] = 0.0

      dpars[3] = rsig
      
      dpars[4] = 0.09
      dpars[5] = -0.25

      dpars[6] = rsig/2.1

      dpars[7] = -0.21
      dpars[8] = -0.16

      dpars[9] = rsig/4.5 
      
      dpars[10] = -0.23
      dpars[11] = 0.17

      dpars[12] = rsig/1.2
		  
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



#     x0 = 0.123
#    y0 = 0.5193
#   r(x,y) = sqrt( (x-x0)^2 + (y-y0)^2 )
#  ufuncr(r) = -1/(4.0*pi)*(expint(r^2/(2.0*sigma^2)) + log(r^2))
# ulim0 = (eulergamma + log(1/(2*sigma^2)))/(4*pi)
# ffuncr(r) = -exp(-r^2/(2*sigma^2))/(2*pi*sigma^2)    

uharmext(x,y) = sin.(10*(x .+ y)) .+ x.^2  .- 3*y + 8 + exp(-500*x^2)
fharmext(x,y) = 1000 * exp.(-500 .* x.^2) .* (1000 * x^2 .- 1) .- 200 * sin.(10 * (x .+ y)) .+ 2
#ufunc(x,y) = exactfgaussn([x,y],dpars)
#ffunc(x,y) = fgaussn([x,y],dpars)
#ufunc(x,y) = exacttrig([x,y],dpars)
#ufunc(x,y) = sin(10*(x + y)) + 64*x.^7 .- 112*x.^5 .+ 56*x.^3 .- 7*x - 3*y + 8 + exp(-500*x^2)
#ffunc(x,y) = 1000 * exp(-500 * x^2) * (1000 * x^2 - 1) - 200 * sin(10 * (x + y)) + 64*7*6*x.^5 .- 112*5*4*x.^3 .+ 56*3*2*x.^1
#ffunc(x,y) = (1024*x.^11 .- 2816*x.^9 .+ 2816*x.^7 .- 1232*x.^5 .+ 220*x.^3 .- 11*x + y^1.0)/1.0
#ufunc(x,y) = (1024/12/13)*x.^13 .- (2816/10/11)*x.^11 .+ (2816/8/9)*x.^9 .- (1232/6/7)*x.^7 .+ (220/4/5)*x.^5 .- (11/2/3)*x.^3 .+ y^3/3/2 .+ 1.0*0.0
#	ffunc(x,y) = exp(-((x-dpars[1])^2 + (y-dpars[2])^2)/dpars[3])*(((x-dpars[1])^2 + (y-dpars[2])^2)/dpars[3] - 1)*4/dpars[3] + exp(-((x-dpars[4])^2 + (y-dpars[5])^2)/dpars[6])*(((x-dpars[4])^2 + (y-dpars[5])^2)/dpars[6] - 1)*4/dpars[6] + exp(-((x-dpars[7])^2 + (y-dpars[8])^2)/dpars[9])*(((x-dpars[7])^2 + (y-dpars[8])^2)/dpars[9] - 1)*4/dpars[9] + exp(-((x-dpars[10])^2 + (y-dpars[11])^2)/dpars[12])*(((x-dpars[10])^2 + (y-dpars[11])^2)/dpars[12] - 1)*4/dpars[12]
#frhs = ffunc
ufunc(x,y) = uharmext(x,y)
ffunc(x,y) = fharmext(x,y)
	 
# Discretize
	
numpanels1 = 4000#100#200
numpanels2 = 180
    #numpanelsv = [numpanels1,numpanels2]
    numpanelsv = [numpanels1]
panelorder = 16
c0 = [0.0,0.0]
    #	curve1 = AnalyticDomains.starfish(n_arms = 5, amplitude = 0.3, radius = 0.23)
    	curve1 = AnalyticDomains.saw(radius = 0.85 * 0.2,exterior=false)
#	curve2 = AnalyticDomains.starfish(n_arms = 5, amplitude = 0.3, radius = 0.035,exterior=true)
#	curve = AnalyticDomains.starfish(n_arms = 10, amplitude = 0.7, radius = 0.22)
#curve1 = AnalyticDomains.harmonicext(0.0,0.0,0.25,[0.0,0.0,0.0,0.0,0.02,0.01,0.0,0.01,0.0,0.01],[0.0,0.0,0.01,0.0,0.0,0.0,0.0,0.0,0.0,0.0],1.0,false)
#a_ellipse = 0.02
#b_ellipse = 0.1
#curve2 = AnalyticDomains.ellipse(a = a_ellipse, b = b_ellipse, center=(0,0), interior = true)

#curve2 = AnalyticDomains.harmonicext(0.0,0.0,0.05,[0.0,0.005,0.0,0.0,0.005,0.0,0.005,0.0,0.0,0.0],[0.0,0.0,0.005,0.0,0.0,0.0,0.0,0.0,0.0,0.0],1.0,true)
    #curve = [curve1,curve2]
    curve = [curve1]
numcurve = length(curve)
S = zeros(Float64,2,length(curve) - 1) #Put centers here

#curve = [curve]

dcurve = CurveDiscretization.discretize(curve, numpanelsv, panelorder)
arclength = sum(dcurve.dS)

# Volume grid
xgrid = range(-boxlen/2, boxlen/2, length=Ngrid)
ygrid = range(-boxlen/2, boxlen/2, length=Ngrid)
#xgrid = range(-boxlen/2, boxlen/2, length=Ngrid+1)
#ygrid = range(-boxlen/2, boxlen/2, length=Ngrid+1)
#xgrid = xgrid[1:end-1]
#ygrid = ygrid[1:end-1]
X, Y = PUX.ndgrid(xgrid, ygrid)
xe = [vec(X) vec(Y)]
xet = copy(xe')
interior, interior_near = CurveDiscretization.interior_points(dcurve, xet)
#checkinside(x) = findall(CurveDiscretization.interior_points(dcurve, x)[1])
polygon = hcat(dcurve.points,dcurve.points[:,1])
polygon = StaticArrays.SVector.(polygon[1,:],polygon[2,:])
checkinside(x) = findall([PolygonOps.inpolygon(p, polygon; in=true, on=false, out=false) for p in StaticArrays.SVector.(x[1,:],x[2,:])] .== 1)
#checkinside(x) = checkinside_param(x,c0[1],c0[2],t -> hcat(real.(curve[1].tau.(t)),imag.(curve[1].tau.(t))))
#	checkinside1(x) = checkinside_param(x,c0[1],c0[2],t -> hcat(real.(curve[1].tau.(t)),imag.(curve[1].tau.(t))))

#	checkinside2(x) = checkinside_ellipse(x,c0[1],c0[2],a_ellipse,b_ellipse)
#	checkinside2(x) = checkinside_param(x,c0[1],c0[2],t -> hcat(real.(curve[2].tau.(-t)),imag.(curve[2].tau.(-t))))
#	checkinside(x) = checkinside_param2(x,checkinside1,checkinside2)


    # Setup input and reference in grid
    UREF = ufunc.(X, Y)

    xt, yt = dcurve.points[1,:], dcurve.points[2,:]
    xbdry = vcat(copy(xt'),copy(yt'))
	 xdomain = xet

    # Poisson solve free-space
	println(" * Extension and VFMM")
	Up, Up_bdry,minlev,FEXT,itree, iptr, fvals, centers, boxsize, nboxes, ipou, potmat, uplim = solve_volumepot(ffunc,boxlen,xbdry,xdomain,checkinside,minlev=minlev)
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
