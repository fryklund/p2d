using bieps2d
using FMMLIB2D
using LinearAlgebra
import LinearMaps
import IterativeSolvers
using SpecialFunctions
include("../../src/julia/laplace.jl")
include("../../src/julia/helsingquad_laplace.jl")
include("../../dev/laplacesquare.jl")
include("../../dev/solve_volumepot.jl")


function harm_exact_solutionpoissonsquare(x,y)
#		return (x.^2 .+ y.^2)./4 .+ 1.0
	return sin.(10*(x .+ y)) + 64*x.^7 .- 112*x.^5 .+ 56*x.^3 .- 7*x .- 3*y .+ 8 .+ exp.(-500*x.^2)
end

function harm_exact_rhspoissonsquare(x,y)
#	 return x*0.0 .+ y*0.0 .+ 1.0
	 	 return 1000 * exp.(-500 * x.^2) * (1000 * x.^2 - 1) .- 200 * sin.(10 * (x .+ y)) .+ 64*7*6*x.^5 .- 112*5*4*x.^3 .+ 56*3*2*x.^1
end


function exactfgaussn(targ,dpars)
	pot = 0.0 #+ eps()
   ng=5

	for i = 1:ng
		idp = (i-1)*3

		dx = targ[1] - dpars[idp+1]
		dy = targ[2] - dpars[idp+2]
		r2 = dx*dx + dy*dy

		sigma = dpars[idp+3]
		pot += exp(-r2 / sigma)
	end
	return pot
end




function fgaussn(xy,dpars)
#=
c
c       compute three gaussians, their
c       centers are given in dpars(1:3*nd), and their 
c       variances in dpars(3*nd+1:4*nd)
c
=#

	ng = 5
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


#eulergamma = Base.MathConstants.eulergamma
		rsig = 1.0/1000.0

#     Gaussian source centers and variance
		dpars = Vector{Float64}(undef,15)

      dpars[1] = -0.063#0.275
      dpars[2] = 0.31#0.0

      dpars[3] = rsig
      
      dpars[4] = -0.3#0.09
      dpars[5] = -0.091#-0.25

      dpars[6] = rsig/2.1

      dpars[7] = 0.108#-0.23
      dpars[8] = -0.31#-0.18

      dpars[9] = rsig/4.5 
      
      dpars[10] = -0.01#-0.23
      dpars[11] = 0.02#0.17

      dpars[12] = rsig/1.2
		  
      dpars[13] = 0.345#-0.01
      dpars[14] = 0.081#0.02

      dpars[15] = rsig/3.3


# Assumes sides parallel with x-axis and y-axis
function checkinsidesquare(x,y,boxlen,center,theta)
	xp = x .* cos(-theta) .- y .* sin(-theta)
	yp = y .* cos(-theta) .+ x .* sin(-theta)
	 
	a1 = center[1] - boxlen/2.0
	b1 = center[1] + boxlen/2.0
	a2 = center[2] - boxlen/2.0
	b2 = center[2] + boxlen/2.0 
	return intersect(findall(a1 .< xp .< b1), findall(a2 .< yp .< b2))
end

function solvesquare(minlev)
# ********************************************************
# *** Solves Laplace equation with basic RCIP inside a polygon    *
# ********************************************************
# *** User specified quantities ***************
lambda = -1/2;
nvertices = 4;
panels_per_edge = 100; # must have at least 5 panels per edge
boxlenfmm = 1.0
boxlen =  0.5
theta = pi/3
boxcenter = [0.01,-0.02]
# *********************************************

# Gauss-Legendre points and weights on the interval [-1,1]
T, W = gausslegendre(16)
nq = length(W);
npan = panels_per_edge * nvertices; # number of coarse panels
np = npan * nq;
nsub = [5, 10, 20];

# *** Pbc and PWbc prolongation matrices ***
IP, IPW = IPinit(T,W)
Pbc = Matrix(blockdiag(sparse(Diagonal(ones(16))),sparse(IP) ,sparse(IP),sparse(Diagonal(ones(16)))))
PWbc = Matrix(blockdiag(sparse(Diagonal(ones(16))),sparse(IPW) ,sparse(IPW),sparse(Diagonal(ones(16)))))



#=
x0 = -0.38#-0.35
y0 = -0.135#-0.0

x1 = -10.09#0.06
y1 = 0.42#0.36

x2 = 10.445#0.402
y2 = 0.115#-0.01

x3 = 10.135#0.01
y3 = -0.405#-0.37

sigma0 = 1/40
Crad0 = -2

sigma1 = 1/5
Crad1 = -8

sigma2 = 1/3
Crad2 = -7

sigma3 = 1/2
Crad3 = -6
=#
	 
#x0 = -0.38#-0.35
#y0 = -0.135#-0.0

x0 = -0.35
y0 = -0.135

x1 = -0.09#0.06
y1 = 0.42#0.36

x2 = 0.445#0.402
y2 = 0.09#-0.01

x3 = 0.135#0.01
y3 = -0.405#-0.37


sigma0 = 1/40
Crad0 = -2

sigma1 = 1/40
Crad1 = -2

sigma2 = 1/40
Crad2 = -2

sigma3 = 1/40
Crad3 = -2

r0(x,y) = sqrt( (x-x0)^2 + (y-y0)^2 )
r1(x,y) = sqrt( (x-x1)^2 + (y-y1)^2 )
r2(x,y) = sqrt( (x-x2)^2 + (y-y2)^2 )
r3(x,y) = sqrt( (x-x3)^2 + (y-y3)^2 )

ufuncr(r0,r1,r2,r3) = Crad0*(expint(r0^2/(2.0*sigma0^2)) + log(r0^2)) + Crad1*(expint(r1^2/(2.0*sigma1^2)) + log(r1^2)) + Crad2*(expint(r2^2/(2.0*sigma2^2)) + log(r2^2)) + Crad3*(expint(r3^2/(2.0*sigma3^2)) + log(r3^2)) 
#u(r) = Crad*(expint(r^2/(2.0*sigma^2)) + log(r^2))
#ulim = (eulergamma + log(1/(2*sigma^2)))/(4*pi)
ffuncr(r0,r1,r2,r3) = 2*Crad0*exp(-r0^2/(2*sigma0^2))/(sigma0^2) + 2*Crad1*exp(-r1^2/(2*sigma1^2))/(sigma1^2) + 2*Crad2*exp(-r2^2/(2*sigma2^2))/(sigma2^2) + 2*Crad3*exp(-r3^2/(2*sigma3^2))/(sigma3^2) 

#L = 1
		
#ufunc(x,y) = 1/(cos((2*pi*(x + y))/L) + 2)
#ffunc(x,y) = -(4*pi^2*(-3 - 4*cos((2*pi*(x + y))/L) + cos((4*pi*(x + y))/L)))/(L^2*(2 + cos((2*pi*(x + y))/L))^3)

			
#exact_solutionpoissonsquare(x,y) = exactfgaussn([x,y],dpars)
#exact_solutionpoissonsquare(x,y) = harm_exact_solutionpoissonsquare(x,y)
exact_solutionpoissonsquare(x,y) = ufuncr(r0(x,y),r1(x,y),r2(x,y),r3(x,y))
#exact_solutionpoissonsquare(x,y) = ufunc(x,y)
#exact_rhspoissonsquare(x,y) = fgaussn([x,y],dpars)
#exact_rhspoissonsquare(x,y) = harm_exact_rhspoissonsquare(x,y)
#exact_rhspoissonsquare(x,y) = ffunc(x,y)	 
exact_rhspoissonsquare(x,y) = ffuncr(r0(x,y),r1(x,y),r2(x,y),r3(x,y))
# determine "star" indices, i.e. those close to a corner
starind = Array{Any}(undef,nvertices)
for i = 1 : nvertices
    if i == 1
        starind[i] = vcat(np - 2*nq + 1: np, 1:2*nq)
    else
        start = (i - 1) * nq * panels_per_edge - 2 * nq + 1
        starind[i] = collect(start : start + 4 * nq - 1)
    end
end

Ncheck = 1200

xa = boxcenter[1] - boxlen/2.0
xb = boxcenter[1] + boxlen/2.0
ya	= boxcenter[2] - boxlen/2.0
yb = boxcenter[2] + boxlen/2.0

xx1 = range(xa,xb,length = Ncheck)
yy1 = ones(Ncheck) * ya
xp1 = xx1 .* cos(theta) .- yy1 .* sin(theta)
yp1 = yy1 .* cos(theta) .+ xx1 .* sin(theta)

xx2 = range(xa,xb,length = Ncheck)
yy2 = ones(Ncheck) * yb
xp2 = xx2 .* cos(theta) .- yy2 .* sin(theta)
yp2 = yy2 .* cos(theta) .+ xx2 .* sin(theta)

xx3 = ones(Ncheck) * xa
yy3 = range(ya,yb,length = Ncheck)
xp3 = xx3 .* cos(theta) .- yy3 .* sin(theta)
yp3 = yy3 .* cos(theta) .+ xx3 .* sin(theta)

xx4 = ones(Ncheck) * xb
yy4 = range(ya,yb,length = Ncheck)
xp4 = xx4 .* cos(theta) .- yy4 .* sin(theta)
yp4 = yy4 .* cos(theta) .+ xx4 .* sin(theta)

xbdrycheck = vcat(hcat(copy(xp1'),copy(xp2'),copy(xp3'),copy(xp4')),hcat(copy(yp1'),copy(yp2'),copy(yp3'),copy(yp4')))
# initialize geometry, Laplace DLP matrix and right hand side
x, y, xp, yp, tangent_x, tangent_y, curv, jac, thetas, s_coarse, panels_x, panels_y = geom_init(npan, T, W, nvertices, 0, boxlen, theta, boxcenter)

K = LaplaceKernelDL(x, y, tangent_x, tangent_y, curv, jac)
rhs = exact_solutionpoissonsquare.(x,y)

# compute volume potential
frhs(x,y) = exact_rhspoissonsquare(x,y)
xbdry = vcat(copy(x'),copy(y'))
Ngrid = 100;
fudge = 0.99	 
xgrid = range(-boxlenfmm/2 * fudge, boxlenfmm/2 * fudge, length=Ngrid)
ygrid = range(-boxlenfmm/2 * fudge, boxlenfmm/2 * fudge, length=Ngrid)
#xgrid = xgrid[1:end-1]
#ygrid = ygrid[1:end-1]
X_target, Y_target = PUX.ndgrid(xgrid, ygrid)
xe = [vec(X_target) vec(Y_target)]
xet = copy(xe')
xdomain = xet
curve(t) = t.*0
checkinside(x) = checkinsidesquare(x[1,:],x[2,:],boxlen,boxcenter,theta)
Up, Up_bdry,minlev2,FEXT,itree, iptr, fvals, centers, boxsize, nboxes, ipou, potmat, uplim = solve_volumepot(frhs,boxlenfmm,xbdry,xdomain,checkinside,minlev=minlev)
rhs = rhs .- Up_bdry

# solve for actual density function on the coarse grid
rho_coarse = IterativeSolvers.gmres(lambda*I + K, rhs; reltol=eps())	 

# cretate K^\circ and K^*
Kcirc = deepcopy(K);
Kstar = zeros(size(K))
for i = 1 : nvertices
    Kcirc[starind[i], starind[i]] .= 0.0
    Kstar[starind[i], starind[i]] .= K[starind[i], starind[i]]
end
# *** Recursion for the R matrix ***

if !isempty(nsub)
    rho_hat = zeros(Float64,length(rho_coarse), length(nsub))
    for i = 1 : length(nsub)
        R=(1/lambda)*Matrix(Diagonal(ones(np)))
        Rblock = Rcomp(thetas,lambda,T,W,Pbc,PWbc,np,starind, false, nsub[i])
        
        for k = 1 : nvertices
            R[starind[k], starind[k]] .= Rblock[starind[k], starind[k]]
        end
        rho_tilde = IterativeSolvers.gmres(I + Kcirc * R,rhs; reltol=eps(),maxiter=np)	 

        rho_hat[:,i] = R * rho_tilde
    end
end

R=(1/lambda)*Matrix(Diagonal(ones(np)))
Rblock = Rcomp(thetas,lambda,T,W,Pbc,PWbc,np,starind,true,nsub)

for k = 1 : nvertices
    R[starind[k], starind[k]] = Rblock[starind[k], starind[k]]
end

# solve for transformed density on coarse grid 
rho_tilde = IterativeSolvers.gmres(I + Kcirc * R,rhs; reltol=eps(),maxiter=np)	 
	 
rho_hat_rec = R * rho_tilde

# reconstruct fine density
nsub_fine = 10
xf, yf, xpf, ypf, tangent_xf, tangent_yf, temp1, jacf, temp2, s_fine, panels_xf, panels_yf = geom_init(npan, T, W, nvertices, nsub_fine, boxlen, theta, boxcenter)
 
rho_fine = reconstruct_fine_density(lambda, thetas, Pbc, T, W, R, starind, nsub_fine, npan, rho_tilde)
## Post-processing 
     
# no subdivisions
	 #=
M = 100;
x = range(-1.0/2*0.99, 1.0/2*0.99, length = M)
y = x
X_target, Y_target = bieps2d.ndgrid(x, y)
=#
        
u_dlp_refined = evaluateLaplaceDLP(X_target[:], Y_target[:], xf, yf, tangent_xf, tangent_yf, jacf, rho_fine)

u_dlp_refined = helsing_quad(u_dlp_refined, rho_fine, xf, yf, xpf, ypf, jacf, vcat(panels_xf, panels_xf[1]), vcat(panels_yf, panels_yf[1]), X_target[:], Y_target[:])

u_dlp_refined = u_dlp_refined .+ Up[:] 

UREF = exact_solutionpoissonsquare.(X_target[:], Y_target[:])
u_error = abs.(u_dlp_refined .- UREF)
idx = checkinsidesquare(X_target[:],Y_target[:],boxlen,boxcenter,theta);
E = zeros(size(u_error))
E[idx] .= u_error[idx]

println(" * Compute error")
Einf = norm(E, Inf)/norm(UREF[idx],Inf)
EL1 = sum(abs.(E[idx]))/sum(abs.(UREF[idx]))
h = X_target[2,1]-X_target[1,1]
EL2 = sqrt(sum(E[idx].^2)*h^2)/sqrt(sum(UREF[idx].^2)*h^2)
    @show Einf
	 @show EL1
    @show EL2

return [Einf EL1 EL2]'
end
#errorv[minlev,:] .= [Einf EL1 EL2]'

#pcolor(X_target, Y_target, reshape(log10.(u_error), size(X_target)));
#heatmap(xgrid, ygrid, reshape(log10.(u_error), size(X_target)))



