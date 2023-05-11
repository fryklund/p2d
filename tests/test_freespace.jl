# Dev code for free space solver
using LinearAlgebra
using .CurveDiscretization
using Random
eulergamma = Base.MathConstants.eulergamma


@testset "FreeSpaceSolver" begin
Random.seed!(1)
## Starting out with Poisson
# Setup problem
L = 5.0

# Exact solution from Gaussian blob rhs (from Vico & Greengard)
sigma = L/20
x0 = 0.123
y0 = 0.2193
r(x,y) = sqrt( (x-x0)^2 + (y-y0)^2 )
# solution is Gaussian
ufuncr(r) = exp(-r^2/(2*sigma^2))
ffuncr(r) = exp(-r^2/(2*sigma^2))*(r^2-2*sigma^2)/sigma^4

ufunc(x, y) = ufuncr(r(x,y))
ffunc(x, y) = ffuncr(r(x,y))

	function solve(N)
		# Grid domain
		x = range(-L/2, L/2, length=N+1)
		x = x[1:end-1] # Note that intervals are closed/open!
		y = x
		X,Y = ndgrid(x, y)
		# Nonuniform test points
		M = 1000
		xt = L*(1/2 .- rand(M, 1))
		yt = L*(1/2 .- rand(M, 1))

		# Get rhs
		F = ffunc.(X,Y)
		Fedges=[F[1,:];F[end,:];F[:,1];F[:,end]]
		Fgridresolution = norm(Fedges, Inf) / norm(vec(F), Inf)
		@show Fgridresolution

		# Get reference solution
		Uref = ufunc.(X,Y)
		U, ut = FreeSpace.fs_poisson(F, L, xt, yt)
		E = U .- Uref
		relerr = norm(E[:], Inf) / norm(Uref[:], Inf)
		@show relerr

		# Compute error at nonuniform test points
		enu = ut - ufunc.(xt,yt)
    relerrnu = norm(enu, Inf) / norm(ut, Inf)
	
# Compute numerical derivatives to check that we actually satisfy the PDE
        Unorm = norm(sqrt.( vec(U).^2), Inf)
        h = X[2]-X[1]
        int = 2:N-1
        Ux = (U[int.+1,int] .- U[int.-1,int])/(2*h)
        LU = (-4*U[int,int] .+ U[int.+1,int] .+ U[int.-1,int] .+ U[int,int.+1] .+ U[int,int.-1]) / h^2
      
        pde = LU .- F[int,int]

        pdeerr_rel = maximum(abs.(pde))/Unorm

        return relerr, relerrnu, pdeerr_rel

			end
 # Solve above problem for a variety of N
    iters = 3;
    diverr_list = zeros(iters)
    pdeerr_list = zeros(iters)
    N_list = zeros(iters)
    for i=0:iters-1
        N = 64*2^i
        relerr, relerrnu, pdeerr_rel = solve(N)
        @test relerr < 1e-13
        @test relerrnu < 1e-13
        N_list[i+1] = N
        pdeerr_list[i+1] = pdeerr_rel
    end
    # Estimate convergence constant of FD test (should be 2)
    A = [ones(iters) -log.(N_list)]    
    pde_conv = (A\log.(pdeerr_list))[2]
    @test abs(pde_conv - 2) < 0.05   
end
