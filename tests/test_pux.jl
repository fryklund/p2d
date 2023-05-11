using FFTW
using LinearAlgebra

function per_gradient(F, L)
	# Check input
	M = size(F)
	@assert length(M)>1 && M[1]==M[2] "Grid must be square"
	N = M[1]
	Fhat = fft(F)    
	k1, k2 = FreeSpace.k_vectors([N, N], [L, L])
	k1 = ifftshift(k1)
	k2 = ifftshift(k2)
	Fx = zeros(Complex{Float64}, size(Fhat))
	Fy = zeros(Complex{Float64}, size(Fhat))
	for j=1:N
    for i=1:N
			K1 = k1[i]
			K2 = k2[j]
			Fx[i,j] = 1im*K1*Fhat[i,j]
			Fy[i,j] = 1im*K2*Fhat[i,j]            
		end
	end
	# Transform back in-place
	ifft!(Fx)
	ifft!(Fy)
	# Return real parts
	return real(Fx), real(Fy)
end

@testset "PUX" begin
	Ngrid_int = 150
	# Discretize
	numpanels = 10
	panelorder = 16
	curve = AnalyticDomains.starfish(n_arms = 3, amplitude=0.1)
	dcurve = CurveDiscretization.discretize(curve, numpanels, panelorder, equal_arclength=true)

	# Volume grid
	# PUX
	pux_ep = 2.0
	pux_P = 40
	# Prepare
	PUXStor, X, Y, Lgrid, Ngrid, interior = PUX.pux_precompute(curve, dcurve, Ngrid_int, pux_ep; P=pux_P)
		
	f(x, y) = cos(2*pi*x) * cos(2*pi*y)
	fx(x, y) = -2*pi*sin(2*pi*x) * cos(2*pi*y)
	fy(x, y) = -2*pi*cos(2*pi*x) * sin(2*pi*y)
	#	f(x, y) = x.^2 + y.^2 + x.*y
	# fx(x, y) = 2*x + y
	# fy(x, y) = 2*y + x
	F = f.(X,Y)
	Fxref = fx.(X,Y)
	Fyref = fy.(X,Y)  
	# Extend
	println(" * PUX")
	FEXT = PUX.pux_eval(F, PUXStor)	
	# Error
	Fx, Fy = per_gradient(FEXT, Lgrid)
	Fx[.!interior] .= 0
	Fxref[.!interior] .= 0
	Fy[.!interior] .= 0
	Fyref[.!interior] .= 0
  Ex = abs.(Fx-Fxref)    
  Ey = abs.(Fy-Fyref)
  E = max.(Ex,Ey)
  logE = log10.(E)
	Emax = norm(vec(E[interior]), Inf)
	@show Emax
	@test Emax < 1e-6
end
