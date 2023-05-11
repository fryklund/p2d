using ReTest
using LinearAlgebra

import rbfqr

function f(op,xk)
	x = xk[:,1]
	y = xk[:,2]

	if length(op) == 1
		if op == "0"
			y = sin.(x .- 2*y)
		elseif op == "x"
			y = cos.(x .- 2*y)
		elseif op == "y"
			y = .-2*cos.(x .- 2*y)
		elseif op == "L"
			y = .-5*sin.(x .- 2*y)
		end
	elseif length(op) == 2
		if op == "xx"
			y = .-sin.(x .- 2*y)
		elseif op == "xy"
			y =  2*sin.(x .- 2*y)
		elseif op == "yy"
			y = .-4*sin.(x .- 2*y)
		end  
	end
end


@testset "RBF-QR"  begin


# The shape parameter
ep = 0.5

# Use Halton nodes in the unit square
N = 200
d=2
xk = rbfqr.RBFdiscretization.halton(N,d)
# Evaluation points
Ne = 40 # Per dimension
x = range(0,1,length=Ne)
xx,yy = rbfqr.RBFdiscretization.ndgrid(x,x)
xe = hcat(xx[:], yy[:])
A, Psi = rbfqr.rbfqr_diffmat_2d("1", xe, xk, ep)
		#=
Ax = rbfqr_diffmat_2d("x", xe, Psi)[1]
Ay = rbfqr_diffmat_2d("y", xe, Psi)[1]
Axx = rbfqr_diffmat_2d("xx", xe, Psi)[1]
Axy = rbfqr_diffmat_2d("xy", xe, Psi)[1]
Ayy = rbfqr_diffmat_2d("yy", xe, Psi)[1]
Lo = rbfqr_diffmat_2d("L", xe, Psi)[1]
=#
uk = f("0",xk)

@test maximum(abs.(A*uk-f("0",xe))) < 1e-9
		#=
@test maximum(abs.(Ax*uk-f("x",xe))) < 1e-7
@test maximum(abs.(Ay*uk-f("y",xe))) < 1e-7
@test maximum(abs.(Axx*uk-f("xx",xe))) < 1e-5
@test maximum(abs.(Axy*uk-f("xy",xe))) < 1e-6
@test maximum(abs.(Ayy*uk-f("yy",xe))) < 1e-5
@test maximum(abs.(L*uk-f("L",xe))) < 1e-5
 =#
end
