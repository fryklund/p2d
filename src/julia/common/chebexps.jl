using FFTW

function chebdiff(polin,n)
#=        
c 
c       this subroutine differentiates the Chebychev
c       expansion polin getting the expansion polout
c 
c 
c                       input parameters:
c 
c  polin - the Chebychev expansion to be differentiated
c  n - the order of the expansion polin
c   IMPORTANT NOTE: n is {\bf the order of the expansion, which is
c         one less than the number of terms in the expansion!!}
c         also nothe that the order of the integrated expansion is
c         n+1 (who could think!)
c 
c                       output parameters:
c 
c  polout - the Chebychev expansion of the derivative of the function
c         represented by the expansion polin
c 
    =#
    polout = zeros(n+1)
        for i=1:n+1
            polout[i]=0
        end
    
        for k=1:n-1
            polout[n-k]=polout[n-k+2]+(n-k)*polin[n-k+1] *2
        end
        polout[1]=polout[1]/2
        return polout
        end

function chebexps(itype,n,x,u,v,whts)
#=
        implicit real *8 (a-h,o-z)
        save
        dimension x(1),whts(1),u(n,n),v(n,n)
c 
c         this subroutine constructs the chebychev nodes
c         on the interval [-1,1], and the weights for the
c         corresponding order n quadrature. it also constructs
c         the matrix converting the coefficients
c         of a chebychev expansion into its values at the n
c         chebychev nodes. no attempt has been made to
c         make this code efficient, but its speed is normally
c         sufficient, and it is mercifully short.
c 
c                 input parameters:
c 
c  itype - the type of the calculation to be performed
c          itype=0 means that only the chebychev nodes are
c                  to be constructed.
c          itype=1 means that only the nodes and the weights
c                  are to be constructed
c          itype=2 means that the nodes, the weights, and
c                  the matrices u, v are to be constructed
c  n - the number of chebychev nodes and weights to be generated
c 
c                 output parameters:
c 
c  x - the order n chebychev nodes - computed independently
c          of the value of itype.
c  u - the n*n matrix converting the  values at of a polynomial of order
c         n-1 at n chebychev nodes into the coefficients of its
c         chebychev expansion - computed only in itype=2
c  v - the n*n matrix converting the coefficients
c         of an n-term chebychev expansion into its values at
c         n chebychev nodes (note that v is the inverse of u)
c          - computed only in itype=2
c  whts - the corresponding quadrature weights - computed only
c         if itype .ge. 1
c 
c       . . . construct the chebychev nodes on the interval [-1,1]

=#
	done = 1.0
	h = pi / (2*n)
	for i = 1:n
		t = (2*i - 1) * h
		x[n-i+1] = cos(t)
	end
	if itype == 0
		 return
	end

	if itype > 1
		u[1,1:n] .= 1
		u[2,1:n] .= x[:]
	end

# construct all quadrature weights and the rest of the rows

	 for i = 1:n
		tjm2 = 1
		tjm1 = x[i]
		whts[i] = 2

		ic = -1
		for j = 2:n-1

#		calculate the T_j(x(i))
 
			tj = 2 * x[i] * tjm1 - tjm2
 
			if itype == 2
				u[j+1,i] = tj
			end
			tjm2 = tjm1
			tjm1 = tj
#       calculate the contribution of this power to the
#       weight

			ic = -ic
			if ic >= 0
				rint = -2 * (done /(j + 1) - done/(j-1))
				whts[i] = whts[i] - rint * tj
			end
		end
		whts[i] = whts[i] / n
	end
	if itype != 2
		 return
	end
#        now, normalize the matrix of the chebychev transform

	for i = 1:n
		d = 0.0
		for j = 1:n
			 d = d + u[i,j]^2
		end
		d = done / sqrt(d)
		for j = 1:n
			u[i,j] = u[i,j] * d
		end
	end

#  now, rescale the matrix
 
	ddd = 2.0
	ddd = sqrt(ddd)
   dd = n
   dd = done / sqrt(dd/2)
   for i=1:n
		for j=1:n
			u[j,i] = u[j,i]*dd
		end
        u[1,i]=u[1,i] / ddd
	end
#        finally, construct the matrix v, converting the values at the
#        chebychev nodes into the coefficients of the chebychev
#        expansion
 
	dd = n
	dd = dd / 2
	for i = 1:n
		for j = 1:n
			v[j,i] = u[i,j] * dd
		end
	end
	for i = 1:n
		for j = 1:n
			d = v[j,i]
			v[j,i] = u[j,i] * n / 2
			u[j,i] = d / n * 2
		end
	end

	for i = 1:n
        v[1,i] = v[1,i] * 2
	end
   for i = 1:n
		for j = 1:i-1
			d = u[j,i]
			u[j,i] = u[i,j]
			u[i,j] = d
			d = v[j,i]
			v[j,i] = v[i,j]
			v[i,j] = d
		end
	end
	return
end #chebexps

function chebtens_exps_2d(itype,n,type,x,u,ldu,v,ldv,w)
#=
c                 input parameters:
c
c  itype - the type of the calculation to be performed
c          itype=0 means that only the gaussian nodes are 
c                  to be constructed. 
c          itype=1 means that only the nodes and the weights 
c                  are to be constructed
c          itype=2 means that the nodes, the weights, and
c                  the matrices u, v are to be constructed
c          itype=3 only construct u
c          itype=4 only construct v
      =#
	itype1d = 0
	if itype >= 1
		itype1d = 1
	end
	if itype >= 2
		itype1d = 2
	end
   x1d = Vector{Float64}(undef,n)
	w1d = Vector{Float64}(undef,n)
	u1d = Matrix{Float64}(undef,n,n)
	v1d = Matrix{Float64}(undef,n,n)
	chebexps(itype1d,n,x1d,u1d,v1d,w1d)
	ipt = 0
	for i = 1:n
		for j = 1:n
			ipt = ipt + 1
         x[1,ipt] = x1d[j]
			x[2,ipt] = x1d[i]               
      end
	end

   if itype >= 1
		ipt = 0
		for i = 1:n
			for j = 1:n
				ipt = ipt + 1
				w[ipt] = w1d[i]*w1d[j]
			end
		end
	end



	if itype == 2 || itype == 3
#     construct u from 1d u
		if type == 'f' || type == 'F'
			ipt = 0
			for io = 1:n
				for jo = 1:n
					ipt = ipt + 1
					ipol = 0
					for i = 1:n
						for j = 1:n
							ipol = ipol + 1
							u[ipol,ipt] =  u1d[i,io] * u1d[j,jo]
						end
					end
				end
			end
		elseif type == 't' || type == 'T'
			ipt = 0
			for io = 1:n
				for jo = 1:n
					ipt = ipt + 1
					ipol = 0
					for i = 1:n
						for j=1:n+1-i
							ipol = ipol + 1
							u[ipol,ipt] = u1d[i,io] * u1d[j,jo]
						end
					end
				end
			end
		end
  end
      
	if itype == 2 || itype == 4
	 # construct v from 1d v
		if type == 'f' || type == 'F'
			ipol = 0
			for io = 1:n
				for jo = 1:n
					ipol = ipol + 1
					ipt = 0
					for i = 1:n
						for j = 1:n
	                  ipt = ipt + 1
		               v[ipt,ipol] =  v1d[i,io] * v1d[j,jo]
						end
					end
            end
        end

		elseif type == 't' || type == 'T'
			ipol = 0
			for io = 1:n
				for jo = 1:n+1-io
					ipol = ipol + 1
					ipt = 0
               for i = 1:n
						for j = 1:n
							ipt = ipt + 1
							v[ipt,ipol] = v1d[i,io] * v1d[j,jo]
						end
					end
				end
        end
      end
   end         

	if itype == 2 || itype == 3
#     construct u from 1d u
		if type == 'f' || type == 'F'         
			ipt = 0
			for io = 1:n
				for jo = 1:n
					ipt = ipt + 1
               ipol = 0
               for i = 1:n
						for j = 1:n
	                  ipol = ipol + 1
							u[ipol,ipt] =  u1d[i,io] * u1d[j,jo]
						end
               end
            end
			end
		elseif type == 't' || type == 'T'
			ipt = 0
			for io = 1:n
            for jo = 1:n
               ipt = ipt + 1
               ipol = 0
               for i = 1:n
						for j = 1:n+1-i
							ipol = ipol + 1
							u[ipol,ipt] = u1d[i,io] * u1d[j,jo]
						end
               end
            end
			end
		end
	end


	if itype == 2 || itype == 4
#     construct v from 1d v
		if type == 'f' || type == 'F'         
			ipol = 0
			for io = 1:n
				for jo = 1:n
					ipol = ipol + 1
					ipt = 0
					for i = 1:n
						for j = 1:n
							ipt = ipt + 1
						   v[ipt,ipol] = v1d[i,io] * v1d[j,jo]
						end
					end
				end
			end

		elseif type == 't' || type == 'T'
			ipol = 0
			for io = 1:n
				for jo = 1:n+1-io
					ipol = ipol + 1
               ipt = 0
               for i = 1:n
						for j = 1:n
							ipt = ipt + 1
							v[ipt,ipol] = v1d[i,io] * v1d[j,jo]
						end
               end
				end
			end
		end
	end 
	return
end

function chebpols(x::Float64,a::Float64,b::Float64,N::Int64)
	# Necessary transformation to ensure that x lies in [-1,1]
	x = (2 * (x-a) / (b-a) ) - 1;
	# Initialization
	pols = zeros(N,length(x))
	# T_1(x) = 1
	pols[1,:] .= 1
	# T_2(x) = x
	if N > 1
		 pols[2,:] .= x
	end
	# for n>2, T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
	if N > 2
		for k = 3:N
			pols[k,:] .= 2 * x.* pols[k-1,:] .- pols[k-2,:]
		end
	end
	return pols
end		

function eval_func_2D(coefficients::Matrix{Float64},x::Float64,y::Float64,n1::Int64,a1::Float64,b1::Float64,n2::Int64,a2::Float64,b2::Float64)
	pols_x = chebpols(x,a1,b1,n1)
	pols_y = chebpols(y,a2,b2,n2)
	z = (pols_y' * coefficients * pols_x)[1]
	return z
end



## 2D Spectral interpolation using DCT
# n1,n2: number of Chebyshev nodes (first kind)
# a1,a2: intervall [a1,b1], [a2,b2]
# b1,b2: intervall [a1,b1], [a2,b2]
# f: function to be interpolated, form f(x,y)

function interpspec2D_DCT(n1::Int64,a1::Float64,b1::Float64,n2::Int64,a2::Float64,b2::Float64,f)
	c = zeros(n2,n1)
	coefficients = zeros(n2,n1)

	# Chebyshev grid calculation
	X,Y = chebynodesfirst_grid(a1,b1,n1,a2,b2,n2)

	# Evaluation of f on Chebyshev nodes
	fcheby = f.(X,Y)

	# First dimension - coefficients for [k]
	for k = 1:n2
		c[k,:] = coeff_dct(fcheby[k,:])
	end

	# Second dimension - coefficients for [k,l]
	for l=1:n1
		coefficients[:,l] = coeff_dct(copy(c[:,l]))
	end
	return coefficients
end

# Same as interpspec2D_DCT, but the function values are provided.

function interpspec2D_DCT_compcoeff(n1::Int64, n2::Int64, fcheby::Matrix{Float64})
	c = zeros(n2,n1)
	coefficients = zeros(n2,n1)
	
	# First dimension - coefficients for (k)
	for k=1:n2
		c[k,:] = coeff_dct(fcheby[k,:])   
	end

	# Second dimension - coefficients for (k,l)
	for l=1:n1
		coefficients[:,l] = coeff_dct(copy(c[:,l]))
	end
	return coefficients
end


# Chebyshevnodes of first kind

function chebynodesfirst(a::Float64, b::Float64, n::Int64)
	nodes = zeros(n)
	h = pi / (2*n)
	@inbounds for ii = 1:1:n
		t = (2*ii - 1) * h
		nodes[ii] = 0.5 * (b + a) - 0.5 * (a - b) * cos(t)
	end
	return nodes
end

## 2D Chebyshev grid (first kind)

# Calculates the [X,Y] grid obtained by the mesh of two vx, vy vectors
function chebynodesfirst_grid(a1::Float64, b1::Float64, n1::Int64, a2::Float64, b2::Float64, n2::Int64)

vx = chebynodesfirst(a1,b1,n1)
vy = chebynodesfirst(a2,b2,n2)

X,Y = meshgrid(vx,vy);

return X,Y
end

## Calculation of Chebyshev coefficients using the DCT method

function coeff_dct(value::Vector{Float64})
# x =-cos(pi/N*((0:N-1)'+1/2))
#value = value[end:-1:1]
N = length(value)
coefficients = sqrt(2/N) * dct(value)
coefficients[1] = coefficients[1] / sqrt(2)
return  coefficients
end




function meshgrid(xin::Vector{Float64},yin::Vector{Float64})
nx =length(xin)
ny = length(yin)
xout = zeros(ny,nx)
yout = zeros(ny,nx)
for jx = 1:nx
    for ix = 1:ny
        xout[ix,jx] = xin[jx]
        yout[ix,jx] = yin[ix]
    end
end
return xout, yout
end

#=

## Calculation of Chebyshev coefficients using the FFT method

function coeff_fft(value::Vector{Float64})
	N = length(value) - 1
	coeff = real(fft(vcat(value,value[N:-1:2])))
	coeff_reorder = vcat(coeff[1]/(2*N), coeff[2:N]/N, coeff[N+1]/(2*N))
	return  coeff_reorder
end


function coeff_fft_2d(n1::Int64, n2::Int64, fcheby::Matrix{Float64})
	c = zeros(n2,n1)
	coefficients = zeros(n2,n1)

	# First dimension - coefficients for (k)
	for k=1:n2
		 c[k,:] = coeff_fft(fcheby[k,:])   
	end

	# Second dimension - coefficients for (k,l)
	for l=1:n1
		 coefficients[:,l] = coeff_fft(copy(c[:,l]))
	end
	return coefficients
end

=#
