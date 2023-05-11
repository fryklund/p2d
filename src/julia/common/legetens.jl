#=
c
c
c
c     this file contains routines for working with tensor
c     product gauss-legendre nodes and polynomials in 2 and
c     3 dimensions; also provides some related spectral
c     differentiation operators
c
c--------------------------------------------------------------
c
c     CONVENTIONS
c
c     ndeg refers to polynomial degree
c      
c     n refers to order of approximation and number of nodes
c     used in each direction (n = ndeg+1)
c
c     type character specifies the set of tensor
c     polynomials to use. current options
c
c     type = 'F', full degree polynomials (T_i(x)T_j(y)T_k(z) where
c                 each of i, j, and k goes from 0 to ndeg)
c     type = 'T', total degree polynomials (T_i(x)T_j(y)T_k(z) where
c                 sum of i,j,k>=0 is less than or equal to ndeg)
c
c     TODO: implement "Euclidean degree"
c
c     a tensor grid of points is traversed with x on the inner
c        loop, y on the loop above that, and z above that.
c        e.g. for a 2x2x2 grid, we have the order:
c            (x1,y1,z1), (x2,y1,z1), (x1,y2,z1), (x2,y2,z1)
c     (x1,y1,z2), (x2,y1,z2), (x1,y2,z2), (x2,y2,z2)
c      
c     tensor product polynomials are numbered analagously
c        e.g. for full degree at ndeg = 3, we would have the order
c        1, T_1(x), T_2(x), T_1(y), T_1(x)T_1(y), T_2(x)T_1(y),
c        T_2(y), T_1(x)T_2(y), T_2(x)T_2(y), T_1(z), T_1(x)T_1(z),
c        T_2(x) T_1(z), T_1(y)T_1(z), T_1(x)T_1(y)T_1(z), T_2(x)
c        T_1(y)T_1(z), T_2(y) T_1(z), T_1(x)T_2(y) T_1(z),
c        T_2(x) T_2(y) T_1(z), T_2(z), T_1(x)T_2(z), T_2(x) T_2(z),
c        T_1(y)T_2(z), T_1(x)T_1(y)T_2(z), T_2(x) T_1(y) T_2(z),
c        T_2(y) T_2(z), T_1(x)T_2(y) T_2(z), T_2(x) T_2(y) T_2(z)
c 
c        e.g. for total degree, we would have the order
c        1, T_1(x), T_2(x), T_1(y), T_1(x)T_1(y), T_2(y),
c        T_1(z), T_1(x)T_1(z), T_1(y)T_1(z), T_2(z)
c
c-------------------------------------------------------------     
c
c     ROUTINES
c      
c     legetens_npol_*d - total number of polynomials up
c                        to a given degree of specified
c                        type in 2 or 3 dimensions
c     legetens_exps_*d - get nodes, weights, and, optionally,
c                        projection and evaluation matrices      
c                        in 2 or 3 dimensions
c     legecoeff_dmat - rectangular 1d spectral diff mat
c     legecoeff_d2mat - rectangular 1d spectral 2nd deriv mat
c     legetens_pols_*d - evaluate the tensor legendre polys at
c                        a given point      
c     legetens_lapmat_*d - rectangular spectral laplacian in
c                          2 and 3 dims
c     legetens_eyemat_*d - rectangular "identity" matrix between
c                          different polynomial orders in
c                          2 and 3 dims 
=#

function legetens_ind2pow_2d(ndeg,type,iind2p)
    
    if type == 'f' || type == 'F'
	ipol = 0
	for i = 1:ndeg+1
	    for j = 1:ndeg+1
		ipol = ipol+1
		iind2p[1,ipol] = j-1
		iind2p[2,ipol] = i-1                
	    end
	end
    elseif type == 't' || type == 'T'
	ipol = 0
	for i = 1:ndeg+1
	    for j = 1:ndeg+1+1-i
		ipol = ipol+1
		iind2p[1,ipol] = j-1
		iind2p[2,ipol] = i-1
	    end
	end
    end
    return
end

function legval2coefs_2d(norder,fvals,umat)
    #     converts function values at 2d Legendre tensor product grid
    #     to Legendre expansion coefficients
    fcv = Matrix{Float64}(undef,norder,norder)
    for j = 1:norder
	for k = 1:norder
	    dd = 0.0
	    for k1 = 1:norder
		dd = dd + umat[k,k1] * fvals[k1,j]
	    end
	    fcv[k,j] = dd
	end
    end

    fcoefs = Matrix{Float64}(undef,norder,norder)
    for j = 1:norder
	for k = 1:norder
	    dd = 0.0
	    for j1 = 1:norder
		dd = dd + umat[j,j1] * fcv[k,j1]
	    end
            fcoefs[k,j] = dd
	end
    end
    return fcoefs[:]
end

function polydiff_2d(ipoly,ndeg,ttype,polin,idir,dmat,polout)
    #=
    c
    c       This subroutine differentiates a tensor
    c       product legendre expansion expressed in 
    c       its coefficient basis and returns
    c       the tensor product legendre expansion
    c       coeffs of the derivative
    c 
    c       input arguments:
    c
    c     ndeg refers to polynomial degree
    c      
    c     n refers to order of approximation and number of nodes
    c     used in each direction (n = ndeg+1)
    c
    c     type character specifies the set of tensor
    c     polynomials to use. current options
    c
    c     ttype = 'F', full degree polynomials (T_i(x)T_j(y)T_k(z) where
    c                 each of i, j, and k goes from 0 to ndeg)
    c     ttype = 'T', total degree polynomials (T_i(x)T_j(y)T_k(z) where
    c                 sum of i,j,k>=0 is less than or equal to ndeg)
    c     
    c     polin - real *8 (*)
    c       tensor product legendre expansion coeffs of 
    c       input polynomial
    c
    c     dmat - real *8 (ndeg+1,ndeg+1)
    c       differentiation martix in 1d
    c
    c     idir - integer
    c       whether to compute x,y,or z derivative of expansion
    c       idir = 1, compute x derivative
    c       idir = 2, compute y derivative
    c       idir = 3, compute z derivative
    c
    c
    c
    =#
    coef1 = zeros(ndeg+1,ndeg+1)
    coef2 = zeros(ndeg+1,ndeg+1)
    iind2pow = zeros(2,(ndeg+1)^2)

    npol = 0
    polytens_ind2pow_2d(ndeg,ttype,iind2pow)
    polytens_npol_2d(ndeg,ttype,npol)
    #
    #   extract coef mat
    #
    #

    coef1 = 0
    coef2 = 0
    for i=1:npol
        i1 = iind2pow[1,i]+1
        i2 = iind2pow[2,i]+1
        coef1[i1,i2] = polin(i)
    end

    if(idir==1)
        for k=1:ndeg+1
            for l=1:ndeg+1
                for i=1:ndeg+1
                    coef2[l,k] = coef2[l,k] + dmat[l,i]*coef1[i,k]
                end
            end
        end
    end

    if(idir==2)
        for k=1:ndeg+1
            for l=1:ndeg+1
                for i=1:ndeg+1
                    coef2[k,l,j] = coef2[k,l] + dmat[l,i]*coef1[k,i]
                end
            end
        end
    end

    for i=1:npol
        i1 = iind2pow[1,i]+1
        i2 = iind2pow[2,i]+1
        polout[i] = coef2[i1,i2] 
    end


return
end


function polytens_ind2pow_2d(ndeg,type,iind2p)



    if (type == 'f' || type == 'F')
        ipol = 0
        for i = 1:ndeg+1
            for j = 1:ndeg+1
                ipol = ipol+1
                iind2p[1,ipol] = j-1
                iind2p[2,ipol] = i-1                
            end
        end
    elseif (type == 't' || type == 'T')
        ipol = 0
        for  i = 1:ndeg+1
            for j = 1:ndeg+1+1-i
                ipol = ipol+1
                iind2p[1,ipol] = j-1
                iind2p[2,ipol] = i-1
            end
        end
    end

    return
end



function polytens_npol_2d(ndeg,type,npol)
    #
    #     return the number of polynomials of a given type
    #     up to degree ndeg
    #      
    n = ndeg + 1
    
    if (type == 'f' || type == 'F')
        npol = n^2
    elseif (type == 't' || type=='T') then
        npol = n*(n+1)/2
    end

    return
end
