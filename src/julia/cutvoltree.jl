using bieps2d
using OffsetArrays
using Octavian

include("rbfqr_diffmat_2d_fast.jl")
include("common/chebexps.jl")
include("common/legetens.jl")
include("common/chebyshev2dinterpolation.jl")

# Data structure for PUX precomputation
mutable struct Extdata
    idxsupp::Vector{Int64}
    idxin::Vector{Int64}
    cin::Matrix{Int64}
    csupp::Matrix{Int64}
    A::Matrix{Float64}
end


#=

# TODO: STORE data from checking if points are inside or outside, stupid to do it in fun_ext
c     1. nd is always 1 for now; 2. norder can only be 8; 3. iperiod 
c     is 0 - other boundary conditions have not been translated;
c     4. the internal flag for eps is iprec that takes values 0-3
c        (3,6,9,12 digits of accuracy);
c     5. boxsize(0) and centers(:,1) are fixed.

generate level restricted oct tree based on resolving a 
c    function to desired precision
c
c
c    The function handle is of the form
c    call fun(xy)
c
c      where xy is the location in (-L/2,L/2)^2
c
c
c    For the Helmholtz/Maxwell tree, the boxes are refined until 
c     Re(zk)*boxsize<5, beyond which the function resolution criterion
c    kicks in. 
c
c    A function is said to be resolved if it's interpolant at the 8
c    children nodes, agrees with the function values at those nodes
c    upto the user specified tolerance. 
c    The error is scaled by h**(eta)
c    where eta is user specified and h is the boxsize. If there is 
c    any confusion, the user should seet \eta to 0
c    Let \tilde{f} denote the interpolant of f, then
c    the refinement criterion is 
c      \int_{B_{j} |\tilde{f}-f|^{p} *h^{\eta} < 
c        \varepsilon V_{j}^{1/p}/V_{0}^{1/p}/(\int_{B_{0}}|f|^{p})^{1/p}
c    This implies that
c       \int_{B_{0}} |\tilde{f}-f|^{p} =
c          \sum_{j} \int_{B_{j}} |\tilde{f}-f|^{p}
c          \leq \sum_{j} \eps^{p}*h^{\eta p} 
c                  V_{j}/V_{0}/(\int_{B_{0}} |f|^p)
c      If \eta = 0,
c          \leq \eps^{p}/(\int_{B_{0}} |f|^{p})
c
c    i.e., this strategy guarantees that the interpolated function
c      approximates the function with relative lp accuracy of \eps
c      
c    This code has 2 main user callable routines
c      make_vol_tree_mem -> Returns the memory requirements, 
c          tree length, number of boxes, number of levels
c      make_vol_tree -> Makes the actual tree, returns centers of boxes,
c          colleague info, function values on leaf boxes
c       
c          
c          iptr(1) - laddr
c          iptr(2) - ilevel
c          iptr(3) - iparent
c          iptr(4) - nchild
c          iptr(5) - ichild
c          iptr(6) - ncoll
c          iptr(7) - coll
c          iptr(8) - ltree
c
=#

function cutvol_tree_mem(tol::Float64,boxlen::Float64,norder::Int64,iptype::Int64,eta::Float64,fun::Function,rintl::OffsetVector{Float64, Vector{Float64}},checkinside::Function,xk::Matrix{Float64},minlev::Int64,bdrytol::Float64,curve)
    #=
    c
    c      get memory requirements for the tree
    c
    c      input parameters:
    c        tol - double precision
    c           precision requested
    c        zk - double complex
    c           Helmholtz parameter
    c        boxlen - double precision
    c           length of box in which volume is contained, 
    c           if boxlen = L, then volume is [-L/2,L/2]^2
    c        norder - integer
    c           order of discretization
    c        eta - double precision
    c           scaling parameter for error
    c        fun - function handle
    c           function to evalute it everywhere in the volume
    c        nd - integer
    c           number of real functions returned by fun
    c
    c        output:
    c           nlevels - integer
    c             number of levels
    c           nboxes - integer
    c             number of boxes
    c           nlboxes - integer
    c             number of leaf boxes
    c           ltree - integer
    c             length of tree
    c           rintl(0:nlevels) - real *8
    c             lp norm to scale the functions by
    c             (on input rintl should be of size(0:200)
    c
    c      
    c
    =#
    nbmax = 1000000
    nlmax = 200
    boxsize = OffsetArray(Vector{Float64}(undef,nlmax+1),0:nlmax)
    laddr = OffsetArray(Matrix{Int64}(undef,2,nlmax+1),1:2,0:nlmax)
    ilevel =	Vector{Int64}(undef,nbmax)
    iparent = Vector{Int64}(undef,nbmax)
    nchild =	Vector{Int64}(undef,nbmax)
    ichild = Matrix{Int64}(undef,4,nbmax)
    insideidx = Matrix{Int64}(undef, norder * norder, nbmax)
    ninsideidx = Vector{Int64}(undef, nbmax)

    fvals = zeros(Float64,norder^2,nbmax)
    centers = Matrix{Float64}(undef,2,nbmax)
    rintbs = Vector{Float64}(undef,nbmax)
    psis = Vector{Psistruct}(undef,nbmax)   # Vector of data structures for rbf-qr, store only for cut leaf boxes
    
    # Data structures for cut-boxes
    icut = Matrix{Int64}(undef,2,nbmax) # 1 if a cut box, 0 if inside, 2 if outside
    next = Vector{Int64}(undef,nbmax) # Number of boxes to extend to, we only store for leaf boxes
    ext = Matrix{Int64}(undef,24,nbmax) # Indices of boxes to extend to (including itself), we only stor cut-boxes that have no children
    
    # Set tree info for level 0
    laddr[1:2,0] .= 1
    ilevel[1] = 0
    iparent[1] = -1
    nchild[1] = 0
    ichild[1:4,1] .= -1
    centers[1:2,1] .= 0

    
    npbox = norder^2

    grid = Matrix{Float64}(undef,2,npbox)
    
    xq = Vector{Float64}(undef,norder)
    wts = Vector{Float64}(undef,norder)
    umat = Matrix{Float64}(undef,norder,norder)	
    vmat = Matrix{Float64}(undef,norder,norder)
    itype = 2 # What type of calculation to be performed, see function chebexps
    chebexps(itype,norder,xq,umat,vmat,wts)
    xq /= 2

    mesh2d(xq,norder,xq,norder,grid)

    npols = norder^2
    umat2 = Matrix{Float64}(undef,norder,norder)
    wts2 = Vector{Float64}(undef,npbox)
    xref2 = Matrix{Float64}(undef,2,npbox)
    itype = 1
    chebtens_exps_2d(itype,norder,'f',xref2,umat2,npols,vmat,1,wts2)

    #       compute fvals at the grid

    boxsize[0] = boxlen
    xy = Vector{Float64}(undef,2)
    rint = 1.0
    rintbs[1] = 0.0

    #   note extra factor of 4 sincee wts2 are on [-1,1]^2 
    #   as opposed to [-1/2,1/2]^2
    rsc = boxlen^2/4

    # Check if a geometry is inscribed in the box [-L/2,L/2]^2.
    idxin = checkinside(grid*boxlen)
    nin = length(idxin)

    icut[:,1] .= 1
    insideidx[1:nin,1] .= idxin
    insideidx[nin+1:end,1] .= setdiff(range(1,npols),idxin)
    ninsideidx[1] = nin

    fvals[idxin,1] .= fun.(grid[1,idxin],grid[2,idxin])	 

    rintl[0] = rint
    nbctr = 1
    ilevitr = 0
    @inbounds for ilev = 0:nlmax-1
	ilevitr = ilev 
	irefine = 0
	ifirstbox = laddr[1,ilev] 
	ilastbox = laddr[2,ilev]
	
	nbloc = ilastbox - ifirstbox + 1
	irefinebox = Vector{Int64}(undef,nbloc)
        
	if iptype == 2
	    rsc = sqrt(1.0 / boxsize[0]^2)
        elseif iptype == 1
	    rsc = 1.0 / boxsize[0]^2
        elseif iptype == 0
	    rsc = 1.0
	end

	rsc = rsc * rint
	println(ilev," ",rint," ",rsc)
	irefine, irefinebox = cutvol_tree_find_box_refine(iptype,eta,tol,norder,fvals,umat,boxsize[ilev],nbmax,ifirstbox,nbloc,rsc,irefinebox,icut,bdrytol,curve)
	if ilev < minlev
	    irefine = 1
	    irefinebox[:] .= 1
	end
        #		figure out if current set of boxes is sufficient
	nbadd = 0 
        @inbounds for i = 1:nbloc
	    if irefinebox[i] == 1
		nbadd = nbadd + 4
	    end
	end

	nbtot = nbctr + nbadd

        #         if current memory is not sufficient reallocate
	if nbtot > nbmax
	    println("Reallocating")
	    nbmax = 2 * nbmax
	    centers2 = Matrix{Float64}(undef,2,nbmax)
	    ilevel2 = Vector{Int64}(undef,nbmax)
	    iparent2 = Vector{Int64}(undef,nbmax)
	    nchild2 = Vector{Int64}(undef,nbmax)
	    ichild2 = Matrix{Int64}(undef,4,nbmax)
	    fvals2 = Matrix{Float64}(undef,npbox,nbmax)
	    rintbs2 = Vector{Float64}(undef,nbmax)
	    icut2 = Matrix{Int64}(undef,2,nbmax)
	    psis2 = Vector{Psistruct}(undef,nbmax) 
	    insideidx2 = Matrix{Int64}(undef,npbox,nbmax)
	    ninsideidx2 = Matrix{Int64}(undef,nbmax)

            vol_tree_copy(nbctr,npbox,centers,ilevel,iparent,nchild,ichild,fvals,icut,psis,insideidx,ninsideidx,centers2,ilevel2,iparent2,nchild2,ichild2,fvals2,icut2,psis2,insideidx2,ninsideidx2)
	    rintbs2 = rintbs

            nbmax = nbtot
       	    centers = Matrix{Float64}(undef,2,nbmax)
	    ilevel = Vector{Int64}(undef,nbmax)
	    iparent = Vector{Int64}(undef,nbmax)
	    nchild = Vector{Int64}(undef,nbmax)
	    ichild = Matrix{Int64}(undef,4,nbmax)
	    fvals = Matrix{Float64}(undef,npbox,nbmax)
	    rintbs = Vector{Float64}(undef,nbmax)
	    icut = Matrix{Int64}(undef,2,nbmax)
	    psis = Vector{Psistruct}(undef,nbmax)
	    insideidx = Matrix{Int64}(undef,npbox,nbmax)
	    ninsideidx = Vector{Int64}(undef,nbmax)

            vol_tree_copy(nbctr,npbox,centers2,ilevel2,iparent2,nchild2,ichild2,fvals2,icut2,psis2,insideidx2,ninsideidx2,centers,ilevel,iparent,nchild,ichild,fvals,icut,psis,insideidx,ninsideidx)
	    rintbs = rintbs2
  	end

	if irefine == 1
	    boxsize[ilev+1] = boxsize[ilev] / 2
            laddr[1,ilev+1] = nbctr + 1
 	    nbctr = cutvol_tree_refine_boxes(irefinebox,npbox,fvals,fun,grid,nbmax,ifirstbox,nbloc,centers,boxsize[ilev+1],nbctr,ilev+1,ilevel,iparent,nchild,ichild,xk,checkinside,icut,psis,insideidx,ninsideidx,curve)

            rsc = boxsize[ilev+1]^2/4
            rintbs, rint = update_rints(npbox,nbmax,fvals,ifirstbox,nbloc,iptype,nchild,ichild,wts2,rsc,rintbs,rint)
            rintl[ilev+1] = rint
            laddr[2,ilev+1] = nbctr
        else
            break
        end
    end

    nboxes = nbctr
    nlevels = ilevitr
    if nlevels >= 2
	nbtot = 8 * nboxes
	if nbtot > nbmax
	    println("Reallocating")
	    nbmax = 2 * nbmax
	    centers2 = Matrix{Float64}(undef,2,nbmax)
	    ilevel2 = Vector{Int64}(undef,nbmax)
	    iparent2 = Vector{Int64}(undef,nbmax)
	    nchild2 = Vector{Int64}(undef,nbmax)
	    ichild2 = Matrix{Int64}(undef,4,nbmax)
	    fvals2 = Matrix{Float64}(undef,npbox,nbmax)
	    rintbs2 = Vector{Float64}(undef,nbmax)
	    icut2 = Matrix{Int64}(undef,2,nbmax)
	    psis2 = Vector{Psistruct}(undef,nbmax)
	    insideidx2 = zeros(Int64,npbox,nbmax)
	    ninsideidx2 = zeros(Int64,nbmax)
	    
	    vol_tree_copy(nboxes,npbox,centers,ilevel,iparent,nchild,ichild,fvals,icut,psis,insideidx,ninsideidx,centers2,ilevel2,iparent2,nchild2,ichild2,fvals2,icut2,psis2,insideidx2,ninsideidx2)
	    rintbs2 = rintbs
	    
	    nbmax = nbtot
	    
	    centers = Matrix{Float64}(undef,2,nbmax)
	    ilevel = Vector{Int64}(undef,nbmax)
	    iparent = Vector{Int64}(undef,nbmax)
	    nchild = Vector{Int64}(undef,nbmax)
	    ichild = Matrix{Int64}(undef,4,nbmax)
	    fvals = Matrix{Float64}(undef,npbox,nbmax)
	    rintbs = Vector{Float64}(undef,nbmax)
	    icut = Matrix{Int64}(undef,2,nbmax)
	    psis = Vector{Psistruct}(undef,nbmax)
	    insideidx = zeros(Int64,npbox,nbmax)
	    ninsideidx = zeros(Int64,nbmax)
	    
	    vol_tree_copy(nboxes,npbox,centers2,ilevel2,iparent2,nchild2,ichild2,fvals2,icut2,psis2,insideidx2,ninsideidx2,centers,ilevel,iparent,nchild,ichild,fvals,icut,psis,insideidx,ninsideidx)
	    rintbs = rintbs2
	end

	nnbors = zeros(Int64,nbmax)
	
	nbors = ones(Int64,9,nbmax) * (-1)

        iper = 0
        computecoll(nlevels,nboxes,laddr,boxsize,centers,iparent,nchild,ichild,iper,nnbors,nbors)

	if nlevels >= 2 # Always satisfied
	    
	    nboxes = cutvol_tree_fix_lr(fun,norder,npbox,fvals,grid,centers,nlevels,nboxes,boxsize,nbmax,nlmax,laddr,ilevel,iparent,nchild,ichild,nnbors,nbors,xk,checkinside,icut,psis,insideidx,ninsideidx,bdrytol)
	    
	    nboxes = cutvol_tree_fix_lr(fun,norder,npbox,fvals,grid,centers,nlevels,nboxes,boxsize,nbmax,nlmax,laddr,ilevel,iparent,nchild,ichild,nnbors,nbors,xk,checkinside,icut,psis,insideidx,ninsideidx,bdrytol)
	    
	end
    end
    ltree = 17 * nboxes + 2 * (nlevels + 1)
    ncut = length(intersect(findall(nchild .== 0),findall(icut[2,:] .== 1)))

    return nboxes,nlevels,ltree, ncut
end #vol_tree_mem

function cutvol_tree_build(tol::Float64,boxlen::Float64,norder::Int64,iptype::Int64,eta::Float64,fun::Function,nlevels::Int64,nboxes::Int64,ltree::Int64,rintl::OffsetVector{Float64, Vector{Float64}},itree::Vector{Int64},iptr::Vector{Int64},fvals::Matrix{Float64},centers::Matrix{Float64},boxsize::OffsetVector{Float64, Vector{Float64}},checkinside::Function,xk::Matrix{Float64},minlev::Int64,psis::Vector{Psistruct},insideidx::Matrix{Int64},ninsideidx::Vector{Int64},icut::Matrix{Int64},bdrytol::Float64,curve)
    #=
    c
    c      compute the tree
    c
    c      input parameters:
    c        tol - double precision
    c           precision requested
    c        zk - double complex
    c           Helmholtz parameter
    c        boxlen - double precision
    c           length of box in which volume is contained, 
    c           if boxlen = L, then volume is [-L/2,L/2]^2
    c        norder - integer
    c           order of discretization
    c        iptype - integer
    c           error norm
    c           iptype = 0 - linf
    c           iptype = 1 - l1
    c           iptype = 2 - l2
    c        eta - double precision
    c           scaling parameter for error
    c        fun - function handle
    c           function to evalute it everywhere in the volume
    c        nd - integer
    c           number of real functions returned by fun
    c        nlevels - integer
    c          number of levels
    c        nboxes - integer
    c          number of boxes
    c        ltree - integer
    c          length of tree = 2*(nlevels+1)+17*nboxes
    c        rintl - real *8 (0:nlevels)
    c          estimate of lp norm for scaling the errors
    c          at various levels. 
    c          We require the estimate at each level to make sure
    c          that the memory estimate code is consitent
    c          with the build code else there could be potential
    c          memory issues 
    c         
    c
    c      output:
    c        itree - integer(ltree)
    c          tree info
    c        iptr - integer(8)
    c          iptr(1) - laddr
    c          iptr(2) - ilevel
    c          iptr(3) - iparent
    c          iptr(4) - nchild
    c          iptr(5) - ichild
    c          iptr(6) - ncoll
    c          iptr(7) - coll
    c          iptr(8) - ltree
    c        fvals - double precision (nd,norder**2,nboxes)
    c          function values at discretization nodes
    c        centers - double precision (2,nboxes)
    c          xyz coordinates of box centers in the oct tree
    c        boxsize - double precision (0:nlevels)
    c          size of box at each of the levels
    c
    =#
    xy = zeros(Float64,2)
    xq = Vector{Float64}(undef,norder)
    wts = Vector{Float64}(undef,norder)
    umat = Matrix{Float64}(undef,norder,norder)	
    vmat = Matrix{Float64}(undef,norder,norder)
    iptr[1] = 1
    iptr[2] = 2 * (nlevels + 1) + 1
    iptr[3] = iptr[2] + nboxes
    iptr[4] = iptr[3] + nboxes
    iptr[5] = iptr[4] + nboxes
    iptr[6] = iptr[5] + 4 * nboxes
    iptr[7] = iptr[6] + nboxes
    iptr[8] = iptr[7] + 9 * nboxes

    # Data structures for cut-boxes
    next = Vector{Int64}(undef,nboxes) # Number of boxes to extend to, we only store for leaf boxes
    ext = Matrix{Int64}(undef,24,nboxes) # Indices of boxes to extend to (including itself), we only stor cut-boxes that have no children

    boxsize[0] = boxlen

    centers[1,1] = 0
    centers[2,1] = 0


    #      set tree info for level 0

    itree[1] = 1
    itree[2] = 1
    itree[iptr[2]] = 0
    itree[iptr[3]] = -1
    itree[iptr[4]] = 0

    itree[iptr[5] .+ 1:4 .- 1] .= -1

    npbox = norder^2
    grid = Matrix{Float64}(undef,2,npbox)

    #     Generate a grid on the box [-1/2,1/2]^2

    itype = 2
    chebexps(itype,norder,xq,umat,vmat,wts)
    xq /= 2

    mesh2d(xq,norder,xq,norder,grid)

    idxin = checkinside(grid*boxlen)
    nin = length(idxin)
    
    icut[:,1] .= 1
    insideidx[1:nin,1] .= idxin
    insideidx[nin+1:end,1] = setdiff(range(1,npbox),idxin)
    ninsideidx[1] = nin
    #	end
    fvals[idxin,1] .= fun.(grid[1,idxin],grid[2,idxin])	 
    

    nbctr = 1
    ilevitr = 0
    @inbounds	for ilev = 0:nlevels-1
	ilevitr = ilev
	irefine = 0
	ifirstbox = itree[2*ilev+1] 
	ilastbox = itree[2*ilev+2]
	nbloc = ilastbox - ifirstbox + 1

        irefinebox = Vector{Int64}(undef,nbloc)
	if iptype == 2
	    rsc = sqrt(1.0 / boxsize[0]^2)
        elseif iptype == 1
	    rsc = 1.0 / boxsize[0]^2
	elseif iptype == 0
	    rsc = 1.0
	end


	
	rsc = rsc * rintl[ilev]
	irefine, irefinebox = cutvol_tree_find_box_refine(iptype,eta,tol,norder,fvals,umat,boxsize[ilev],nboxes,ifirstbox,nbloc,rsc,irefinebox,icut,bdrytol,curve)
	# Minimum numbers of levels
	if ilev < minlev
	    irefine = 1
	    irefinebox[:] .= 1
	end
	if irefine == 1
	    boxsize[ilev+1] = boxsize[ilev] / 2
	    itree[2*ilev+3] = nbctr + 1
	    nbctr = cutvol_tree_refine_boxes(irefinebox,npbox,fvals,fun,grid,nboxes,ifirstbox,nbloc,centers,boxsize[ilev+1],nbctr,ilev+1,view(itree,iptr[2]:iptr[3]-1),view(itree,iptr[3]:iptr[4]-1),view(itree,iptr[4]:iptr[5]-1),reshape(view(itree,iptr[5]:iptr[6]-1),(4,nboxes)),xk,checkinside,icut,psis,insideidx,ninsideidx,curve)
	    
	    itree[2*ilev+4] = nbctr
	else
	    break
        end
    end
    nboxes0 = nbctr
    nlevels = ilevitr + 1

    for i = 1:nboxes0
	itree[iptr[6]+i-1] = 0
	for j = 1:9
	    itree[iptr[7]+9*(i-1)+j-1] = -1
	end
    end
    iper = 0

    laddrpass = OffsetArray(Matrix{Int64}(undef,2,nlevels+1),1:2,0:nlevels)
    temp = reshape(itree[iptr[1]:iptr[2]-1],(2,(nlevels + 1)))
    for i = 1:2
	for j = 0:nlevels
	    laddrpass[i,j] = temp[i,j+1]
	end
    end

    computecoll(nlevels,nboxes0,laddrpass,boxsize,centers,itree[iptr[3]:iptr[4]-1],itree[iptr[4]:iptr[5]-1],reshape(itree[iptr[5]:iptr[6]-1],(4,nboxes)),iper,view(itree,iptr[6]:iptr[7]-1),reshape(view(itree,iptr[7]:iptr[8]-1),(9,nboxes)))
    
    if nlevels >= 2
	
	nboxes1 = cutvol_tree_fix_lr(fun,norder,npbox,fvals,grid,centers,nlevels,nboxes0,boxsize,nboxes,nlevels,laddrpass,view(itree,iptr[2]:iptr[3]-1),view(itree,iptr[3]:iptr[4]-1),view(itree,iptr[4]:iptr[5]-1),reshape(view(itree,iptr[5]:iptr[6]-1),(4,nboxes)),view(itree,iptr[6]:iptr[7]-1),reshape(view(itree,iptr[7]:iptr[8]-1),(9,nboxes)),xk,checkinside,icut,psis,insideidx,ninsideidx,bdrytol)

	itree[iptr[1]:iptr[2]-1] .= reshape(OffsetArray(laddrpass,1:2,1:nlevels+1),2*(nlevels+1))
    end

    for i = 1:nboxes1
	itree[iptr[6]+i-1] = 0
	for j = 1:9
	    itree[iptr[7]+9*(i-1)+j-1] = -1
	end
    end

    iper = 0

    laddrpass = OffsetArray(Matrix{Int64}(undef,2,nlevels+1),1:2,0:nlevels) 
    temp = reshape(itree[iptr[1]:iptr[2]-1],(2,(nlevels + 1)))
    for i = 1:2
	for j = 0:nlevels
	    laddrpass[i,j] = temp[i,j+1]
	end
    end

    computecoll(nlevels,nboxes1,laddrpass,boxsize,centers,itree[iptr[3]:iptr[4]-1],itree[iptr[4]:iptr[5]-1],reshape(itree[iptr[5]:iptr[6]-1],(4,nboxes)),iper,view(itree,iptr[6]:iptr[7]-1),reshape(view(itree,iptr[7]:iptr[8]-1),(9,nboxes)))
    
    if nlevels >= 2
	
	nboxes = cutvol_tree_fix_lr(fun,norder,npbox,fvals,grid,centers,nlevels,nboxes1,boxsize,nboxes,nlevels,laddrpass,view(itree,iptr[2]:iptr[3]-1),view(itree,iptr[3]:iptr[4]-1),view(itree,iptr[4]:iptr[5]-1),reshape(view(itree,iptr[5]:iptr[6]-1),(4,nboxes)),view(itree,iptr[6]:iptr[7]-1),reshape(view(itree,iptr[7]:iptr[8]-1),(9,nboxes)),xk,checkinside,icut,psis,insideidx,ninsideidx,bdrytol)

	itree[iptr[1]:iptr[2]-1] .= reshape(OffsetArray(laddrpass,1:2,1:nlevels+1),2*(nlevels+1))
    end
    return nboxes
end # vol_tree_build



function cutvol_tree_find_box_refine(iptype::Int64,eta::Float64,tol::Float64,norder::Int64,fvals::Matrix{Float64},umat::Matrix{Float64},boxsize::Float64,nboxes::Int64,ifirstbox::Int64,nbloc::Int64,rsc::Float64,irefinebox::Vector{Int64},icut::Matrix{Int64},bdrytol::Float64,curve)
    
    npbox = norder * norder
    fcoefs = Matrix{Float64}(undef,npbox,nbloc)
    iind2p = Matrix{Int64}(undef,2,npbox)
    legetens_ind2pow_2d(norder-1,'f',iind2p)
    rmask = Vector{Float64}(undef,npbox)

    rsum = 0
    for i = 1:npbox
	rmask[i] = 0.0
	i1 = iind2p[1,i] + iind2p[2,i]
	if i1 == norder-1
	    rmask[i] = 1.0
	    rsum = rsum + 1
	end
    end

    if iptype == 2
	rsum = sqrt(rsum)
    elseif iptype == 0
	rsum = 1
    end
    
    alpha = 1
    beta = 0
    bs = boxsize / 4.0
    bs2 = 2 * bs
    rscale2 = bs2^eta
    for i = 1:nbloc
	irefinebox[i] = 0
	ibox = ifirstbox + i - 1
	if (icut[2,ibox] == 1) & (boxsize > bdrytol) # 0.004 Harmonic ext (Askham). ex. # 0.002 VFMM ex.
	    irefinebox[i] = 1
	elseif (icut[2,ibox] != 1)
	    fcoefs[:,i] = legval2coefs_2d(norder,reshape(fvals[:,ibox],(norder,norder)),umat)
	    err = fun_err(npbox,reshape(fcoefs[:,i],(norder,norder)),rmask,iptype,rscale2)
	    err = err/rsum
	    if(err > tol*rsc)
	        irefinebox[i] = 1
	    end
	end
    end
    irefine = maximum(irefinebox[1:nbloc])
    #       make tree uniform
    ifunif = 0
    if ifunif == 1
	irefinebox[1:nbloc] .= irefine
    end
    return irefine, irefinebox
end # vol_tree_find_box_refine


# TODO: RETURN CORRECT ARGUMENTS
function cutvol_tree_refine_boxes(irefinebox::Vector{Int64},npbox::Int64,fvals::Matrix{Float64},fun::Function,grid::Matrix{Float64},nboxes::Int64,ifirstbox::Int64,nbloc::Int64,centers::Matrix{Float64},bs::Float64,nbctr::Int64,nlctr::Int64,ilevel::AbstractVector,iparent::AbstractVector,nchild::AbstractVector,ichild::AbstractMatrix,xk::Matrix{Float64},checkinside::Function,icut::Matrix{Int64},psis::Vector{Psistruct},insideidx::Matrix{Int64},ninsideidx::Vector{Int64},curve)


    isum = cumsum(irefinebox)
    bsh = bs/2
    xind = [-1,1,-1,1]
    yind = [-1,-1,1,1]
    xy = zeros(Float64,2,npbox)
    @inbounds	for i = 1:nbloc
	ibox = ifirstbox + i - 1
	if irefinebox[i] == 1
	    nbl = nbctr + (isum[i]-1)*4
	    nchild[ibox] = 4
            @inbounds for j = 1:4
		jbox = nbl+j
		centers[1,jbox] = centers[1,ibox] + xind[j] * bsh
		centers[2,jbox] = centers[2,ibox] + yind[j] * bsh
                #	 DEV = true
                #if DEV
		if (icut[1,ibox] == 0)
		    icut[:,jbox] .= 0
		    insideidx[:,jbox] .= range(1,npbox) # All points are inside
		    ninsideidx[jbox] = npbox
		    xy = centers[:,jbox] .+ grid * bs
		    fvals[:,jbox] .= fun.(xy[1,:],xy[2,:])	 
		elseif (icut[1,ibox] == 2)
		    icut[:,jbox] .= 2 # All points are outside
		    insideidx[:,jbox] .= range(1,npbox)
		    ninsideidx[jbox] = 0
		    fvals[:,jbox] .= 0.0
		elseif (icut[1,ibox] == 1)
		    # Cell might be cut. Check.
		    iscut = checkcut(centers[:,jbox],bs,curve)
		    if iscut # True
			icut[1,jbox] = 1
			xy = centers[:,jbox] .+ grid * bs
			idxin = checkinside(xy)
			nin = length(idxin)
			if nin == npbox
			    icut[2,jbox] = 0
			    insideidx[:,jbox] .= range(1,npbox) # All points are inside
			    ninsideidx[jbox] = npbox
			    fvals[:,jbox] .= fun.(xy[1,:],xy[2,:])	
			elseif nin == 0
			    icut[2,jbox] = 2
			    insideidx[:,jbox] .= range(1,npbox)
			    ninsideidx[jbox] = 0
			    fvals[:,jbox] .= 0.0
			else
			    icut[2,jbox] = 1
			    insideidx[1:nin,jbox] .= idxin
			    insideidx[nin+1:end,jbox] = setdiff(range(1,npbox),idxin)
			    ninsideidx[jbox] = nin
			    fvals[idxin,jbox] .= fun.(xy[1,idxin],xy[2,idxin])
			end
		    else # Check if inside or outside, enough to check center
			xy = centers[:,jbox] .+ grid * bs
			idxin = checkinside(xy)
			nin = length(idxin)
			if isempty(checkinside(centers[:,jbox])) # is outside
			    icut[:,jbox] .= 2 # All points are outside
			    insideidx[:,jbox] .= range(1,npbox)
			    ninsideidx[jbox] = 0
			    fvals[:,jbox] .= 0.0
			    if(nin != 0)
                                #								PyPlot.scatter(xy[1,:],xy[2,:])	 	  
                                #								PyPlot.scatter(xy[1,idxin],xy[2,idxin])	 
				println("OUTSIDE ERROR ", nin)
				#				icut[2,jbox] = 1
				insideidx[1:nin,jbox] .= idxin
				insideidx[nin+1:end,jbox] = setdiff(range(1,npbox),idxin)
				ninsideidx[jbox] = nin
				fvals[idxin,jbox] .= fun.(xy[1,idxin],xy[2,idxin])
			    end
			else # All points are inside
			    icut[:,jbox] .= 0
			    insideidx[:,jbox] .= range(1,npbox) # All points are inside
			    ninsideidx[jbox] = npbox
			    fvals[:,jbox] .= fun.(xy[1,:],xy[2,:])
			    if(nin != npbox)
				println("INSIDE ERROR ", nin)
                                #								PyPlot.scatter(xy[1,:],xy[2,:])	 	 
                                #								PyPlot.scatter(xy[1,idxin],xy[2,idxin])
				icut[2,jbox] = 1
				insideidx[1:nin,jbox] .= idxin
				insideidx[nin+1:end,jbox] = setdiff(range(1,npbox),idxin)
				ninsideidx[jbox] = nin
				fvals[idxin,jbox] .= fun.(xy[1,idxin],xy[2,idxin])
			    end
			end
		    end
		end



                iparent[jbox] = ibox
                nchild[jbox] = 0
		ichild[1:4,jbox] .= -1
                ichild[j,ibox] = jbox
                ilevel[jbox] = nlctr
		
	    end
	end
    end
    nbctr = nbctr + isum[nbloc] * 4
    return nbctr
end # vol_tree_refine_boxes

function update_rints(npbox::Int64,nbmax::Int64,fvals::Matrix{Float64},ifirstbox::Int64,nbloc::Int64,iptype::Int64,nchild::AbstractVector,ichild::AbstractMatrix,wts::Vector{Float64},rsc::Float64,rintbs::Vector{Float64},rint::Float64)
    #=
    c
    c------------------------
    c  This subroutine updates the integrals of the function to
    c  be resolved on the computational domain. It subtracts the
    c  integral of the boxes which have been refined and adds
    c  in the integrals corresponding to the function values 
    c  tabulated at the children
    c
    c  Input arguments:
    c  
    c    - nd: integer
    c        number of functions
    c    - npbox: integer
    c        number of points per box where the function is tabulated
    c    - nbmax: integer
    c        max number of boxes
    c    - fvals: real *8 (nd,npbox,nbmax)
    c        tabulated function values
    c    - ifirstbox: integer
    c        first box in the list of boxes to be processed
    c    - nbloc: integer
    c        number of boxes to be processed
    c    - iptype: integer
    c        Lp version of the scheme
    c        * iptype = 0, linf
    c        * iptype = 1, l1
    c        * iptype = 2, l2
    c    - nchild: integer(nbmax)
    c        number of children 
    c    - ichild: integer(8,nbmax)
    c        list of children
    c    - wts: real *8 (npbox)
    c        quadrature weights for intgegrating functions 
    c    - rsc: real *8
    c        scaling parameter for computing integrals
    c  
    c  Inout arguemnts:
    c
    c     - rintbs: real *8(nbmax)
    c         the integral for the new boxes cretated will be updated
    c     - rint: real *8
    c         the total integral will be updated
    c    
    c  
    c      
    implicit real *8 (a-h,o-z)
    integer, intent(in) :: nd,npbox,nbmax
    real *8, intent(in) :: fvals(nd,npbox,nbmax)
    integer, intent(in) :: ifirstbox,nbloc,iptype
    integer, intent(in) :: nchild(nbmax),ichild(4,nbmax)
    real *8, intent(in) :: wts(npbox),rsc
    real *8, intent(inout) :: rintbs(nbmax),rint
    =#

    #	compute the integrals for the newly formed boxes
    #   and update the overall integral

    if iptype == 0
	for i = 1:nbloc
	    ibox = ifirstbox + i - 1
	    if nchild[ibox] > 0 
		for j = 1:4
		    jbox = ichild[j,ibox]
		    rintbs[jbox] = maximum(fvals[1:npbox,jbox])
		    if rintbs[jbox] > rint
			rint = rintbs[jbox]
		    end
                end
	    end
	end
    end

    if iptype == 1
	for i = 1:nbloc
	    ibox = ifirstbox + i - 1
	    if nchild[ibox] > 0
                #     subtract contribution of ibox from rint
                rint = rint - rintbs[ibox]
	    end
	end


        #     add back contribution of children

	for i = 1:nbloc
	    ibox = ifirstbox + i - 1
	    if nchild[ibox] > 0
		for j = 1:4
		    jbox = ichild[j,ibox]
		    rintbs[jbox] = 0.0
		    for l = 1:npbox
			rintbs[jbox] = rintbs[jbox] + abs(fvals[l,jbox]) * wts[l] * rsc
		    end
		    rint = rint + rintbs[jbox]
		end
	    end
	end
    end
    
    #    note that if iptype = 2, then rintbs stores squares
    #    of the integral on the box
    if iptype == 2
	rintsq = rint^2
	for i = 1:nbloc
	    ibox = ifirstbox + i - 1
	    if nchild[ibox] > 0
		rintsq = rintsq - rintbs[ibox]
	    end
	end
	for i = 1:nbloc
	    ibox = ifirstbox + i - 1
	    if nchild[ibox] > 0
		for j = 1:4
		    jbox = ichild[j,ibox]
		    rintbs[jbox] = 0.0
		    for l = 1:npbox
			rintbs[jbox] = rintbs[jbox] + fvals[l,jbox]^2 * wts[l] * rsc
		    end
		    rintsq = rintsq + rintbs[jbox]
		end
	    end
	end
	rint = sqrt(rintsq)
    end
    return rintbs, rint
end #update_rints


function computecoll(nlevels::Int64,nboxes::Int64,laddr::AbstractMatrix,boxsize::OffsetVector{Float64, Vector{Float64}},centers::Matrix{Float64},iparent::AbstractVector,nchild::AbstractVector,ichild::AbstractMatrix,iper::Int64,nnbors::AbstractVector,nbors::AbstractMatrix)
    #=
    c     This subroutine computes the colleagues for an adaptive
    c     pruned tree. box j is a colleague of box i, if they share a
    c     vertex or an edge and the two boxes are at the same
    c     level in the tree
    c
    c     INPUT arguments
    c     nlevels     in: integer
    c                 Number of levels
    c
    c     nboxes      in: integer
    c                 Total number of boxes
    c
    c     laddr       in: integer(2,0:nlevels)
    c                 indexing array providing access to boxes at
    c                 each level. 
    c                 the first box on level i is laddr(1,i)
    c                 the last box on level i is laddr(2,i)
    c
    c     boxsize     in: double precision(0:nlevels)
    c                 Array of boxsizes
    c 
    c     centers     in: double precision(2,nboxes)
    c                 array of centers of boxes
    c   
    c     iparent     in: integer(nboxes)
    c                 iparent(i) is the box number of the parent of
    c                 box i
    c
    c     nchild      in: integer(nboxes)
    c                 nchild(i) is the number of children of box i
    c
    c     ichild      in: integer(4,nboxes)
    c                 ichild(j,i) is the box id of the jth child of
    c                 box i
    c
    c     iper        in: integer
    c                 flag for periodic implementations. 
    c                 Currently not used. Feature under construction.
    c
    c----------------------------------------------------------------
    c     OUTPUT
    c     nnbors      out: integer(nboxes)
    c                 nnbors(i) is the number of colleague boxes of
    c                 box i
    c
    c     nbors       out: integer(9,nboxes)
    c                 nbors(j,i) is the box id of the jth colleague
    c                 box of box i
    c---------------------------------------------------------------
    implicit none
    integer nlevels,nboxes
    integer iper
    integer laddr(2,0:nlevels)
    double precision boxsize(0:nlevels)
    double precision centers(2,nboxes)
    integer iparent(nboxes), nchild(nboxes), ichild(4,nboxes)
    integer nnbors(nboxes)
    integer nbors(9,nboxes)

    c     Temp variables
    integer ilev,ibox,jbox,kbox,dad
    integer i,j,ifirstbox,ilastbox
    =#

    #     Setting parameters for level = 0
    nnbors[1] = 1
    nbors[1,1] = 1
    @inbounds	for ilev = 1:nlevels
        #     Find the first and the last box at level ilev      
	ifirstbox = laddr[1,ilev]
	ilastbox = laddr[2,ilev]
        #        Loop over all boxes to evaluate neighbors, list1 and updating
        #        hunglists of targets

        #$OMP PARALLEL DO DEFAULT(SHARED)
        #$OMP$PRIVATE(ibox,dad,i,jbox,j,kbox)
        @inbounds	Threads.@threads	for ibox = ifirstbox:ilastbox
            #        Find the parent of the current box         
	    dad = iparent[ibox]
            #        Loop over the neighbors of the parent box
            #        to find out list 1 and list 2
	    for i = 1:nnbors[dad]
		jbox = nbors[i,dad]
                for j = 1:4
                    #               ichild[j,jbox] is one of the children of the
                    #		          neighbors of the parent of the current
                    #               box
		    kbox = ichild[j,jbox]
                    if kbox > 0
                        #               Check if kbox is a nearest neighbor or in list 2
			if (abs(centers[1,kbox]-centers[1,ibox]) <= 1.05*boxsize[ilev]) && (abs(centers[2,kbox]-centers[2,ibox]) <= 1.05*boxsize[ilev])           
			    nnbors[ibox] = nnbors[ibox] + 1
                            nbors[nnbors[ibox],ibox] = kbox
			end
		    end
		end
	    end
            #           End of computing colleagues of box i
	end
        #C$OMP END PARALLEL DO         
    end
    return
end # computecoll			  

function vol_tree_copy(nb::Int64,npb::Int64,centers::Matrix{Float64},ilevel::AbstractVector,iparent::AbstractVector,nchild::AbstractVector,ichild::AbstractMatrix,fvals::Matrix{Float64},icut::Matrix{Int64},psis::Vector{Psistruct},insideidx::Matrix{Int64},ninsideidx::Vector{Int64},centers2::Matrix{Float64},ilevel2::Vector{Int64},iparent2::Vector{Int64},nchild2::Vector{Int64},ichild2::Matrix{Int64},fvals2::Matrix{Float64},icut2::Matrix{Int64},psis2::Vector{Psistruct},insideidx2::Matrix{Int64},ninsideidx2::Vector{Int64})
    #=
    implicit none
    integer nd,nb,npb
    real *8 centers(2,nb),centers2(2,nb)
    integer ilevel(nb),ilevel2(nb)
    integer iparent(nb),iparent2(nb)
    integer nchild(nb),nchild2(nb)
    integer ichild(4,nb),ichild2(4,nb)
    real *8 fvals(nd,npb,nb),fvals2(nd,npb,nb)

    integer i,j,nel
    =#
    #	nel = npb*nb
    fvals2[:,1:nb] .= fvals[:,1:nb]
    insideidx2[:,1:nb] .= insideidx[:,1:nb]
    ninsideidx2[1:nb] .= ninsideidx[1:nb]
    centers2[1:2,1:nb] .= centers[1:2,1:nb]
    ilevel2[1:nb] = ilevel[1:nb]
    iparent2[1:nb] = iparent[1:nb]
    nchild2[1:nb] = nchild[1:nb]
    icut2[:,1:nb] = icut[:,1:nb]
    ichild2[1:4,1:nb] .= ichild[1:4,1:nb]
    #=
    @inbounds for i = 1:nb
    centers2[1:2,i] .= centers[1:2,i]
    ilevel2[i] = ilevel[i]
    iparent2[i] = iparent[i]
    nchild2[i] = nchild[i]
    icut2[i] = icut[i]
    if icut[i] == 1
    psis2[i] = psis[i]
    end
    ichild2[1:4,i] .= ichild[1:4,i]
    end
    =#
    return
end # vol_tree_copy


function cutvol_tree_fix_lr(fun::Function,norder::Int64,npbox::Int64,fvals::Matrix{Float64},grid::Matrix{Float64},centers::Matrix{Float64},nlevels::Int64,nboxes::Int64,boxsize::OffsetVector{Float64, Vector{Float64}},nbmax::Int64,nlmax::Int64,laddr::AbstractMatrix,ilevel::AbstractVector,iparent::AbstractVector,nchild::AbstractVector,ichild::AbstractMatrix,nnbors::AbstractVector,nbors::AbstractMatrix,xk::Matrix{Float64},checkinside::Function,icut::Matrix{Int64},psis::Vector{Psistruct},insideidx::Matrix{Int64},ninsideidx::Vector{Int64},bdrytol::Float64)

    #       convert an adaptive tree into a level restricted tree
    #		  For boxes cut by the boundary make sure that there are no coarse neighbors outside Omega.
    #=			  
    implicit none
    integer nd,norder,npbox,nlevels,nboxes,nlmax
    integer nbmax
    real *8 fvals(nd,npbox,nbmax),grid(2,npbox)
    real *8 centers(2,nbmax),boxsize(0:nlmax)
    integer laddr(2,0:nlmax),ilevel(nbmax),iparent(nbmax)
    integer nchild(nbmax),ichild(4,nbmax),nnbors(nbmax)
    integer nbors(9,nbmax)
    integer laddrtail(2,0:nlmax),isum
    integer, allocatable :: iflag(:)

    integer i,j,k,l,ibox,jbox,kbox,ilev,idad,igranddad
    integer nbloc,ict,iper
    real *8 xdis,ydis,distest

    external fun
    =#
    iflag = Vector{Int64}(undef,nbmax)
    laddrtail = OffsetArray(Matrix{Int64}(undef,2,nlmax+1),1:2,0:nlmax)
    
    #allocate(iflag(nbmax))

    #     Initialize flag array
    #C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)

    iflag[1:nboxes] .= 0
    #C$OMP END PARALLEL DO     


    #=
    c     Flag boxes that violate level restriction by "1"
    c     Violation refers to any box that is directly touching
    c     a box that is more than one level finer
    c
    c     Method:
    c     1) Carry out upward pass. For each box B, look at
    c     the colleagues of B's grandparent
    c     2) See if any of those colleagues are childless and in
    c     contact with B.
    c
    c     Note that we only need to get up to level two, as
    c     we will not find a violation at level 0 and level 1
    c
    c     For such boxes, we set iflag(i) = 1
    c

    PUX: Also, we will mark boxes that are outside with a fine neighbor that is cut. These boxes violate the boundary level-restriction
    =#
    @inbounds for ilev = nlevels:-1:2
        #        This is the distance to test if two boxes separated
        #        by two levels are touching
	distest = 1.05 * (boxsize[ilev-1] + boxsize[ilev-2]) / 2.0
	distest_cut = 1.05 * (boxsize[ilev] + boxsize[ilev-1]) / 2.0
        #C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox,idad,igranddad,i,jbox)         
        #C$OMP$ PRIVATE(ict,xdis,ydis,zdis)
        @inbounds Threads.@threads for ibox = laddr[1,ilev]:laddr[2,ilev] 
	    idad = iparent[ibox]
	    igranddad = iparent[idad]
            
            #           Loop over colleagues of granddad            
	    for i = 1:nnbors[igranddad]
		jbox = nbors[i,igranddad]
                #              Check if the colleague of grandad
                #		         is a leaf node. This automatically
                #              eliminates the granddad
		if (nchild[jbox] == 0) & (iflag[jbox] == 0)
		    xdis = centers[1,jbox] - centers[1,idad]
                    ydis = centers[2,jbox] - centers[2,idad]
                    ict = 0
                    if abs(xdis) <= distest
			ict = ict + 1
		    end
                    if abs(ydis) <= distest
			ict = ict + 1
		    end
                    if ict == 2
			iflag[jbox] = 1
                    end
		end
                #              End of checking criteria for the colleague of
                #              granddad
	    end
            #           End of looping over colleagues of
            #           granddad

            #        Check if boundary level restriction is violated.			 
            #        If ibox is cut, then check if colleagues of dad are coarse neigbors.	#        Then check if they are outside and leaf nodes. Such boxes violate the level-restriction on the boundary. 
	    if icut[2,ibox] == 1
                #				Loop over colleagues of parent
		for i = 1:nnbors[idad]
		    jbox = nbors[i,idad]
                    #           Check if the colleague of dad is a leaf node and not already
                    #   			flaged and outside
		    if (nchild[jbox] == 0) & (iflag[jbox] == 0) & (icut[2,jbox] == 2)
			xdis = centers[1,jbox] - centers[1,ibox]
                        ydis = centers[2,jbox] - centers[2,ibox]
                        ict = 0
                        if abs(xdis) <= distest_cut
			    ict = ict + 1
			end
                        if abs(ydis) <= distest_cut
			    ict = ict + 1
			end
                        if ict == 2
			    iflag[jbox] = 1
                        end
		    end
		    
		end
	    end
	    
	end
        #        End of looping over boxes at ilev         
        #C$OMP END PARALLEL DO
    end
    #     End of looping over levels and flagging boxes
    
    #=
    c     Find all boxes that need to be given a flag+
    c     A flag+ box will be denoted by setting iflag(box) = 2
    c     This refers to any box that is not already flagged and
    c     is bigger than and is contacting a flagged box
    c     or another box that has already been given a flag +.
    c     It is found by performing an upward pass and looking
    c     at the flagged box's parents colleagues and a flag+
    c     box's parents colleagues and seeing if they are
    c     childless and present the case where a bigger box 
    c     is contacting a flagged or flag+ box.

    Boundary level restriction: Find boxes outside that are leaf nodes and 
    colleagues of flagged (iflag[box] = 1) cut boxes. These boxes are given	flag+. NO! Not always neccessary to subdivide. Ignore these for now.
    =#
    
    @inbounds	for ilev = nlevels:-1:1
        #        This is the distance to test if two boxes separated
        #        by one level are touching
	distest = 1.05 * (boxsize[ilev] + boxsize[ilev-1]) / 2.0
        #C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox,idad,i,jbox,xdis,ydis)
        #C$OMP$PRIVATE(zdis,ict)
        @inbounds	Threads.@threads	for ibox = laddr[1,ilev]:laddr[2,ilev]
	    if iflag[ibox] == 1 || iflag[ibox] == 2 
		idad = iparent[ibox]
                #              Loop over dad's colleagues               
                for i = 1:nnbors[idad]
		    jbox = nbors[i,idad]
                    #                 Check if the colleague of dad
                    #                 is a leaf node. This automatically
                    #                 eliminates the dad
		    if nchild[jbox] == 0 && iflag[jbox] == 0
			xdis = centers[1,jbox] - centers[1,ibox]
                        ydis = centers[2,jbox] - centers[2,ibox]
                        ict = 0
                        if abs(xdis) <= distest
			    ict = ict + 1
			end
                        if abs(ydis) <= distest
			    ict = ict + 1
			end				
                        if ict == 2
			    iflag[jbox] = 2
                        end
		    end
                    #                 End of checking criteria for the colleague of
                    #                dad
		end
                #              End of looping over dad's colleagues

                #					Now loop over colleagues. If ibox is a cut-cell and colleagues are at the same level and outside, then they need to be refined
                #					if icut[ibox] == 1
                #						for i = 1:nnbors[ibox]
                #							jbox = nbors[i,ibox]
                #							Check if the colleague is outside and leaf boxs and not already outside
                #							if nchild[jbox] == 0 & iflag[jbox] == 0 & icut[jbox] == 2
                #								iflag[jbox] = 1 # This box also needs to be checked later, thus not flag+
                #							end
                #						end      
                #					end
	    end
            #           End of checking if current box is relevant for
            #           flagging flag+ boxes
	end
        #        End of looping over boxes at ilev        
        #C$OMP END PARALLEL DO 
    end
    
    #     End of looping over levels
    #=
    c     Subdivide all flag and flag+ boxes. Flag all the children
    c     of flagged boxes as flag++. Flag++ boxes are denoted
    c     by setting iflag(box) = 3. The flag++ boxes need 
    c     to be checked later to see which of them need further
    c     refinement. While creating new boxes, we will
    c     need to update all the tree structures as well.
    c     Note that all the flagged boxes live between
    c     levels 1 and nlevels - 2. We process the boxes via a
    c     downward pass. We first determine the number of boxes
    c     that are going to be subdivided at each level and 
    c     everything else accordingly
    =#
    laddrtail[1,0:nlevels] .= 0
    laddrtail[2,0:nlevels] .= -1
    
    @inbounds	for ilev = 1:nlevels-2
        #=			  
        c        First subdivide all the flag and flag+
        c        boxes with boxno nboxes+1, nboxes+ 2
        c        and so on. In the second step, we reorganize
        c        all the structures again to bring it back
        c        in the standard format
        =#
	laddrtail[1,ilev+1] = nboxes + 1

	nbloc = laddr[2,ilev] - laddr[1,ilev] + 1

	
	nboxes = cutvol_tree_refine_boxes_flag(iflag,npbox,fvals,fun,grid,nbmax,laddr[1,ilev],nbloc,centers,boxsize[ilev+1],nboxes,ilev,ilevel,iparent,nchild,ichild,xk,checkinside,icut,psis,insideidx,ninsideidx)
	laddrtail[2,ilev+1] = nboxes
    end
    #     Reorganize the tree to get it back in the standard format

    
    vol_tree_reorg(nboxes,npbox,centers,nlevels,laddr,laddrtail,ilevel,iparent,nchild,ichild,fvals,iflag,icut,psis,insideidx,ninsideidx)

    #     Compute colleague information again      

    #C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
    nnbors[1:nboxes] .= 0

    nbors[1:9,1:nboxes] .= -1
    
    #C$OMP END PARALLEL DO     
    iper = 0
    computecoll(nlevels,nboxes,laddr, boxsize,centers,iparent,nchild,ichild,iper,nnbors,nbors)
    #=			  
    c     Processing of flag and flag+ boxes is done
    c     Start processing flag++ boxes. We will use a similar
    c     strategy as before. We keep checking the flag++
    c     boxes that require subdivision if they still
    c     violate the level restriction criterion, create
    c     the new boxes, append them to the end of the list to begin
    c     with and in the end reorganize the tree structure.
    c     We shall accomplish this via a downward pass
    c     as new boxes that get added in the downward pass
    c     will also be processed simultaneously.
    c     We shall additionally also need to keep on updating
    c     the colleague information as we proceed in the 
    c     downward pass

    c     Reset the flags array to remove all the flag and flag+
    c     cases. This is to ensure reusability of the subdivide
    c     _flag routine to handle the flag++ case
    =#
    
    #C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox)
    #	for ibox = 1:nboxes
    #		if iflag[ibox] != 3 
    #			iflag[ibox] = 0
    #		end
    #	end

    iflag[iflag .!= 3] .= 0 

    #C$OMP END PARALLEL DO      
    
    laddrtail[1,0:nlevels] .= 0
    laddrtail[2,0:nlevels] .= -1

    for ilev = 2:nlevels-1
        #=
        c     Step 1: Determine which of the flag++ boxes need
        c     further division. In the event a flag++ box needs
        c     further subdivision then flag the box with iflag(box) = 1
        c     This will again ensure that the subdivide_flag routine
        c     will take care of handling the flag++ case
        =#
	vol_updateflags(ilev,nboxes,nlevels,laddr,iparent,nchild,ichild,nnbors,nbors,centers,boxsize,iflag,icut,checkinside,grid,npbox,bdrytol)

	vol_updateflags(ilev,nboxes,nlevels,laddrtail,iparent,nchild,ichild,nnbors,nbors,centers,boxsize,iflag,icut,checkinside,grid,npbox,bdrytol)
	
        #      Step 2: Subdivide all the boxes that need subdivision
        #      in the laddr set and the laddrtail set as well
	laddrtail[1,ilev+1] = nboxes + 1

	nbloc = laddr[2,ilev]-laddr[1,ilev]+1

	nboxes = cutvol_tree_refine_boxes_flag(iflag,npbox,fvals,fun,grid,nbmax,laddr[1,ilev],nbloc,centers,boxsize[ilev+1],nboxes,ilev,ilevel,iparent,nchild,ichild,xk,checkinside,icut,psis,insideidx,ninsideidx)

	nbloc = laddrtail[2,ilev] - laddrtail[1,ilev] + 1

	nboxes = cutvol_tree_refine_boxes_flag(iflag,npbox,fvals,fun,grid,nbmax,laddrtail[1,ilev],nbloc,centers,boxsize[ilev+1],nboxes,ilev,ilevel,iparent,nchild,ichild,xk,checkinside,icut,psis,insideidx,ninsideidx)
	
	laddrtail[2,ilev+1] = nboxes         
        #      Step 3: Update the colleague information for the newly
        #      created boxes

        #C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox,i,idad,jbox,j,kbox)
        Threads.@threads	for ibox = laddrtail[1,ilev+1]:laddrtail[2,ilev+1]
	    nnbors[ibox] = 0
            #           Find the parent of the current box         
	    idad = iparent[ibox]
            #           Loop over the neighbors of the parent box
            #           to find out colleagues
	    for i = 1:nnbors[idad]
		jbox = nbors[i,idad]
                for j = 1:4
                    #               ichild[j,jbox] is one of the children of the
                    #               neighbors of the parent of the current
                    #               box
		    kbox = ichild[j,jbox]
                    if kbox >= 0
                        #               Check if kbox is a nearest neighbor or in list 2
			if (abs(centers[1,kbox]-centers[1,ibox]) <= 1.05*boxsize[ilev+1]) && (abs(centers[2,kbox]-centers[2,ibox]) <= 1.05*boxsize[ilev+1])
			    nnbors[ibox] = nnbors[ibox] + 1
			    nbors[nnbors[ibox],ibox] = kbox
                        end
                    end
                end
	    end
            #           End of computing colleagues of box i
	end
        #C$OMP END PARALLEL DO         
    end
    
    #     Reorganize tree once again and we are all done      
    vol_tree_reorg(nboxes,npbox,centers,nlevels,laddr,laddrtail,ilevel,iparent,nchild,ichild,fvals,iflag,icut,psis,insideidx,ninsideidx)

    #     Compute colleague information again      

    #C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
    #	for i = 1:nboxes
    #		nnbors[i] = 0
    #		for j = 1:9
    #			nbors[j,i] = -1
    #		end
    #	end
    nnbors[1:nboxes] .= 0
    nbors[1:9,1:nboxes] .= -1
    #C$OMP END PARALLEL DO    
    computecoll(nlevels,nboxes,laddr, boxsize,centers,iparent,nchild,ichild,iper,nnbors,nbors)
    return nboxes
end # vol_tree_fix_lr


function cutvol_tree_refine_boxes_flag(iflag::Vector{Int64},npbox::Int64,fvals::Matrix{Float64},fun::Function,grid::Matrix{Float64},nboxes::Int64,ifirstbox::Int64,nbloc::Int64,centers::Matrix{Float64},bs::Float64,nbctr::Int64,nlctr::Int64,ilevel::AbstractVector,iparent::AbstractVector,nchild::AbstractVector,ichild::AbstractMatrix,xk::Matrix{Float64},checkinside::Function,icut::Matrix{Int64},psis::Vector{Psistruct},insideidx::Matrix{Int64},ninsideidx::Vector{Int64})
    #=
    implicit none
    integer nd,npbox,nboxes
    real *8 fvals(nd,npbox,nboxes)
    integer nbloc,nbctr,nlctr
    real *8 centers(2,nboxes),bs,grid(2,npbox),xy(2)
    integer ilevel(nboxes),iparent(nboxes)
    integer ichild(4,nboxes),nchild(nboxes)
    integer iflag(nboxes)
    integer ifirstbox,ilastbox
    integer, allocatable :: isum(:)

    integer i,ibox,nel,j,l,jbox,nbl,ii
    integer xind(4),yind(4)

    real *8 bsh
    data xind/-1,1,-1,1/
    data yind/-1,-1,1,1/

    external fun
    =#

    
    xind = [-1,1,-1,1]
    yind = [-1,-1,1,1]
    xy = Vector{Float64}(undef,2)
    ilastbox = ifirstbox + nbloc - 1
    
    bsh = bs / 2

    isum = cumsum(iflag[ifirstbox:(ifirstbox + nbloc-1)] .> 0)
    #C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox,j,jbox,nbl,l)
    #C$OMP$PRIVATE(xy)
    @inbounds for ibox = ifirstbox:ilastbox
	if iflag[ibox] > 0
	    nchild[ibox] = 4
	    nbl = nbctr + (isum[ibox-ifirstbox+1]-1) * 4
	    for j = 1:4
		jbox = nbl + j
		centers[1,jbox] = centers[1,ibox] + xind[j] * bsh
		centers[2,jbox] = centers[2,ibox] + yind[j] * bsh

                #				DEV = true
                #if DEV
	        if (icut[1,ibox] == 0)
		    icut[:,jbox] .= 0
		    insideidx[:,jbox] .= range(1,npbox) # All points are inside
		    ninsideidx[jbox] = npbox
		    xy = centers[:,jbox] .+ grid * bs
		    fvals[:,jbox] .= fun.(xy[1,:],xy[2,:])	 
		elseif (icut[1,ibox] == 2)
		    icut[:,jbox] .= 2 # All points are outside
		    insideidx[:,jbox] .= range(1,npbox)
		    ninsideidx[jbox] = 0
		elseif (icut[1,ibox] == 1)
		    # Cell might be cut. Check.
		    iscut = checkcut(centers[:,jbox],bs,curve)
		    if iscut # True
			icut[1,jbox] = 1
			xy = centers[:,jbox] .+ grid * bs
			idxin = checkinside(xy)
			nin = length(idxin)
			if nin == npbox
			    icut[2,jbox] = 0
			    insideidx[:,jbox] .= range(1,npbox) # All points are inside
			    ninsideidx[jbox] = npbox
			    fvals[:,jbox] .= fun.(xy[1,:],xy[2,:])	
			elseif nin == 0
			    icut[2,jbox] = 2
			    insideidx[:,jbox] .= range(1,npbox)
			    ninsideidx[jbox] = 0
			else
			    icut[2,jbox] = 1
			    insideidx[1:nin,jbox] .= idxin
			    insideidx[nin+1:end,jbox] = setdiff(range(1,npbox),idxin)
			    ninsideidx[jbox] = nin
			    fvals[idxin,jbox] .= fun.(xy[1,idxin],xy[2,idxin])
			end
		    else # Check if inside or outside, enough to check center
			if isempty(checkinside(centers[:,jbox])) # is outside
			    icut[:,jbox] .= 2 # All points are outside
			    insideidx[:,jbox] .= range(1,npbox)
			    ninsideidx[jbox] = 0
			else # All points are inside
			    icut[:,jbox] .= 0
			    insideidx[:,jbox] .= range(1,npbox) # All points are inside
			    ninsideidx[jbox] = npbox
			    xy = centers[:,jbox] .+ grid * bs  
			    fvals[:,jbox] .= fun.(xy[1,:],xy[2,:])	  
			end
		    end
		end
	        #=
                else # ORIGINAL
		xy = centers[:,jbox] .+ grid * bs
		idxin = checkinside(xy)
		nin = length(idxin)
		#	if (icut[ibox] == 0)
		if nin == npbox
		icut[2,jbox] = 0
		insideidx[:,jbox] .= range(1,npbox) # All points are inside
		ninsideidx[jbox] = npbox
		fvals[idxin,jbox] .= fun.(xy[1,idxin],xy[2,idxin])	 
		#		elseif icut[ibox] == 2
		elseif nin == 0
		icut[2,jbox] = 2 # All points are outside
		insideidx[:,jbox] .= range(1,npbox)
		ninsideidx[jbox] = 0
		fvals
		else # Need to check if cut,  icut =  1 if a cut box, 0 if inside, 2 if outside
		#	idxin = checkinside(xy)
                #					nin = length(idxin)
		insideidx[1:nin,jbox] .= idxin
		insideidx[nin+1:end,jbox] = setdiff(range(1,npbox),idxin)
		ninsideidx[jbox] = nin
		
		icut[2,jbox] = 1
		fvals[idxin,jbox] .= fun.(xy[1,idxin],xy[2,idxin])	 
		
		end

                end
		=#
                #=
		
		=#
                #				fun_ext_singlebox(view(fvals,:,jbox),fun,xk,grid,checkinside,centers[:,jbox],bs,icut[jbox],view(psis,jbox),view(insideidx,1:ninsideidx[jbox],jbox))
                #				igrid = centers[:,jbox] .+ grid * bs
                #				fvals[:,jbox] .= fun.(igrid[1,:],igrid[2,:])
		iparent[jbox] = ibox
                nchild[jbox] = 0

		ichild[1:4,jbox] .= -1
                ichild[j,ibox] = jbox
                ilevel[jbox] = nlctr+1 
                if iflag[ibox] == 1
		    iflag[jbox] = 3
		end
                if iflag[ibox] == 2
		    iflag[jbox] = 0
		end
	    end
	end
    end
    #C$OMP END PARALLEL DO
    
    if nbloc > 0
	nbctr = nbctr + isum[nbloc]*4
    end

    return nbctr
end #vol_tree_refine_boxes_flag								

function vol_tree_reorg(nboxes::Int64,npbox::Int64,centers::Matrix{Float64},nlevels::Int64,laddr::AbstractMatrix,laddrtail::AbstractMatrix,ilevel::AbstractVector,iparent::AbstractVector,nchild::AbstractVector,ichild::AbstractMatrix,fvals::Matrix{Float64},iflag::Vector{Int64},icut::Matrix{Int64},psis::Vector{Psistruct},insideidx::Matrix{Int64},ninsideidx::Vector{Int64})
    #=
    c    This subroutine reorganizes the current data in all the tree
    c    arrays to rearrange them in the standard format.
    c    The boxes on input are assumed to be arranged in the following
    c    format
    c    boxes on level i are the boxes from laddr(1,i) to 
    c    laddr(2,i) and also from laddrtail(1,i) to laddrtail(2,i)
    c
    c    At the end of the sorting, the boxes on level i
    c    are arranged from laddr(1,i) to laddr(2,i)  
    c
    c    INPUT/OUTPUT arguments
    c    nboxes         in: integer
    c                   number of boxes
    c
    c    nd             in: integer
    c                   number of real value functions
    c
    c    npbox          in: integer
    c                   number of grid points per function
    c
    c    centers        in/out: double precision(3,nboxes)
    c                   x and y coordinates of the center of boxes
    c
    c    nlevels        in: integer
    c                   Number of levels in the tree
    c
    c    laddr          in/out: integer(2,0:nlevels)
    c                   boxes at level i are numbered between
    c                   laddr(1,i) to laddr(2,i)
    c
    c    laddrtail      in: integer(2,0:nlevels)
    c                   new boxes to be added to the tree
    c                   structure are numbered from
    c                   laddrtail(1,i) to laddrtail(2,i)
    c
    c    ilevel      in/out: integer(nboxes)
    c                ilevel(i) is the level of box i
    c
    c    iparent     in/out: integer(nboxes)
    c                 iparent(i) is the parent of box i
    c
    c    nchild      in/out: integer(nboxes)
    c                nchild(i) is the number of children 
    c                of box i
    c
    c    ichild       in/out: integer(8,nboxes)
    c                 ichild(j,i) is the jth child of box i
    c
    c    iflag        in/out: integer(nboxes)
    c                 iflag(i) is a flag for box i required to generate
    c                 level restricted tree from adaptive tree

    implicit none
    c     Calling sequence variables and temporary variables
    integer nboxes,nlevels,npbox,nd
    double precision centers(2,nboxes)
    integer laddr(2,0:nlevels), tladdr(2,0:nlevels)
    integer laddrtail(2,0:nlevels)
    integer ilevel(nboxes)
    integer iparent(nboxes)
    integer nchild(nboxes)
    integer ichild(4,nboxes)
    integer iflag(nboxes)
    double precision fvals(nd,npbox,nboxes)
    =#      
    #      integer, allocatable :: tilevel(:),tiparent(:),tnchild(:)
    #      integer, allocatable :: tichild(:,:),tiflag(:)
    #      integer, allocatable :: iboxtocurbox(:),ilevptr(:),ilevptr2(:)

    #      double precision, allocatable :: tfvals(:,:,:),tcenters(:,:)

    #      allocate(tilevel(nboxes),tiparent(nboxes),tnchild(nboxes))
    
    tilevel = Vector{Int64}(undef,nboxes)
    tiparent = Vector{Int64}(undef,nboxes)
    tnchild = Vector{Int64}(undef,nboxes)
    tiflag = Vector{Int64}(undef,nboxes)
    iboxtocurbox = Vector{Int64}(undef,nboxes)
    tichild = Matrix{Int64}(undef,4,nboxes)
    tfvals = Matrix{Float64}(undef,npbox,nboxes)
    tcenters = Matrix{Float64}(undef,2,nboxes)
    tladdr = OffsetArray(Matrix{Int64}(undef,2,nlevels+1),1:2,0:nlevels)

    ticut = Matrix{Int64}(undef,2,nboxes)
    tpsis = Vector{Psistruct}(undef,nboxes)
    tinsideidx = zeros(Int64,npbox,nboxes)
    tninsideidx = zeros(Int64,nboxes)
    #   allocate(tichild(4,nboxes),tiflag(nboxes),iboxtocurbox(nboxes))
    #  allocate(tfvals(nd,npbox,nboxes),tcenters(2,nboxes))

    for ilev = 0:nlevels
	tladdr[1,ilev] = laddr[1,ilev]
	tladdr[2,ilev] = laddr[2,ilev]
    end
    
    vol_tree_copy(nboxes,npbox,centers,ilevel,iparent,nchild,ichild,fvals,icut,psis,insideidx,ninsideidx,tcenters,tilevel,tiparent,tnchild,tichild,tfvals,ticut,tpsis,tinsideidx,tninsideidx)

    for ibox = 1:nboxes
	tiflag[ibox] = iflag[ibox]
    end
    
    #     Rearrange old arrays now

    for ilev = 0:1
	for ibox = laddr[1,ilev]:laddr[2,ilev]
	    iboxtocurbox[ibox] = ibox
	end
    end

    #allocate(ilevptr(nlevels+1),ilevptr2(nlevels))
    ilevptr = Vector{Int64}(undef,nlevels+1)
    ilevptr2 = Vector{Int64}(undef,nlevels)
    ilevptr[2] = laddr[1,2]


    for ilev = 2:nlevels
	nblev = laddr[2,ilev] - laddr[1,ilev] + 1
        ilevptr2[ilev] = ilevptr[ilev] + nblev
        nblev = laddrtail[2,ilev] - laddrtail[1,ilev] + 1
        ilevptr[ilev+1] = ilevptr2[ilev] + nblev
    end

    curbox = laddr[1,2]
    for ilev = 2:nlevels
	laddr[1,ilev] = curbox
	for ibox = tladdr[1,ilev]:tladdr[2,ilev]
	    ilevel[curbox] = tilevel[ibox]
	    nchild[curbox] = tnchild[ibox]
	    centers[1,curbox] = tcenters[1,ibox]
	    centers[2,curbox] = tcenters[2,ibox]
	    for i = 1:npbox
		fvals[i,curbox] = tfvals[i,ibox]
		insideidx[i,curbox] = tinsideidx[i,ibox]
	    end
	    ninsideidx[curbox] = tninsideidx[ibox]
	    iflag[curbox] = tiflag[ibox] 
	    icut[:,curbox] .= ticut[:,ibox]
	    iboxtocurbox[ibox] = curbox
	    curbox = curbox + 1
	end
	for ibox = laddrtail[1,ilev]:laddrtail[2,ilev]
	    ilevel[curbox] = tilevel[ibox]
	    centers[1,curbox] = tcenters[1,ibox]
	    centers[2,curbox] = tcenters[2,ibox]
	    nchild[curbox] = tnchild[ibox]
	    for i = 1:npbox
		fvals[i,curbox] = tfvals[i,ibox]
		insideidx[i,curbox] = tinsideidx[i,ibox]
	    end
	    ninsideidx[curbox] = tninsideidx[ibox]
	    iflag[curbox] = tiflag[ibox]
	    icut[:,curbox] = ticut[:,ibox]
	    iboxtocurbox[ibox] = curbox
	    curbox = curbox + 1
	end
	laddr[2,ilev] = curbox-1
    end

    #     Handle the parent children part of the tree 
    #     using the mapping iboxtocurbox

    
    for ibox = 1:nboxes
	if tiparent[ibox] == -1
	    iparent[iboxtocurbox[ibox]] = -1
	end
	if tiparent[ibox] > 0
	    iparent[iboxtocurbox[ibox]] = iboxtocurbox[tiparent[ibox]]
	end
	for i = 1:4
	    if tichild[i,ibox] == -1
		ichild[i,iboxtocurbox[ibox]] = -1
	    end
	    if tichild[i,ibox] > 0
		ichild[i,iboxtocurbox[ibox]] = iboxtocurbox[tichild[i,ibox]]
	    end
	end
    end

    return
end # vol_tree_reorg

function vol_updateflags(curlev::Int64,nboxes::Int64,nlevels::Int64,laddr::AbstractMatrix,iparent::AbstractVector,nchild::AbstractVector,ichild::AbstractMatrix,nnbors::AbstractVector,nbors::AbstractMatrix,centers::Matrix{Float64},boxsize::OffsetVector{Float64, Vector{Float64}},iflag::Vector{Int64},icut::Matrix{Int64},checkinside::Function,grid::Matrix{Float64},npbox::Int64,bdrytol::Float64)
    #=
    c      This subroutine is to check the boxes flagged as flag++
    c      and determine which of the boxes need refinement. The flag
    c      of the box which need refinement is updated to iflag(box)=1
    c      and that of the boxes which do not need refinement is
    c      updated to iflag(box) = 0
    c
    c      INPUT arguments
    c      curlev         in: integer
    c                     the level for which boxes need to be processed
    c
    c      nboxes         in: integer
    c                     total number of boxes
    c
    c      nlevels        in: integer
    c                     total number of levels
    c
    c      laddr          in: integer(2,0:nlevels)
    c                     boxes from laddr(1,ilev) to laddr(2,ilev)
    c                     are at level ilev
    c
    c      nchild         in: integer(nboxes)
    c                     nchild(ibox) is the number of children
    c                     of box ibox
    c
    c      ichild         in: integer(4,nboxes)
    c                     ichild(j,ibox) is the box id of the jth
    c                     child of box ibox
    c
    c      nnbors         in: integer(nboxes)
    c                     nnbors(ibox) is the number of colleagues
    c                     of box ibox
    c
    c      nbors          in: integer(9,nboxes)
    c                     nbors(j,ibox) is the jth colleague of box
    c                     ibox
    c
    c      centers        in: double precision(3,nboxes)
    c                     x and y coordinates of the box centers
    c
    c      boxsize        in: double precision(0:nlevels)
    c                     boxsize(i) is the size of the box at level i
    c
    c      iflag          in/out: integer(nboxes)
    c                     iflag(ibox)=3 if it is flag++. iflag(ibox) =1
    c                     or 0 at the end of routine depending on
    c                     whether box needs to be subdivided or not
    c
    implicit none
    c     Calling sequence variables
    integer curlev, nboxes, nlevels
    integer laddr(2,0:nlevels),nchild(nboxes),ichild(4,nboxes)
    integer nnbors(nboxes), nbors(9,nboxes)
    integer iflag(nboxes)
    double precision centers(2,nboxes),boxsize(0:nlevels)

    c     Temporary variables
    integer i,j,k,l,ibox,jbox,kbox,lbox, ict
    double precision distest,xdis,ydis
    =#

    distest = 1.05 * (boxsize[curlev] + boxsize[curlev+1]) / 2.0
    distest_cut = 1.05 * (boxsize[curlev] + boxsize[curlev-1]) / 2.0

    
    #     Loop over all boxes at the current level     
    #C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox,i,jbox,j,kbox,xdis,ydis)
    #C$OMP$PRIVATE(zdis,ict)
    for ibox = laddr[1,curlev]:laddr[2,curlev]
	if (iflag[ibox] == 3)
	    iflag[ibox] = 0
            #           Loop over colleagues of the current box

	    for i = 1:nnbors[ibox]
                #              Loop over colleagues of flag++ box        
		jbox = nbors[i,ibox]				 
                #=             
                c              Loop over the children of the colleague box
                c              Note we do not need to exclude self from
                c              the list of colleagues as a self box which
                c              is flag++ does not have any children 
                c              and will not enter the next loop
                =#
		for j = 1:4
		    kbox = ichild[j,jbox]
		    if kbox > 0
			# DO I NEED THIS ONE?
			# The idea is to mark this box if it is outside, is leaf and in contact with one level down. We check this later on, so shouldn't need this. Also, we need to check if it is in contact with a box one level down that is cut.
			#	if icut[jbox] == 1
			#		iflag[ibox] = 1
			#		@goto cont
			#	end
			if nchild[kbox] > 0
			    xdis = centers[1,kbox] - centers[1,ibox]
			    ydis = centers[2,kbox] - centers[2,ibox]
			    ict = 0
			    if abs(xdis) <= distest
				ict = ict + 1
			    end
			    if abs(ydis) <= distest
				ict = ict + 1
			    end
			    if ict == 2
				iflag[ibox] = 1
				@goto cont 
			    end
			end
		    end
                    #                 End of looping over the children of the child
                    #                 of the colleague box
		end
                #              End of looping over the children of the colleague box       
	    end
            #           End of looping over colleagues	
	    @label cont
	    continue        
	end
	
        #        End of testing if the current box needs to checked for         
    end
    #     End of looping over boxes at the current level      
    #$OMP END PARALLEL DO      
    # We need to check all boxes again. A boundary box may now have been flagged, thus must check boundary level restriction.
    xind = [-1,1,-1,1]
    yind = [-1,-1,1,1]
    xy = Vector{Float64}(undef,2)
    bs = boxsize[curlev]
    bsh = bs / 2
    cs = Vector{Float64}(undef,2)	  
    for ibox = laddr[1,curlev]:laddr[2,curlev]
	

	# If ibox is cut and flagged, see if colleagues should be flagged.
	# We do this by temporarily create the children and see if they are cut, and if so if they are fine neigbours. If so, then flag colleague.
	if (icut[2,ibox] == 1) & (iflag[ibox] == 1) & (nchild[ibox] == 0)
	    for j = 1:4
		cs[1] = centers[1,ibox] + xind[j] * bsh/2
		cs[2] = centers[2,ibox] + yind[j] * bsh/2
		xy = cs .+ grid * bs/2
		idxinj = checkinside(xy) #TODO: SAVE THESE
		ninj = length(idxinj)
		if (ninj != npbox) & (ninj != 0)
		    # Child is cut, check if any leaf coarse neighbours outside. No need to exclude ibox here, it is already flagged.
		    for i = 1:nnbors[ibox]	
			jbox = nbors[i,ibox]
			if (icut[2,jbox] == 2) & (nchild[jbox] == 0) # Is outside
			    xdis = cs[1] - centers[1,jbox]
			    ydis = cs[2] - centers[2,jbox]
			    ict = 0
			    if abs(xdis) <= distest
				ict = ict + 1
			    end
			    if abs(ydis) <= distest
				ict = ict + 1
			    end
			    if ict == 2
				iflag[jbox] = 1
			    end
			end
		    end
		end
	    end
	end

	if (icut[2,ibox] == 1) & (nchild[ibox] == 0) & (bs > bdrytol)
	    iflag[ibox] = 1
	end
        
	#Check if box is outside and leaf and has fine neighbors that are cut or flagged
	if (icut[2,ibox] == 2) & (nchild[ibox] == 0)
	    for i = 1:nnbors[ibox]
                #              Loop over colleagues of flag++ box        
		jbox = nbors[i,ibox]				

                #=             
                c              Loop over the children of the colleague box
                c              Note we do not need to exclude self from
                c              the list of colleagues as a self box which
                c              is flag++ does not have any children 
                c              and will not enter the next loop
                =#
		if (nchild[jbox] > 0)
		    for j = 1:4
			kbox = ichild[j,jbox]
			if (kbox > 0) & (icut[2,kbox] == 1) #& (iflag[jbox] == 1)# kbox > 0 redundant # WRONG TO CHECK IF jbox IS FLAGGED. JUST SUFFICIENT T
			    xdis = centers[1,kbox] - centers[1,ibox]
			    ydis = centers[2,kbox] - centers[2,ibox]
			    ict = 0
			    if abs(xdis) <= distest
				ict = ict + 1
			    end
			    if abs(ydis) <= distest
				ict = ict + 1
			    end
			    if ict == 2
				iflag[ibox] = 1
				@goto cont2
			    end
			end
		    end
		elseif (iflag[jbox] == 1) & (icut[1,jbox] == 1)
		    for k = 1:4
			cs[1] = centers[1,jbox] + xind[k] * bsh/2
			cs[2] = centers[2,jbox] + yind[k] * bsh/2
			
			xy = cs .+ grid * bs/2
			idxink = checkinside(xy) #TODO: Save these
			nink = length(idxink)
			if (nink != npbox) & (nink != 0)
			    # Child is cut, check if any leaf coarse neighbours outside. No need to exclude ibox here, it is already flagged.
			    
			    xdis = cs[1] - centers[1,ibox]
			    ydis = cs[2] - centers[2,ibox]
			    ict = 0
			    if abs(xdis) <= distest
				ict = ict + 1
			    end
			    if abs(ydis) <= distest
				ict = ict + 1
			    end
			    if ict == 2
				iflag[ibox] = 1
			    end
			end
		    end	 
		end
                #                 End of looping over the children of the child
                #                 of the colleague box
	    end
            #              End of looping over the children of the colleague box
	end
	@label cont2      
    end
    
    return
end #vol_updateflags			  


function mesh2d(x::Vector{Float64},nx::Int64,y::Vector{Float64},ny::Int64,xy::Matrix{Float64})
    ind = 0
    for iy = 1:ny
	for ix = 1:nx
            ind = ind+1
            xy[1,ind] = x[ix]
            xy[2,ind] = y[iy]
	end
    end
    return
end # mesh2d

function interppot_sort(norder::Int64,xtarg::Matrix{Float64},nchild::Vector{Int64},ichild::Matrix{Int64},potmat::Matrix{Float64},centers::Matrix{Float64},boxsize::OffsetVector{Float64, Vector{Float64}},iboxtarg::Vector{Int64})
    # Sort points
    boxlen = 1.0
    ntarg = length(iboxtarg)
    @inbounds	for ieval = 1:ntarg
	ibox = 1
	iboxnext = 1	
	while nchild[ibox] != 0
	    mindist = boxlen
	    kbox = 0
	    for j = 1:4
		kbox = ichild[j,ibox]
		kdist = sqrt((centers[1,kbox]-xtarg[1,ieval])^2 + (centers[2,kbox]-xtarg[2,ieval])^2)
		if kdist < mindist
		    mindist = kdist
		    iboxnext = kbox
		end
	    end
	    ibox = iboxnext
	end
	iboxtarg[ieval] = ibox
    end
end

function interppot_eval(norder::Int64,xtarg::Matrix{Float64},ilev::Vector{Int64},nchild::Vector{Int64},ichild::Matrix{Int64},potmat::Matrix{Float64},centers::Matrix{Float64},boxsize::OffsetVector{Float64, Vector{Float64}},iboxtarg::Vector{Int64},pottarg::Vector{Float64})
    # Sort points
    ntarg = length(iboxtarg)
    # Evaluate potential by interpolating over box
    Threads.@threads	for ieval = 1:ntarg
	ibox = iboxtarg[ieval]
	ilevel = ilev[ibox]
	bs = boxsize[ilevel]
	ixeval = xtarg[:,ieval]
	a1 = centers[1,ibox] - bs / 2
	b1 = centers[1,ibox] + bs / 2
	a2 = centers[2,ibox] - bs / 2
	b2 = centers[2,ibox] + bs / 2

	potbox = copy(transpose(reverse(reshape(potmat[:,ibox],(norder,norder)))))	
	# Compute Chebyshev coefficients with DCT
        #		potcoefs = interpspec2D_DCT_compcoeff(norder, norder, potbox)
	potcoefs = DCT_compcoeff_2d(norder, norder, potbox)	 
	pottarg[ieval] = eval_func_2D(potcoefs,ixeval[1],ixeval[2],norder,a1,b1,norder,a2,b2)
    end
end

function interppot_eval_grad(norder::Int64,xtarg::Matrix{Float64},ilev::Vector{Int64},nchild::Vector{Int64},ichild::Matrix{Int64},potmat::Matrix{Float64},centers::Matrix{Float64},boxsize::OffsetVector{Float64, Vector{Float64}},iboxtarg::Vector{Int64},pottarg::Matrix{Float64})
    # Sort points
    ntarg = length(iboxtarg)
    # Evaluate potential by interpolating over box
    Threads.@threads	for ieval = 1:ntarg
	ibox = iboxtarg[ieval]
	ilevel = ilev[ibox]
	bs = boxsize[ilevel]
	ixeval = xtarg[:,ieval]
	a1 = centers[1,ibox] - bs / 2
	b1 = centers[1,ibox] + bs / 2
	a2 = centers[2,ibox] - bs / 2
	b2 = centers[2,ibox] + bs / 2

	potbox = copy(transpose(reverse(reshape(potmat[:,ibox],(norder,norder)))))	
	# Compute Chebyshev coefficients with DCT
        #potcoefs = interpspec2D_DCT_compcoeff(norder, norder, potbox)
        potcoefs = DCT_compcoeff_2d(norder, norder, potbox)	 
        dcx = .-diffx_interpspec2D_DCT_compcoeff(norder, norder, potcoefs)
        dcy = .-diffy_interpspec2D_DCT_compcoeff(norder, norder, potcoefs)

	pottarg[ieval,1] = eval_func_2D(collect(dcx'),ixeval[1],ixeval[2],norder,a1,b1,norder,a2,b2)
        pottarg[ieval,2] = eval_func_2D(collect(dcy'),ixeval[1],ixeval[2],norder,a1,b1,norder,a2,b2)
    end
end


# Analytic extension
function fun_ext_analytic(nboxes::Int64,norder,fvals,fun::Function,centers,ilevel,boxsize)
    type = 'f'
    xref = Matrix{Float64}(undef,2,norder^2)
    wts = Vector{Float64}(undef,norder^2)
    umat = Matrix{Float64}(undef,norder,norder)	
    vmat = Matrix{Float64}(undef,norder,norder)
    itype = 0 # What type of calculation to be performed, see function chebexps
    chebtens_exps_2d(itype,norder,type,xref,umat,1,vmat,1,wts)
    
    chebgrid = xref / 2 # Scale to [-0.5,0.5]^2
    for ibox = 1:nboxes
	ilev = ilevel[ibox]
	igrid = centers[:,ibox] .+ chebgrid * boxsize[ilev]		  
	fvals[:,ibox] .= fun.(igrid[1,:],igrid[2,:])
    end
end



function fun_err(n::Int64,fcoefs::Matrix{Float64},rmask::Vector{Float64},iptype::Int64,rscale::Float64)
    #=			  
    c       this subroutine estimates the error based on the expansion
    c       coefficients in a given basis
    c       
    c       input
    c        nd - integer
    c          number of functions
    c        n -  integer
    c           number of points at which function is tabulated
    c        fcoefs: double precision(nd,n) 
    c          tensor product legendre coeffs of func
    c        rmask: double precision(n)
    c           coefficients which will be accounted for in the error
    c        iptype: integer
    c           type of error to be computed
    c           iptype = 0 - linf error
    c           iptype = 1 - l1 error
    c           iptype = 2 - l2 error
    c        rscale: double precision
    c          scaling factor
    c   alpha = 1
    beta = 0
    bs = boxsize / 4.0
    bs2 = 2 * bs
    rscale2 = bs2^eta
    #   note extra factor of 4 sincee wts2 are on [-1,1]^2 
    #   as opposed to [-1/2,1/2]^2
    rsc = boxlen^2/4

    rscale = rscale2
    c       output
    c         err: double precision
    c           max scaled error in the functions
    c
    implicit none
    integer n,i,iptype,idim,nd
    real *8 rscale,err
    real *8 fcoefs(nd,n),rmask(n)
    real *8, allocatable :: errtmp(:),ftmp(:,:)
    real *8 alpha, beta

    allocate(errtmp(nd),ftmp(nd,n))
    =#
    errtmp = 0.0
    ftmp = Vector{Float64}(undef,n)
#    alpha = 1.0
#    beta = 0.0
    err = 0.0
    if iptype == 0
	for i = 1:n
	    if rmask[i] > 0.5
		if errtmp < abs(fcoefs[i]) 
		    errtmp = abs(fcoefs[i])
		end
	    end
	end
    end

    if iptype == 1
	for i = 1:n 
	    ftmp[i] = abs(fcoefs[i])
        end
	errtmp = ftmp' * rmask
        #      dgemv('n',nd,n,alpha,ftmp,nd,rmask,1,beta,errtmp,1)
    end


    if iptype == 2
	for i = 1:n 
	    ftmp[i] = fcoefs[i]^2
        end
        #  call dgemv('n',nd,n,alpha,ftmp,nd,rmask,1,beta,errtmp,1)
	errtmp = ftmp' * rmask
	errtmp = sqrt(errtmp)
    end

    err = errtmp * rscale

    #err = 0
    #do idim=1,nd
    #if(errtmp(idim).gt.err) err = errtmp(idim)
    #enddo
    return err
end #fun_err

# Given a simply connected domain centered at [c01,c02], check if inside domain	

# Only works for convex and simply connected geometry.
function checkinside_param(x::Array{Float64},c01::Float64,c02::Float64,param::Function)
    C = hcat(ones(Float64,size(x,2))*c01,ones(Float64,size(x,2))*c02)
    xp = hcat(sqrt.((x[1,:] .- c01).^2 .+ (x[2,:] .- c02).^2), atan.(x[2,:] .- c02, x[1,:] .- c01)) # Polar coordinates
    idxin = findall(x -> x .<= 0.0, vec(xp[:,1] .- sqrt.(sum((param(xp[:,2]) .- C).^2,dims=2))))
    return idxin
end

function checkinside_ellipse(x::Array{Float64},c01::Float64,c02::Float64,a,b)
    idxin = findall(sum((x[1,:] .- c01).^2 ./ a.^2 .+ (x[2,:] .- c02).^2 ./ b.^2,dims=2) .<= 1.0)
    return idxin
end

function checkinside_param2(x::Array{Float64},checkparamouter::Function,checkparaminner::Function)
    idxin = setdiff(checkparamouter(x),checkparaminner(x))
    return idxin
end

function checkoutside_multconnected(x::Array{Float64},curve)
    idxin = collect(1:size(x,2)) 
    for icurve = 1:length(curve) 
	checkinside1temp(x) = checkinside_param(x,curve[icurve].center[1],curve[icurve].center[2],t -> hcat(real.(curve[icurve].tau.(-t)),imag.(curve[icurve].tau.(-t))))
	idxin = setdiff(idxin,checkinside1temp(x))
    end
    return idxin 
end

function checkcut(c,bs,paramv)

    xa = c[1] - bs/2
    xb = c[1] + bs/2
    ya = c[2] - bs/2
    yb = c[2] + bs/2

    itrmax = 100
    tol = 1e-12
    tt = range(0,2*pi,100)
    for i = 1:length(paramv)
	c01 = paramv[i].center[1] + eps()
	c02 = paramv[i].center[2] + eps()
	s = paramv[i].orientation
	param(t) = paramv[i].tau(s*t)
	dparam(t) = s*paramv[i].dtau(s*t)
	pnts = param.(tt)
	# Check left side of square 
	t = 1
	itr = 0

	t0 = atan((ya + yb)/2 - c02, xa - c01) # Polar coordinates, initial guess is center on the side
        #		t0 = tt[findmin(abs.(pnts .- xa .- 1im*(ya + yb)/2))[2]]
	while (abs(t) > tol) & (itr < itrmax)
	    t = real(param(t0) - xa)/real(dparam(t0))
	    t0 = t0 - t
	    itr = itr + 1
	end
	if itr < itrmax # Converged
	    y0 = imag(param(t0))
	    if (y0 >= ya) & (y0 <= yb)   
		return true
	    end
	end

	# Check right side of square 
	t = 1
	itr = 0
	t0 = atan((ya + yb)/2 - c02, xb -c01) # Polar coordinates, initial guess is center on the side
        #		t0 = tt[findmin(abs.(pnts .- xb .- 1im*(ya + yb)/2))[2]]
	while (abs(t) > tol) & (itr < itrmax)
	    t = real(param(t0) - xb)/real(dparam(t0))
	    t0 = t0 - t
	    itr = itr + 1
	end
	if itr < itrmax # Converged
	    y0 = imag(param(t0))
	    if (y0 >= ya) & (y0 <= yb)
		return true
	    end
	end

	# Check bottom side of square 
	t = 1
	itr = 0
	t0 = atan(ya-c02,(xa + xb)/2 - c01) # Polar coordinates, initial guess is center on the side
        #		t0 = tt[findmin(abs.(pnts .- (xa + xb)/2 .- 1im*ya))[2]]
	while (abs(t) > tol) & (itr < itrmax)
	    t = (imag(param(t0)) - ya)/imag(dparam(t0))
	    t0 = t0 - t
	    itr = itr + 1
	end
	if itr < itrmax # Converged
	    x0 = real(param(t0))
	    if (x0 >= xa) & (x0 <= xb)
		return true
	    end
	end
	
	# Check top side of square 
	t = 1
	itr = 0
	t0 = atan(yb - c02,(xa + xb)/2 - c01) # Polar coordinates, initial guess is center on the side
        #		t0 = tt[findmin(abs.(pnts .- (xa + xb)/2 .- 1im*yb))[2]] 
	while (abs(t) > tol) & (itr < itrmax)
	    t = (imag(param(t0)) - yb)/imag(dparam(t0))
	    t0 = t0 - t
	    itr = itr + 1
	end
	if itr < itrmax # Converged
	    x0 = real(param(t0))
	    if (x0 >= xa) & (x0 <= xb)
		return true
	    end
	end
    end
    # No intersections found 
    return false
end

# square
#=
function checkcut(c,bs,paramv)
xa = c[1] - bs/2
xb = c[1] + bs/2
ya	= c[2] - bs/2
yb = c[2] + bs/2


return !isempty(intersect(findall(xa .< xbdrycheck[1,:] .< xb), findall(ya .< xbdrycheck[2,:] .< yb)))
return true

end
=#
function approxcutarea(xbdry,a1,b1,a2,b2,checkinside)
    x = [a1,b1,b1,a1]
    y = [a2,a2,b2,b2]
    
    corners = vcat(copy(x'),copy(y')) 
    idxcornersin = checkinside(corners)
    idxincutbox = intersect(intersect(findall(xbdry[1,:] .> a1),findall(xbdry[1,:] .< b1)),intersect(findall(xbdry[2,:] .> a2),findall(xbdry[2,:] .< b2)))

    bdrycut = xbdry[:,idxincutbox]
    corner = bdrycut[:,end]
    bdrycut = bdrycut[:,2:end]
    corners = hcat(corners[:,idxcornersin])

    while !isempty(corners)
	val,idxnext = findmin(sqrt.((corners[1,:] .- corner[1]).^2 .+ (corners[2,:] .- corner[2]).^2))
	corner = corners[:,idxnext]
	bdrycut = hcat(bdrycut,corner)
	corners = corners[:,1:end .!= idxnext]
    end
    bdrycut = hcat(bdrycut,bdrycut[:,1]) 
    area = abs(1/2 * sum(bdrycut[1,1:end-1].*bdrycut[2,2:end] .- bdrycut[1,2:end].*bdrycut[2,1:end-1]))
    return area
end




function checkrestriction(centers,nlevels,nboxes,boxsize,nbmax,nlmax,itree,iptr,icut)

    bar = reshape(itree[iptr[1]:iptr[2]-1],(2,(nlevels + 1)))
    laddr = OffsetArray(Matrix{Int64}(undef,2,nlevels+1),1:2,0:nlevels)
    #TODO: bar = reshape(itree[iptr[1]:iptr[2]-1],(2,(nlevels + 1)))
    for i = 1:2
	for j = 0:nlevels
	    laddr[i,j] = bar[i,j+1]
	end
    end

    ilev = view(itree,iptr[2]:iptr[3]-1)
    iparent = itree[iptr[3]:iptr[4]-1]
    nchild = itree[iptr[4]:iptr[5]-1]
    ichild = reshape(itree[iptr[5]:iptr[6]-1],(4,nboxes))
    nnbors = itree[iptr[6]:iptr[7]-1]
    nbors = reshape(itree[iptr[7]:iptr[8]-1],(9,nboxes))


    
    iflag = Vector{Int64}(undef,nbmax)
    laddrtail = OffsetArray(Matrix{Int64}(undef,2,nlmax+1),1:2,0:nlmax)
    
    #allocate(iflag(nbmax))

    #     Initialize flag array
    #C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
    for i = 1:nboxes
	iflag[i] = 0
    end
    #C$OMP END PARALLEL DO     


    #=
    c     Flag boxes that violate level restriction by "1"
    c     Violation refers to any box that is directly touching
    c     a box that is more than one level finer
    c
    c     Method:
    c     1) Carry out upward pass. For each box B, look at
    c     the colleagues of B's grandparent
    c     2) See if any of those colleagues are childless and in
    c     contact with B.
    c
    c     Note that we only need to get up to level two, as
    c     we will not find a violation at level 0 and level 1
    c
    c     For such boxes, we set iflag(i) = 1
    c

    PUX: Also, we will mark boxes that are outside with a fine neighbor that is cut. These bokes violate the boundary level-restriction
    =#
    for ilev = nlevels:-1:2
        #        This is the distance to test if two boxes separated
        #        by two levels are touching
	distest = 1.05 * (boxsize[ilev-1] + boxsize[ilev-2]) / 2.0
	distest_cut = 1.05 * (boxsize[ilev] + boxsize[ilev-1]) / 2.0
        #C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox,idad,igranddad,i,jbox)         
        #C$OMP$ PRIVATE(ict,xdis,ydis,zdis)
	for ibox = laddr[1,ilev]:laddr[2,ilev] 
	    idad = iparent[ibox]
	    igranddad = iparent[idad]
            
            #           Loop over colleagues of granddad            
	    for i = 1:nnbors[igranddad]
		jbox = nbors[i,igranddad]
                #              Check if the colleague of grandad
                #		         is a leaf node. This automatically
                #              eliminates the granddad
		if (nchild[jbox] == 0) & (iflag[jbox] == 0)
		    xdis = centers[1,jbox] - centers[1,idad]
                    ydis = centers[2,jbox] - centers[2,idad]
                    ict = 0
                    if abs(xdis) <= distest
			ict = ict + 1
		    end
                    if abs(ydis) <= distest
			ict = ict + 1
		    end
                    if ict == 2
			iflag[jbox] = 1
                    end
		end
                #              End of checking criteria for the colleague of
                #              granddad
	    end
            #           End of looping over colleagues of
            #           granddad

            #        Check if boundary level restriction is violated.			 
            #        If ibox is cut, then check if colleagues of dad are coarse neigbors.	#        Then check if they are outside and leaf nodes. Such boxes violate the level-restriction on the boundary. 
	    if icut[2,ibox] == 1
                #				Loop over colleagues of parent
		for i = 1:nnbors[idad]
		    jbox = nbors[i,idad]
                    #           Check if the colleague of dad is a leaf node and not already
                    #   			flaged and outside
		    if (nchild[jbox] == 0) & (iflag[jbox] == 0) & (icut[2,jbox] == 2)
			xdis = centers[1,jbox] - centers[1,ibox]
                        ydis = centers[2,jbox] - centers[2,ibox]
                        ict = 0
                        if abs(xdis) <= distest_cut
			    ict = ict + 1
			end
                        if abs(ydis) <= distest_cut
			    ict = ict + 1
			end
                        if ict == 2
			    iflag[jbox] = 1
                        end
		    end
		    
		end
	    end
	    
	end
    end
    for ibox = 1:nboxes
	if iflag[ibox] == 1
	    println("Box ",ibox," violates lr")
	end
    end
    return iflag
end	 

function fun_reset_leafs(fvals,itree,iptr,nboxes,icut,insideidx,ninsideidx)
    Threads.@threads	for ibox = 1:nboxes
	if (itree[iptr[4] + ibox - 1] == 0) & (icut[2,ibox] == 1) # Leaf box
	    fvals[insideidx[ninsideidx[ibox]+1:end,ibox],ibox] .= 0.0
	end
    end
end

function fun_ext_all(fvals,fun,centers,boxsize,itree,ibox,iptr,grid)
    ilev = itree[iptr[2] + ibox - 1] 
    xeibox = centers[:,ibox] .+ grid * boxsize[ilev]
    fvals[:,ibox] .= fun.(xeibox[1,:],xeibox[2,:])
end

function setup_extension(norder::Int64,ilev::Int64,idad::Int64,nchild::Vector{Int64},ichild::Matrix{Int64},ncoll::Vector{Int64},coll::Matrix{Int64},centers::Matrix{Float64},boxsize::OffsetVector{Float64, Vector{Float64}},icut::Matrix{Int64},inext,iext,ibox::Int64,ipou::Vector{Int64})
	# Loop over colleagues and neighbors to find extension boxes.
	next = 0 # Number of extension boxes (including oneself)
	#Loop over colleagues and their children (and possibly childrens children)
	for icoll = 1:ncoll[ibox]
		jbox = coll[icoll,ibox]
		#	Is it cut or outside?
		if icut[2,jbox] != 0
			# Is it a leaf node?
			if (nchild[jbox] == 0)
				next = next + 1
				iext[next] = jbox
				ipou[jbox] = 1
			else # Check its children.
				for k = 1:4
					kbox = ichild[k,jbox]
					if icut[2,kbox] != 0 # Is not inside
						if (nchild[kbox] == 0) # Leaf node, add to extension list
							next = next + 1
							iext[next] = kbox
							ipou[kbox] = 1
						else # Not childless, check children of children
							for l = 1:4
								lbox = ichild[l,kbox]
								if icut[2,lbox] != 0
									if (nchild[lbox] == 0) # Leaf node, add to extension list
										next = next + 1
										iext[next] = lbox
										ipou[lbox] = 1
									else										
										for m = 1:4
											mbox = ichild[m,lbox]
												if icut[2,mbox] != 0
													if (nchild[mbox] == 0) # Leaf node, add to extension list
														next = next + 1
														iext[next] = mbox
														ipou[mbox] = 1
													else
														for p = 1:4
															pbox = ichild[p,mbox]
															if icut[2,pbox] != 0
																if (nchild[pbox] == 0) # Leaf node, add to extension list
																	next = next + 1
																	iext[next] = pbox
																	ipou[pbox] = 1
																else
																	println("ERROR, too many levels in extension, ibox = ", ibox, " pbox = ", pbox)
																end
															end
														end
													end
												end
											end
										end
									end
								end
							end
						end	
					end
				end
			end
		end
	
	# Loop over coarse neighbors, only cut-cells are possible here. 
#	idad = iparent[ibox]
	jlev = ilev - 1
	ncolldad = ncoll[idad]
	for icoll = 1:ncolldad
		jbox = coll[icoll,idad ]
		if (jbox != 0) & (nchild[jbox] == 0) & (icut[2,jbox] == 1)
			# Check if neighbor to box 		
			if (abs(centers[1,ibox] - centers[1,jbox]) <= 1.05*(boxsize[ilev] / 2 + boxsize[jlev] / 2)) &  (abs(centers[2,ibox] - centers[2,jbox]) <= 1.05*(boxsize[ilev] / 2 + boxsize[jlev] / 2))
				next = next + 1
				iext[next] = jbox
				ipou[jbox] = 1
			end
		end
	end
	inext[ibox] = next
end

function init_extdata(nboxes::Int64, extdata::Vector{Extdata})
    for ibox = 1:nboxes
	extdata[ibox] = Extdata([0],[0],zeros(Int64,1,1),zeros(Int64,1,1),zeros(Float64,1,1))
    end
end

######################### LEAST-SQUARES
function compextregion(norder::Int64,ilev::Vector{Int64},xk::Matrix{Float64},grid::Matrix{Float64},checkinside::Function,centers::Matrix{Float64},boxsize::OffsetVector{Float64, Vector{Float64}},icut::Matrix{Int64},inext::Vector{Int64},iext::AbstractVector{Int64},ibox::Int64,ixk,Binsideidx::AbstractVector{Int64},nBinsideidx::Vector{Int64},Bgrid::Matrix{Float64},rad::Function,xbdry::AbstractMatrix{Float64})

    bs = boxsize[ilev[ibox]] 
    next = inext[ibox]
    npbox = norder^2
    recxk = 3.0 # Recruitment zone
    rec = 3.0 # Extend one box
    xkibox = copy(hcat(recxk * xk[1,:] * bs .+ centers[1,ibox], recxk * xk[2,:]*bs  .+ centers[2,ibox])')
    idxinxk = checkinside(xkibox)
    ixk[idxinxk,ibox] .= true
    ninxk = length(idxinxk)
    temp = centers[:,ibox] .+ Bgrid * bs
    Bgrid = centers[:,ibox] .+ Bgrid * bs
    
    idxinB = checkinside(Bgrid)
    ninB = length(idxinB)
    Binsideidx[1:ninB] .= idxinB
    Binsideidx[ninB+1:end] = setdiff(collect(range(1,length(Bgrid[1,:]))),idxinB)
    nBinsideidx[ibox] = ninB

    c1 = centers[1,ibox]
    c2 = centers[2,ibox]
    a1 = c1 - bs * rec / 2
    b1 = c1 + bs * rec / 2
    a2 = c2 - bs * rec / 2
    b2 = c2 + bs * rec / 2
    areain = approxcutarea(xbdry,a1,b1,a2,b2,checkinside) 
    nsupp = 0

    csupp = zeros(Int64,2,next) # counter for indices
    idxsupp = zeros(Int64, next * npbox)
    
    for j = 1:next
	nsupp = nsupp + 1
	csupp[1,j] = nsupp
	jbox = iext[j]
	jlev = ilev[jbox]
	jgrid = centers[:,jbox] .+ grid * boxsize[jlev]
	idxinjgrid = checkinside(jgrid)
	idxinjgridsupp = findall(rad.(jgrid[1,:],jgrid[2,:],centers[1,jbox],centers[2,jbox],boxsize[jlev]) .<= 1)
 	idxinjgridsupp = setdiff(idxinjgridsupp,idxinjgrid)
	nsupp = nsupp + length(idxinjgridsupp) - 1
	csupp[2,j] = nsupp
	idxsupp[csupp[1,j]:csupp[2,j]] .= idxinjgridsupp
    end
    idxsupp = idxsupp[1:nsupp]
    return idxsupp, csupp, areain
end


function comp_extmat_ls(norder::Int64,norderrbf::Int64,ilev::Vector{Int64},idad::Int64,nchild::Vector{Int64},ichild::Matrix{Int64},ncoll::Vector{Int64},coll::Matrix{Int64},xk::Matrix{Float64},grid::Matrix{Float64},checkinside::Function,centers::Matrix{Float64},boxsize::OffsetVector{Float64, Vector{Float64}},icut::Matrix{Int64},inext::Vector{Int64},iext::AbstractVector{Int64},ibox::Int64,ipou::Vector{Int64},ixk,Binsideidx::AbstractVector{Int64},nBinsideidx::Vector{Int64},Bgrid::Matrix{Float64},rad::Function,xbdry::AbstractMatrix{Float64})

    setup_extension(norder,ilev[ibox],idad,nchild,ichild,ncoll,coll,centers,boxsize,icut,inext,iext,ibox,ipou)
    
    bs = boxsize[ilev[ibox]] 
    next = inext[ibox]
    npbox = norder^2
    recxk = 3.0 # Recruitment zone
    rec = 3.0 # Extend one box
    xkibox = copy(hcat(recxk * xk[1,:] * bs .+ centers[1,ibox], recxk * xk[2,:]*bs  .+ centers[2,ibox])')
    idxinxk = checkinside(xkibox)
    ixk[idxinxk,ibox] .= true
    ninxk = length(idxinxk)
    temp = 	centers[:,ibox] .+ Bgrid * bs
    Bgrid = centers[:,ibox] .+ Bgrid * bs

    idxinB = checkinside(Bgrid)
    ninB = length(idxinB)
    Binsideidx[1:ninB] .= idxinB
    Binsideidx[ninB+1:end] = setdiff(collect(range(1,norderrbf^2)),idxinB)
    nBinsideidx[ibox] = ninB

    c1 = centers[1,ibox]
    c2 = centers[2,ibox]
    a1 = c1 - bs * rec / 2
    b1 = c1 + bs * rec / 2
    a2 = c2 - bs * rec / 2
    b2 = c2 + bs * rec / 2

    nsupp = 0
    csupp = zeros(Int64,2,next) # counter for indices

    idxsupp = zeros(Int64, next * npbox)
    
    for j = 1:next
	nsupp = nsupp + 1
	csupp[1,j] = nsupp
	jbox = iext[j]
	jlev = ilev[jbox]
	jgrid = centers[:,jbox] .+ grid * boxsize[jlev]
	idxinjgrid = checkinside(jgrid)
	idxinjgridsupp = findall(rad.(jgrid[1,:],jgrid[2,:],centers[1,jbox],centers[2,jbox],boxsize[jlev]) .<= 1)
 	idxinjgridsupp = setdiff(idxinjgridsupp,idxinjgrid)
	nsupp = nsupp + length(idxinjgridsupp) - 1
	csupp[2,j] = nsupp
	idxsupp[csupp[1,j]:csupp[2,j]] .= idxinjgridsupp
    end
    idxsupp = idxsupp[1:nsupp]

    return idxsupp, csupp, areain
end

function precompute_extdata_ls(norder::Int64,norderrbf::Int64,norderpou::Int64,ilev::Vector{Int64},iparent::Vector{Int64},nchild::Vector{Int64},ichild::Matrix{Int64},ncoll::Vector{Int64},coll::Matrix{Int64},fvals::Matrix{Float64},fun::Function,xk::Matrix{Float64},grid::Matrix{Float64},checkinside::Function,centers::Matrix{Float64},boxsize::OffsetVector{Float64, Vector{Float64}},icut::Matrix{Int64},inext::Vector{Int64},iext::Matrix{Int64},wvals::Matrix{Float64},nboxes::Int64,ipou::Vector{Int64},extdata::Vector{Extdata},ixk::Matrix{Bool},Binsideidx::Matrix{Int64},nBinsideidx::Vector{Int64},xbdry::Matrix{Float64},areainv::Vector{Float64};allData=true)

    Bgrid = Matrix{Float64}(undef,2,norderrbf^2)
    Bxq = Vector{Float64}(undef,norderrbf)
    Bwts = Vector{Float64}(undef,norderrbf)
    Bumat = Matrix{Float64}(undef,norderrbf,norderrbf)	
    Bvmat = Matrix{Float64}(undef,norderrbf,norderrbf)
    Bitype = 2 # What type of calculation to be performed, see function chebexps
    chebexps(Bitype,norderrbf,Bxq,Bumat,Bvmat,Bwts)
    Bxq /= 2
    mesh2d(Bxq,norderrbf,Bxq,norderrbf,Bgrid)
    recxk = 3.0 # Recruitment zone
    rec = 3.0 # Extend one box
    
    Bgrid = [0.0,0.0] .+ Bgrid * boxsize[0] * rec
    xkibox = copy(hcat(recxk * xk[1,:] * boxsize[0] .+ 0.0, recxk * xk[2,:]*boxsize[0]  .+ 0.0)')
    rad(x,y,c1,c2,bs) = max(abs(x - c1), abs(y - c2)) / (rec * bs)

    if allData 
	Bgrid0 = Matrix{Float64}(undef,2,norder^2)
	Bxq = Vector{Float64}(undef,norder)
	Bwts = Vector{Float64}(undef,norder)
	Bumat = Matrix{Float64}(undef,norder,norder)	
	Bvmat = Matrix{Float64}(undef,norder,norder)
	Bitype = 2 # What type of calculation to be performed, see function chebexps
	chebexps(Bitype,norder,Bxq,Bumat,Bvmat,Bwts)
	Bxq /= 2
	mesh2d(Bxq,norder,Bxq,norder,Bgrid0)
	Bgridcomp = hcat(Bgrid,Bgrid0)
    else
	Bgridcomp = Bgrid 
    end
    Arbfqr =  rbfqr_diffmat_2d_fast(copy(Bgridcomp'), [0.0,0.0], recxk*boxsize[0]/sqrt(2), copy(xkibox'), 1e-5)[1] 
    nxk = size(xk,2)
    II = ones(Float64,nxk)
    II = Diagonal(II)
    Arbfqr = vcat(Arbfqr,II)
    ilocalext = intersect(findall(icut[2,:] .== 1), findall(nchild .== 0))

    dctplandim1 = plan_dct(rand(norderrbf,norderrbf),[1]; flags=FFTW.ESTIMATE, timelimit=Inf)
    dctplandim2 = plan_dct(rand(norderrbf,norderrbf),[2]; flags=FFTW.ESTIMATE, timelimit=Inf)

    if norderpou >= 0
        @inbounds Threads.@threads	 for ibox in ilocalext
	    extdata[ibox].idxsupp, extdata[ibox].csupp, areainv[ibox] = comp_extmat_ls(norder,norderrbf,ilev,iparent[ibox],nchild,ichild,ncoll,coll,xk,grid,checkinside,centers,boxsize,icut,inext,view(iext,:,ibox),ibox,ipou,ixk,view(Binsideidx,:,ibox),nBinsideidx,Bgrid,rad,view(xbdry,:,:))

            #		 Arbfqr =  rbfqr_diffmat_2d_fast(copy(Bgrid[:,Binsideidx[nBinsideidx[ibox]+1:end,ibox]]'), [0.0,0.0], recxk*boxsize[0]/sqrt(2), copy(xkibox[:,findall(view(ixk,:,ibox))]'), 1e-5)[1]
	    extdata[ibox].A = Arbfqr

	end
    else
	ipoudist = ones(Float64,size(ipou)) * 2 * boxsize[0]
        #		@inbounds Threads.@threads	 for ibox in ilocalext
        @inbounds @simd for ibox in ilocalext
	    fillipou(norder,ilev[ibox],iparent[ibox],nchild,ichild,ncoll,coll,centers,boxsize,icut,ibox,ipou,ipoudist)
	end
        #			@inbounds Threads.@threads  for ibox in findall(ipou .> 0)
        @inbounds @simd for ibox in findall(ipou .> 0)
	    inext[ipou[ibox]] += 1
	    iext[inext[ipou[ibox]],ipou[ibox]] = ibox
	end
	@inbounds @simd for ibox in ilocalext
            #                 @inbounds Threads.@threads	 for ibox in ilocalext
	    extdata[ibox].idxsupp, extdata[ibox].csupp, areainv[ibox] = compextregion(norder,ilev,xk,grid,checkinside,centers,boxsize,icut,inext,view(iext,:,ibox),ibox,ixk,view(Binsideidx,:,ibox),nBinsideidx,Bgridcomp,rad,view(xbdry,:,:))

            #		 Arbfqr =  rbfqr_diffmat_2d_fast(copy(Bgrid[:,Binsideidx[nBinsideidx[ibox]+1:end,ibox]]'), [0.0,0.0], recxk*boxsize[0]/sqrt(2), copy(xkibox[:,findall(view(ixk,:,ibox))]'), 1e-5)[1]
	    extdata[ibox].A = Arbfqr

	end  
	
    end
    return dctplandim1, dctplandim2, ilocalext, Arbfqr
end

function globalext_ls(norder,norderrbf,nboxes,fvals,itree,iptr,frhs,xk,grid,checkinside,centers,boxsize,icut,extdata,inext,iext,ipou,ixk,insideidx,ninsideidx,Binsideidx,nBinsideidx,dctplandim1,dctplandim2,wvals,irefine,estintlp,iptype,tol,norderpou,ilocalext,areainv;allData=false)
    npbox = norder^2

    Bgrid = Matrix{Float64}(undef,2,norderrbf^2)
    Bxq = Vector{Float64}(undef,norderrbf)
    Bwts = Vector{Float64}(undef,norderrbf)
    Bumat = Matrix{Float64}(undef,norderrbf,norderrbf)	
    Bvmat = Matrix{Float64}(undef,norderrbf,norderrbf)
    Bitype = 2 # What type of calculation to be performed, see function chebexps
    chebexps(Bitype,norderrbf,Bxq,Bumat,Bvmat,Bwts)
    Bxq /= 2
    mesh2d(Bxq,norderrbf,Bxq,norderrbf,Bgrid)

    rec = 3.0
    recx = 3.0
    
    
    @inbounds @simd for ibox in ilocalext
    	localext_ls(norder,norderrbf,view(itree,iptr[2]:iptr[3]-1),itree[iptr[4] + ibox - 1],view(fvals,:,iext[1:inext[ibox],ibox]),frhs,xk,grid,Bgrid,checkinside,centers,boxsize,inext[ibox],view(iext,:,ibox),wvals,ibox,view(extdata,ibox)[1],findall(view(ixk,:,ibox)),view(insideidx,:,ibox),ninsideidx[ibox],view(Binsideidx,:,ibox),nBinsideidx[ibox],deepcopy(dctplandim1),deepcopy(dctplandim2),irefine,estintlp,iptype,tol,areainv[ibox],allData)
    end

end # globalext	

function localext_ls(norder::Int64,norderrbf::Int64,ilevel::AbstractVector{Int64},nchild::Int64,fvals::AbstractMatrix{Float64},fun::Function,xk::Matrix{Float64},grid::Matrix{Float64},Bgrid::Matrix{Float64},checkinside::Function,centers::Matrix{Float64},boxsize::OffsetVector{Float64, Vector{Float64}},next::Int64,iext::AbstractVector{Int64},wvals::Matrix{Float64},ibox::Int64,extdata::Extdata,idxinxk::AbstractVector{Int64},insideidx::AbstractVector{Int64},ninsideidx::Int64,Binsideidx::AbstractVector{Int64},nBinsideidx::Int64,dctplandim1::FFTW.DCTPlan{Float64, 5, false},dctplandim2::FFTW.DCTPlan{Float64, 5, false},irefine::Vector{Bool},estintlp::Float64,iptype::Int64,tol::Float64,areain::Float64,allData::Bool)

    npboxrbf = norderrbf * norderrbf
    npbox = norder^2
    recxk = 3.0 # Recruitment zone
    rec = 3.0 # Extend one box
    ilev = ilevel[ibox]

    xkinibox = copy(hcat(recxk * xk[1,idxinxk] * boxsize[ilev] .+ centers[1,ibox], recxk * xk[2,idxinxk]*boxsize[ilev]  .+ centers[2,ibox])')

    idxsupp = extdata.idxsupp
    csupp = extdata.csupp

    Bgrid = centers[:,ibox] .+ Bgrid * boxsize[ilev] * rec

    Bgridcomp = Bgrid
    npcomp = npboxrbf 

    idxinB = Binsideidx[1:nBinsideidx]
    idxoutB = Binsideidx[nBinsideidx+1:end]

    fBcheb = zeros(Float64,norderrbf^2)
    C = Vector{Float64}(undef,size(xk,2))
    fBcheb2 = zeros(Float64,npcomp)	 
    @fastmath fBcheb2[idxoutB] .= Octavian.matmul(extdata.A[idxoutB,:],ldiv!(C,factorize(Array(view(extdata.A,vcat(idxinB,idxinxk .+ npcomp),:))), vcat(fun.(view(Bgridcomp,1,idxinB),view(Bgridcomp,2,idxinB)),fun.(xkinibox[1,:],xkinibox[2,:]))))

    fBcheb[:] .= fBcheb2[1:norderrbf^2]
    idxinBtemp = idxinB[findall(idxinB .<= norderrbf^2)]
    fBcheb[idxinBtemp] .= fun.(Bgrid[1,idxinBtemp],Bgrid[2,idxinBtemp])

    @fastmath chebcoefs = (dctplandim2 * copy(transpose(reverse(reshape(fBcheb,(norderrbf,norderrbf)))))) .* sqrt(2.0/norderrbf) # transpose and reverse to get format compatible with dct.
    chebcoefs[:,1] .= chebcoefs[:,1] ./ sqrt(2.0)
    @fastmath chebcoefs = (dctplandim1 * chebcoefs) .* sqrt(2.0/norderrbf) 
    chebcoefs[1,:] .= chebcoefs[1,:] ./ sqrt(2.0)


    c1 = centers[1,ibox]
    c2 = centers[2,ibox]
    bs = boxsize[ilev]
    a1 = c1 - bs * rec / 2
    b1 = c1 + bs * rec / 2
    a2 = c2 - bs * rec / 2
    b2 = c2 + bs * rec / 2
    idxinjgridsupp = 1:norder^2    
    @inbounds @simd for j = 1:next
	jbox = iext[j]
	jlev = ilevel[jbox]
	jgrid = centers[:,jbox] .+ grid * boxsize[jlev]
	@fastmath xpoly = reduce(hcat,chebypoly.(jgrid[1,1:norder],a1,b1,norderrbf))
	@fastmath ypoly = reduce(hcat,chebypoly.(jgrid[2,1:norder:npbox],a2,b2,norderrbf))
        @fastmath temp = Octavian.matmul(Octavian.matmul(ypoly',chebcoefs), xpoly)
        @inbounds @simd for k =1:norder^2   
            fvals[idxinjgridsupp[k],j] = temp[ceil(Int64,idxinjgridsupp[k]/norder),  mod(idxinjgridsupp[k]-1,norder) + 1]
        end
    end

end

function fillipou(norder::Int64,ilev::Int64,idad::Int64,nchild::Vector{Int64},ichild::Matrix{Int64},ncoll::Vector{Int64},coll::Matrix{Int64},centers::Matrix{Float64},boxsize::OffsetVector{Float64, Vector{Float64}},icut::Matrix{Int64},ibox::Int64,ipou::Vector{Int64},ipoudist::Vector{Float64})

	c1i = centers[1,ibox]
	c2i = centers[2,ibox]
	 
	for icoll = 1:ncoll[ibox]
		jbox = coll[icoll,ibox]
		#	Is it cut or outside?
		if icut[2,jbox] != 0
			# Is it a leaf node?
			if (nchild[jbox] == 0)
				dist = sqrt((c1i-centers[1,jbox])^2 + (c2i-centers[2,jbox])^2)
				if dist < ipoudist[jbox]
					ipou[jbox] = ibox
					ipoudist[jbox] = dist
				end
			else # Check its children.
				for k = 1:4
					kbox = ichild[k,jbox]
					if icut[2,kbox] != 0 # Is not inside
						if (nchild[kbox] == 0) # Leaf node, add to extension list
							dist = sqrt((c1i-centers[1,kbox])^2 + (c2i-centers[2,kbox])^2) < ipoudist[kbox] 
							if dist < ipoudist[kbox]
								ipou[kbox] = ibox
								ipoudist[kbox] = dist 
							end
						else # Not childless, check children of children
							for l = 1:4
								lbox = ichild[l,kbox]
								if icut[2,lbox] != 0
									if (nchild[lbox] == 0) # Leaf node, add to extension list
										dist = sqrt((c1i-centers[1,lbox])^2 + (c2i-centers[2,lbox])^2) 
										if dist < ipoudist[lbox]
											ipou[lbox] = ibox
											ipoudist[lbox] = dist 
										end
									else										
										for m = 1:4
											mbox = ichild[m,lbox]
												if icut[2,mbox] != 0
													if (nchild[mbox] == 0) # Leaf node, add to extension list
														dist = sqrt((c1i-centers[1,mbox])^2 + (c2i-centers[2,mbox])^2) 
														if dist < ipoudist[mbox]
															ipou[mbox] = ibox
															ipoudist[mbox] = dist
														end
													else
														for p = 1:4
															pbox = ichild[p,mbox]
															if icut[2,pbox] != 0
																if (nchild[pbox] == 0) # Leaf node, add to extension list
																	dist = sqrt((c1i-centers[1,pbox])^2 + (c2i-centers[2,pbox])^2) 
																	if  dist < ipoudist[pbox]
																		ipou[pbox] = ibox
																		ipoudist[pbox] = pbox 
																	end
																else
																	println("ERROR, too many levels in extension, ibox = ", ibox, " pbox = ", pbox)
																end
															end
														end
													end
												end
											end
										end
									end
								end
							end
						end	
					end
				end
			end
		end
	
end



function computeintbox(norder::Int64,nboxes::Int64,fvals::Matrix{Float64},iptype::Int64,nchild::AbstractVector,ilevel::AbstractVector,boxsize::OffsetVector{Float64, Vector{Float64}})

	npbox = norder^2

	grid = Matrix{Float64}(undef,2,npbox)
	 
	xq = Vector{Float64}(undef,norder)
	wts = Vector{Float64}(undef,norder)
	umat = Matrix{Float64}(undef,norder,norder)	
	vmat = Matrix{Float64}(undef,norder,norder)
	itype = 2 # What type of calculation to be performed, see function chebexps
	chebexps(itype,norder,xq,umat,vmat,wts)
	xq /= 2

	mesh2d(xq,norder,xq,norder,grid)

	npols = norder^2
	umat2 = Matrix{Float64}(undef,norder,norder)
	wts2 = Vector{Float64}(undef,npbox)
	xref2 = Matrix{Float64}(undef,2,npbox)
	itype = 1

	chebtens_exps_2d(itype,norder,'f',xref2,umat2,npols,vmat,1,wts2)
	 
	rint = 0.0
	for ibox = 1:nboxes
		if nchild[ibox] == 0 # Is leaf, add contribution
			for j = 1:npbox
				rint = rint + fvals[j,ibox] * wts2[j] * boxsize[ilevel[ibox]]^2/4
			end
		end
	end
	
return rint
end

# Matlab compability 
function ndgrid(x, y)
	nx = length(x)
	ny = length(y)
	vartype = typeof(x[1])
	@assert typeof(y[1])==vartype
	X = Array{vartype}(undef, nx, ny)
	Y = Array{vartype}(undef, nx, ny)
	for i=1:nx
		for j=1:ny
			X[i,j] = x[i]
			Y[i,j] = y[j]
    end
	end
	return X,Y
end
