__precompile__()
module CurveDiscretization

using FastGaussQuadrature
using FMMLIB2D
using LinearAlgebra
using GSL
using Printf

export DiscreteCurve
export discretize
export multiply_connected
export traparclen
export glpanels
export minmax_panel
export interior_points
export nearest_nb_R
export ndgrid

# Curve discretization
struct DiscreteCurve
    panelorder::Int64
    numpanels::Int64
    numpoints::Int64
    edges::Array{Float64, 2}
    n_edges::Array{Float64, 2}        
    t_edges::Array{Float64, 2}
    points::Array{Float64, 2}
    normals::Array{Float64, 2}
    derivatives::Array{Float64,2}
    weights::Array{Float64}
    dS::Array{Float64}
    curvature::Array{Float64}
    prevpanel::Array{Int64}
    nextpanel::Array{Int64}
    curvenum::Array{Int64}
end


include("legendre.jl")
include("mappings.jl")


"""
Discretize curve using composite Gauss-Legendre quadrature
"""
function discretize(curve, numpanels, panelorder;
                    equal_arclength = true)

    splitcomplex(z) = [real(z), imag(z)] # from C to R^2
    
    if isa(curve, Array)
        # TODO: Would be nicer to use multiple dispatch
        curves = curve
        Ncurves = length(curves)
        @assert length(numpanels) == Ncurves
        dcurves = Array{DiscreteCurve}(undef,Ncurves)
        for i = 1 : Ncurves
            dcurves[i] = discretize(curves[i], numpanels[i], panelorder, equal_arclength = equal_arclength)
        end
        return multiply_connected(dcurves)
    end
    
    start = 0.0
    stop = 2*pi
    # Setup division in parameter
    if equal_arclength
        tsplit, _, _ = traparclen(curve, numpanels)
        push!(tsplit, 2*pi)
    else
        # Equal parameter length
        tsplit = range(start, stop, length = numpanels+1);
    end
    # Prepare structures
    numpoints = numpanels * panelorder
    glpoints, glweights = gausslegendre(panelorder)
    points = zeros(2, numpoints)
    normals = zeros(2, numpoints)
    derivatives = zeros(2, numpoints)
    derivatives2 = zeros(2, numpoints)
    weights = zeros(numpoints)
    dS = zeros(numpoints)
    curvature = zeros(numpoints)
    t_edges = zeros(2, numpanels)
    edges = zeros(4, numpanels) # start and end
    n_edges = zeros(4, numpanels)    
    L = legendre_matrix(panelorder)
    resolution = 0.0
    # Populate
    for i in 1:numpanels
        # setup quadrature for t in [a,b]
        a, b = tsplit[i:i+1]
        t_edges[:, i] = [a, b]
        edges[1:2, i] = splitcomplex(curve.tau(a))
        edges[3:4, i] = splitcomplex(curve.tau(b))
        n_edges[1:2, i] = splitcomplex(curve.normal(a))
        n_edges[3:4, i] = splitcomplex(curve.normal(b))                        
        dt = b-a
        for j=1:panelorder
            idx = j + panelorder*(i-1)
            t = a + (1+glpoints[j])*dt/2
            w = glweights[j]*dt/2
            z = curve.tau(t)
            zp = curve.dtau(t)
            zpp = curve.d2tau(t)
            n = curve.normal(t)
            points[:, idx] = splitcomplex(z)
            derivatives[:, idx] = splitcomplex(zp)
            derivatives2[:, idx] = splitcomplex(zpp)
            normals[:, idx] = splitcomplex(n)
            weights[idx] = w
            dS[idx] = w*abs(zp)
            curvature[idx] = -imag(conj(zp)*zpp/abs(zp)^3)
        end
        # Get resolution estimate
        idx = (1:panelorder) .+ panelorder*(i-1)
        coeff = L * (dS[idx]./weights[idx])
        thisresolution = abs(coeff[end]) + abs(coeff[end-1])
        resolution = max(resolution, thisresolution)
    end

    prevpanel = circshift(1:numpanels, 1)
    nextpanel = circshift(1:numpanels, -1)
    curvenum = ones(Int64, numpoints)
    
    @info @sprintf("  Grid resolution: %.2e\n", resolution) 

    DiscreteCurve(
        panelorder,
        numpanels,
        numpoints,
        edges,
        n_edges,
        t_edges,
        points,
        normals,
        derivatives,
        weights,
        dS,
        curvature,
        prevpanel,
        nextpanel,
        curvenum
    )
end

function multiply_connected(grids)
    # Init
    panelorder = grids[1].panelorder
    numpanels = 0
    numpoints = 0
    points = zeros(2, 0)
    normals = zeros(2, 0)
    derivatives = zeros(2, 0)
    derivatives2 = zeros(2, 0)
    weights = zeros(0)
    dS = zeros(0)
    curvature = zeros(0)
    edges = zeros(4, 0)
    n_edges = zeros(4, 0)    
    t_edges = zeros(2, 0)    
    prevpanel = Int64[]
    nextpanel = Int64[]
    curvenum = Int64[]
    for i = 1 : length(grids)
        @assert grids[i].panelorder == panelorder
        # Concatenate
        edges = hcat(edges, grids[i].edges)
        n_edges = hcat(n_edges, grids[i].n_edges)
        t_edges = hcat(t_edges, grids[i].t_edges)
        points = hcat(points, grids[i].points)
        normals = hcat(normals, grids[i].normals)
        derivatives = hcat(derivatives, grids[i].derivatives)
        weights = vcat(weights, grids[i].weights)
        dS = vcat(dS, grids[i].dS)
        curvature = vcat(curvature, grids[i].curvature)
        prevpanel = vcat(prevpanel, grids[i].prevpanel .+ numpanels)
        nextpanel = vcat(nextpanel, grids[i].nextpanel .+ numpanels)
        append!(curvenum, i*ones(Int64, grids[i].numpoints))
        # Update count
        numpanels += grids[i].numpanels
        numpoints += grids[i].numpoints
    end
    # Construct output
    return DiscreteCurve(
        panelorder,
        numpanels,
        numpoints,
        edges,
        n_edges,
        t_edges,
        points,
        normals,
        derivatives,
        weights,
        dS,
        curvature,
        prevpanel,
        nextpanel,
        curvenum
    )
end

function glpanels(a, b, numpanels, pedges, order)
    # Discretize interval using composite Gauss-Legendre quadrature
    # 
    # Original code by Rikard Ojala
    T, W = gausslegendre(order)
    t = zeros(order*numpanels)
    w = zeros(order*numpanels)
    ptr = 1
    for j = 1:numpanels
        t[ptr:ptr+order-1] = (pedges[j+1]-pedges[j])*T/2 .+ (pedges[j]+pedges[j+1])/2;
        wj = W * (pedges[j+1]-pedges[j])/2
        w[ptr:ptr+order-1] = wj
        ptr = ptr + order
    end
    return t, w
end

function traparclen(curve, N, NL=1000)
    # t,z,W,L = traparclen(f, N, NL=1000)
    #
    # Discretizes the curve given by f(t) t = [0,2*pi] using the trapezoidal rule
    # equidistant in arc-length .
    # N is the number of quadrature points requested and if specified,
    # NL is the number of 16-point Gauss-Legendre panels used to compute the 
    # length of the curve. Default is NL = 1000.
    # Non-adaptive. Make sure that the curve is well resolved by N points.
    #
    # Original code by Rikard Ojala
    if N*2 > NL
        NL = 2 * N
        @info "Using $NL Gauss-Legendre points in traparclen"
    end
    order = 16
    pedges = range(0, 2*pi, length = NL+1)
    T, W = CurveDiscretization.glpanels(0, 2*pi, NL, pedges, order)
    zp = curve.dtau.(T)
    L = W' * abs.(zp)    
    L2 = range(1/N, 1-1/N, length = N-1) * L
    #Initial guess
    t = range(0, 2*pi, length = N + 1)
    t = t[2:end-1]
    dt = 1
    iter = 0
    maxiter = 30
    while norm(dt)/norm(t) > 1e-13 && iter < maxiter
        # Set up n-point GL quadrature on each segment.
        T, W = CurveDiscretization.glpanels(0, 2*pi, N-1, [0;t], order)
        zpt = curve.dtau.(t)
        zpT = curve.dtau.(T)
        #Compute the cumulative sum of all the segment lengths
        F = cumsum(sum(reshape(W .* abs.(zpT), order, N-1), dims=1), dims=2)'
        dt = (F-L2) ./ abs.(zpt)
        # Sort the parameters just in case.
        t = vec(t .- dt)
        sort!(t)
	iter += 1
    end
    if iter == maxiter
        @warn "Newton for equal arclength did not converge, bad discretization"
    end
    t = [0;t]
    z = curve.tau.(t)
    W = 2 * pi/N * ones(N,1)
    return t, z, W, L
end

function minmax_panel(grid::DiscreteCurve)
    hmin, hmax = Inf, 0.0
    for i = 1:grid.numpanels
        idx = (i-1)*grid.panelorder .+ (1:grid.panelorder)
        h = sum(grid.dS[idx])
        hmin = min(hmin, h)
        hmax = max(hmax, h)
    end
    return hmin, hmax
end


ndgrid(v::AbstractVector) = copy(v)

function ndgrid(v1::AbstractVector{T}, v2::AbstractVector{T}) where T
    m, n = length(v1), length(v2)
    v1 = reshape(v1, m, 1)
    v2 = reshape(v2, 1, n)
    return (repeat(v1, 1, n), repeat(v2, m, 1))
end		

function interior_points(grid::DiscreteCurve, zt::Vector{Float64})
    interior,interior_near = interior_points(grid::DiscreteCurve, zt::Matrix{Float64})
    return interior,interior_near
end	 

function interior_points(grid::DiscreteCurve, zt::Matrix{Float64}; exterior = false)
    if exterior
	d = -1.0
    else
	d = 1.0
    end
    # First pass: Laplace DLP identity works to O(1) very close to bdry
    ndS = grid.normals .* grid.dS'
    # Laplace DLP
    density = ones(grid.numpoints)
    U = rfmm2d(source=grid.points, target=zt, dipstr=density, dipvec=ndS,
               ifpot=false, ifpottarg=true)
    
    marker = 1/(2*pi)*U.pottarg    
    interior = marker .> 0.5*d

    # Second pass: find points close to boundary and distrust them
    _, hmax = minmax_panel(grid)
    dx = hmax*3/grid.panelorder # Estimate of largest point gap on bdry
    R = dx
    nnb, _ = nearest_nb_R(grid.points, zt, R)
    # listed points are within R of a boundary point
    # Third pass: check sign of projection against near normal
    veryclose = zeros(length(interior))
    N = size(zt, 2)
    for i=1:N
        if nnb[i] != 0
            rvec = zt[:,i] .- grid.points[:,nnb[i]]
            n = grid.normals[:,nnb[i]]
            rdotn = rvec[1]*n[1] + rvec[2]*n[2]
            interior[i] = rdotn < 0
            veryclose[i] = abs(rdotn) < R/10
        end
    end
    # Fourth pass: Find projection using parametrization
    coeffs = map_panels(grid)
    for i = 1:N
        if veryclose[i] != 0
            # Find corresponding panel
            near_panel = Int64(ceil(nnb[i]/grid.panelorder))
            z = zt[1,i] + 1im*zt[2,i]
            tloc, _ = invert_map(grid, coeffs, near_panel, z)
            interior[i] = imag(tloc) > 0
        end
    end
    interior_near = abs.(marker[interior] .- (1+d)/2) .> 1e-14

    return interior,interior_near
end

function nearest_nb_R(points::Matrix{Float64}, zt::Matrix{Float64}, R::Float64)
    # Bin sort of boundary points
    delta = 1e-8
    xmax = max(maximum(zt[1,:]), maximum(points[1,:])) + delta
    xmin = min(minimum(zt[1,:]), minimum(points[1,:])) - delta
    ymax = max(maximum(zt[2,:]), maximum(points[2,:])) + delta
    ymin = min(minimum(zt[2,:]), minimum(points[2,:])) - delta
    Lx = xmax-xmin
    Ly = ymax-ymin
    Nx = Int64(floor(Lx/R))
    Ny = Int64(floor(Ly/R))
    hx = Lx/Nx
    hy = Ly/Ny
    Nx += 1
    Ny += 1
    bin_count, bin_lists = binsort_points(points, xmin, ymin, hx, hy, Nx, Ny)
    # For each target points, find neareast neighbor
    R2 = R^2
    N = size(zt, 2)        
    nearest_nb = zeros(Int64, N)
    dist = zeros(N)
    for trg_idx=1:N
        z = zt[:,trg_idx]        
        ih, jh = bin_idx(z, xmin, ymin, hx, hy) # home bin
        r2min = Inf
        imin = 0
        # Iterate over nb bins
        for inb = ih.+(-1:1)
            for jnb = jh.+(-1:1)
                if inb>0 && jnb>0 && inb<Nx+1 && jnb<Ny+1
                    # Compare to target points in bin
                    list = bin_lists[inb, jnb]
                    for k=1:length(list)
                        i = list[k]
                        r2 = (points[1,i]-z[1])^2 .+ (points[2,i]-z[2])^2
                        if r2<r2min && r2<R2
                            imin = i
                            r2min = r2
                        end
                    end
                end
            end
        end
        nearest_nb[trg_idx] = imin
        dist[trg_idx] = sqrt(r2min)
    end # nearest nb list built
    return nearest_nb, dist
end

function binsort_points(zt::Matrix{Float64}, xmin, ymin, hx, hy, Nx, Ny)    
    N = size(zt, 2)
    # 1. Count points in each bin
    bin_count = zeros(Int64, Nx, Ny)
    for i=1:N
        ix, iy = bin_idx(zt[:,i], xmin, ymin, hx, hy)
        bin_count[ix, iy] += 1
    end
    # 2. Setup bin lists
    fill_ptr = ones(Int64, Nx, Ny)
    bin_lists = Array{Array{Int64}}(undef,Nx, Ny)
    for i=1:Nx
        for j=1:Ny
            bin_lists[i,j] = zeros(Int64, bin_count[i,j])
        end
    end
    # 3. Fill bins
    for i=1:N
        ix, iy = bin_idx(zt[:,i], xmin, ymin, hx, hy)
        ptr = fill_ptr[ix,iy]
        bin_lists[ix,iy][ptr] = i
        fill_ptr[ix,iy] += 1
    end
    return bin_count, bin_lists
end

@inline function bin_idx(z, xmin, ymin, hx, hy)
    ix = Int64(floor( (z[1]-xmin)/hx )) + 1
    iy = Int64(floor( (z[2]-ymin)/hy )) + 1
    return ix, iy
end


# This code is a direct port of the adaptive discretization routine in Chunkie: https://github.com/fastalgorithms/chunkie
function discretize_adap(curve, panelorder;ta=0.0,tb=2*pi,ifclosed=true,chsmall=Inf,nover=0,tol=1.0e-6,lvlr='a',maxchunklen=Inf,lvlrfrac=2.0,nchmax=10000,lvlrfac=2.0)
    splitcomplex(z) = [real(z), imag(z)] # from C to R^2
    if isa(curve, Array)
        # TODO: Would be nicer to use multiple dispatch
        curves = curve
        Ncurves = length(curves)
        #        @assert length(numpanels) == Ncurves
        dcurves = Array{DiscreteCurve}(undef,Ncurves)
        for i = 1 : Ncurves
            dcurves[i] = discretize_adap(curves[i],panelorder;ta,tb,ifclosed,chsmall,nover,tol,lvlr,maxchunklen,lvlrfrac,nchmax,lvlrfac)
        end
        return multiply_connected(dcurves)
    end
    
    fcurve = curve

    
    ifprocess = zeros(nchmax,1)
    
    k = panelorder
    k2 = 2*k

    xs, ws = gausslegendre(k)
    xs2, ws2 = gausslegendre(k2)
    L = legendre_matrix(k)
    L2 = legendre_matrix(k2)

    # Start subdividing into panels
    
    ab = zeros(2,nchmax);
    adjs = zeros(Int64,2,nchmax);
    ab[1,1]=ta;
    ab[2,1]=tb;
    nch=1;
    if ifclosed
        adjs[1,1]=1;
        adjs[2,1]=1;
    else
        adjs[1,1]=-1;
        adjs[2,1]=-1;
    end
    nchnew=nch;

    maxiter_res=10000;

    rad_curr = 0;

    for ijk = 1:maxiter_res

        #       loop through all existing chunks, if resolved store, if not split
        xmin =  Inf;
        xmax = -Inf;
        ymin =  Inf;
        ymax = -Inf;
        
        ifdone=1;
        for ich=1:nchnew

            if (ifprocess[ich] != 1)
                ifprocess[ich]=1;
                
                a=ab[1,ich];
                b=ab[2,ich];
                
                ts = a .+ (b-a)*(xs2 .+1)./2;
                r = curve.tau.(ts)
                r = vcat(copy(real.(r)'),copy(imag.(r)'))
                d = curve.dtau.(ts)
                d2 = curve.d2tau.(ts)
                dsdt = abs.(d)

                rlself = chunklength(curve,a,b,xs,ws)
                
                zd = d
                vd = abs.(zd);
                zdd= d2
                dkappa = imag.(zdd.*conj.(zd))./abs.(zd).^2;

                cfs = L2*vd;
                errs0 = sum(abs.(cfs[1:k]).^2);
                errs = sum(abs.(cfs[k+1:k2]).^2);
                err1 = sqrt.(errs ./errs0 ./k);
                
                resol_speed_test = err1>tol;

                
                xmax = max(xmax,maximum(r[1,:]));
                ymax = max(ymax,maximum(r[2,:]));
                xmin = min(xmin,minimum(r[1,:]));
                ymin = min(ymin,minimum(r[2,:]));
                
                cfsx = L2*copy(r[1,:])
                cfsy = L2*copy(r[2,:])
                errx = sum(abs.(cfsx[k+1:k2]).^2 ./k);
                erry = sum(abs.(cfsy[k+1:k2]).^2 ./k);
                errx = sqrt.(errx);
                erry = sqrt.(erry);
                
                resol_curve_test = true;
                
                if (ijk >1)
                    if (errx/rad_curr<tol && erry/rad_curr<tol)
                        resol_curve_test = false;
                    end
                end
                total_curve = (b-a)/2*sum(abs.(dkappa).*ws2);
                total_curve_test = total_curve >= (2*pi)/3;
                ###############################
                

                #       . . . mark as processed and resolved if less than tol
                test1 = adjs[1,ich] <= 0
                test2 = adjs[2,ich] <= 0
                test = (test1 || test2) && (rlself > chsmall)

                if (resol_speed_test || resol_curve_test  || total_curve_test || rlself > maxchunklen || (((adjs[1,ich] <= 0)||(adjs[2,ich] <= 0))&&(rlself > chsmall)))
                    #       . . . if here, not resolved
                    #       divide - first update the adjacency list
                    if (nch+1 > nchmax)
                        error("too many chunks")
                    end

                    ifprocess[ich]=0;
                    ifdone=0;

                    if ((nch == 1) && ifclosed)
                        adjs[1,nch]=2;
                        adjs[2,nch]=2;
                        adjs[1,nch+1]=1;
                        adjs[2,nch+1]=1;
                    end

                    if ((nch == 1) && (!ifclosed))
                        adjs[1,nch]=-1;
                        adjs[2,nch]=2;
                        adjs[1,nch+1]=1;
                        adjs[2,nch+1]=-1;
                    end

                    if (nch > 1)
                        iold2=adjs[2,ich];
                        adjs[2,ich]=nch+1;
                        if (iold2 > 0)
                            adjs[1,iold2]=nch+1;
                        end	
                        adjs[1,nch+1]=ich;
                        adjs[2,nch+1]=iold2;
                    end
                    #       now update the endpoints in ab

                    ab[1,ich]=a;
                    ab[2,ich]=(a+b)/2;

                    nch=nch+1;
                    ab[1,nch]=(a+b)/2;
                    ab[2,nch]=b;
                end
            end
        end
        if ((ifdone == 1) && (nchnew == nch))
            break;
        end
        nchnew=nch;
        
        rad_curr = max(xmax-xmin,ymax-ymin);

    end


    #       the curve should be resolved to precision tol now on
    #       each interval ab(,i)
    #       check the size of adjacent neighboring chunks - if off by a
    #       factor of more than 2, split them as well. iterate until done.
    
    
    if (lvlr=='a')||(lvlr=='t')
        maxiter_adj=1000;
        for ijk = 1:maxiter_adj

            nchold=nch;
            ifdone=1;
            for i = 1:nchold
                i1=adjs[1,i];
                i2=adjs[2,i];

                #       calculate chunk lengths

                a=ab[1,i]
                b=ab[2,i]
                
                if lvlr=='a'

                    rlself = chunklength(fcurve,a,b,xs,ws)                        
                    
                    rl1=rlself;
                    rl2=rlself;

                    if (i1 > 0)
                        a1=ab[1,i1];
                        b1=ab[2,i1];
                        rl1 = chunklength(fcurve,a1,b1,xs,ws);
                    end
                    if (i2 > 0)
                        a2=ab[1,i2];
                        b2=ab[2,i2];
                        rl2 = chunklength(fcurve,a2,b2,xs,ws);
                    end
                else
                    
                    rlself = b-a;
                    rl1 = rlself;
                    rl2 = rlself;
                    if (i1 > 0)
                        rl1 = ab[2,i1]-ab[1,i1];
                    end
                    if (i2 > 0)
                        rl2 = ab[2,i2]-ab[1,i2];
                    end
                end

                #       only check if self is larger than either of adjacent blocks,
                #       iterating a couple times will catch everything

                if (rlself > lvlrfac*rl1 || rlself > lvlrfac*rl2)

                    #       split chunk i now, and recalculate nodes, ders, etc

                    if (nch + 1 > nchmax)
                        error("too many chunks")
                    end


                    ifdone=0;
                    a=ab[1,i];
                    b=ab[2,i];
                    ab2=(a+b)/2;

                    i1=adjs[1,i];
                    i2=adjs[2,i];

                    adjs[1,i] = i1;
                    adjs[2,i] = nch+1;

                    #       . . . first update nch+1

                    adjs[1,nch+1] = i;
                    adjs[2,nch+1] = i2;

                    #       . . . if there's an i2, update it

                    if (i2 > 0)
                        adjs[1,i] = nch+1;
                    end

                    nch=nch+1;
                    ab[1,i]=a;
                    ab[2,i]=ab2;

                    ab[1,nch]=ab2;
                    ab[2,nch]=b;
                end
            end

            if (ifdone == 1)
                break;
            end

        end
    end

    #       go ahead and oversample by nover, updating
    #       the adjacency information adjs along the way


    if (nover > 0) 
        for ijk = 1:nover

            nchold=nch;
            for i = 1:nchold
                a=ab[1,i];
                b=ab[2,i];
		#       find ab2 using newton such that 
		#       len(a,ab2)=len(ab2,b)=half the chunk length
                rl = chunklength(fcurve,a,b,xs,ws);
                rlhalf=rl/2;
                thresh=1.0d-8;
                ifnewt=0;
                ab0=(a+b)/2;
                for iter = 1:1000

                    rl1 = chunklength(fcurve,a,ab0,xs,ws);
                    
                    
                    d = fcurve.dtau.(ab0)
                    dsdt = abs.(d)
                    ab1=ab0-(rl1-rlhalf)/dsdt;

                    err=rl1-rlhalf;
                    if (abs(err) < thresh)
                        ifnewt=ifnewt+1;
                    end

                    if (ifnewt == 3)
                        break;
                    end
                    ab0=ab1;
                end
	        
                if (ifnewt < 3) 
                    error("newton failed in chunkerfunc");
                end
                ab2=ab1;

                i1=adjs[1,i];
                i2=adjs[2,i]
                adjs[2,i]=nch+1;
                if (i2 > 0)
                    adjs[1,i2]=nch+1;
                end

                if (nch + 1 > nchmax)
                    error("too many chunks")
                end

                adjs[1,nch+1]=i;
                adjs[2,nch+1]=i2;
	        
                ab[1,i]=a;
                ab[2,i]=ab2;
	        
                nch=nch+1;

                ab[1,nch]=ab2;
                ab[2,nch]=b;
            end
        end
    end

    #       up to here, everything has been done in parameter space, [ta,tb]
    #       . . . finally evaluate the k nodes on each chunk, along with 
    #       derivatives and chunk lengths

    # Prepare structures
    numpanels = nch
    numpoints = numpanels * panelorder
    glpoints, glweights = gausslegendre(panelorder)
    points = zeros(2, numpoints)
    normals = zeros(2, numpoints)
    derivatives = zeros(2, numpoints)
    derivatives2 = zeros(2, numpoints)
    weights = zeros(numpoints)
    dS = zeros(numpoints)
    curvature = zeros(numpoints)
    t_edges = zeros(2, numpanels)
    edges = zeros(4, numpanels) # start and end
    n_edges = zeros(4, numpanels)    
    L = legendre_matrix(panelorder)
    resolution = 0.0
    # Populate
    for i in 1:numpanels
        # setup quadrature for t in [a,b]
        a=ab[1,i];
        b=ab[2,i];
        
        ts = a .+ (b-a)*(xs .+1)/2;
        t_edges[:, i] = [a, b]
        edges[1:2, i] = splitcomplex(curve.tau(a))
        edges[3:4, i] = splitcomplex(curve.tau(b))
        n_edges[1:2, i] = splitcomplex(curve.normal(a))
        n_edges[3:4, i] = splitcomplex(curve.normal(b))                        
        dt = b-a
        for j=1:panelorder
            idx = j + panelorder*(i-1)
            t = a + (1+xs[j])*dt/2
            w = ws[j]*dt/2
            z = curve.tau(t)
            zp = curve.dtau(t)
            zpp = curve.d2tau(t)
            n = curve.normal(t)
            points[:, idx] = splitcomplex(z)
            derivatives[:, idx] = splitcomplex(zp)
            derivatives2[:, idx] = splitcomplex(zpp)
            normals[:, idx] = splitcomplex(n)
            weights[idx] = w
            dS[idx] = w*abs.(zp)
            curvature[idx] = -imag(conj(zp)*zpp/abs.(zp)^3)
        end
        # Get resolution estimate
        idx = (1:panelorder) .+ panelorder*(i-1)
        coeff = L * (dS[idx]./weights[idx])
        thisresolution = abs(coeff[end]) + abs(coeff[end-1])
        resolution = max(resolution, thisresolution)
    end
    
    prevpanel = adjs[1,1:numpanels]
    nextpanel = adjs[2,1:numpanels]
    curvenum = ones(Int64, numpoints)
    
    @info @sprintf("  Grid resolution: %.2e\n", resolution) 

    DiscreteCurve(
        panelorder,
        numpanels,
        numpoints,
        edges,
        n_edges,
        t_edges,
        points,
        normals,
        derivatives,
        weights,
        dS,
        curvature,
        prevpanel,
        nextpanel,
        curvenum
    )
    
    
end

function chunklength(fcurve,a,b,xs,ws)
    ts = a .+ (b-a)*(xs .+1)/2;
    d = fcurve.dtau.(ts)
    dsdt = abs.(d)
    rlself = dot(dsdt,ws)*(b-a)/2;
    return rlself 
end


end # module




