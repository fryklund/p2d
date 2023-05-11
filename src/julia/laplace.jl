import bieps2d.DiscreteCurve

include("helsingquad_laplace.jl")
include("legendre.jl")
function layer_potential_direct(grid::DiscreteCurve,
                                density::Array{Float64},
                                target::Array{Float64};
				slp::Int64 = 0,
				dlp::Int64 = 0,
				S::Matrix{Float64},
				exterior::Bool = false)
    M = grid.curvenum[end] - 1
    N = size(target,2)   
    u = zeros(N)
    for idx_trg = 1:N
        trg = target[:,idx_trg]
        utrg = 0.0
        for i = 1:length(grid.weights)        
            utrg += quad_kernel(density[i],
                                grid.dS[i],
                                grid.normals[:,i],
                                grid.points[:,i],
                                trg;
				slp = slp,
				dlp = dlp)
        end
        u[idx_trg] = utrg
    end
    if (M > 0) & (dlp == 1)			
	for k = 1:M+exterior
	    u .+= density[end-M+k-exterior] .* log.(sqrt.((target[1,:] .- S[1,k]).^2 .+ (target[2,:] .- S[2,k]).^2))
	end
    end		 
    return u .+ exterior * sum(density[1:end-(M>0)*(M+exterior)] .* grid.dS) / 2 / pi
end

function layer_potential_fmm(grid::DiscreteCurve,
                             density::Array{Float64},
                             target::Array{Float64},
			     interior_near:: BitArray{1},
			     S;
			     slp::Int64 = 0,
			     dlp::Int64 = 0,
			     exterior = false)
    # Compute single and/or double layer potential with FMM
    ndS = grid.normals .* grid.dS'
    M = grid.curvenum[end] - 1

    if slp == 1 && dlp != 1
	charge = density.*grid.dS
	dipstr = density * 0.0
    elseif dlp == 1
	charge = density[1:end-M - exterior*(M>0)] * 0.0
	dipstr = density[1:end-M - exterior*(M>0)] 
    else
	@error "Choose slp or dlp"
    end	
    U = rfmm2d(source=grid.points, charge=charge, target=target, dipstr=dipstr, dipvec=ndS,ifpot=false, ifpottarg=true)
    u = U.pottarg ./ (-2*pi)
    ucorrection = layer_potential_correction_Op(grid, density,target[:,interior_near]; slp=slp, dlp=dlp)
    u[interior_near] .+= ucorrection
    if (M > 0) & (dlp == 1)			
	for k = 1:M+exterior
	    u .+= density[end-M+k-exterior] .* log.(sqrt.((target[1,:] .- S[1,k]).^2 .+ (target[2,:] .- S[2,k]).^2))
	end
    end		 
    return u .+ exterior * sum(density[1:end-(M>0)*(M+exterior)] .* grid.dS) / 2 / pi
end

function layer_potential_correction(grid::DiscreteCurve,
                                    density::Array{Float64},
                                    target::Array{Float64};
				    slp::Int64 = 0,
				    dlp::Int64 = 0)
    N = size(target,2)
    n = grid.panelorder
    correction = zeros(N)
    for idx_trg = 1:N
        trg = target[:,idx_trg]
	trgcorrection = 0.0
        for panel_idx = 1:grid.numpanels
	    idx = n * (panel_idx-1) .+ (1:n)
	    # Slice out panel
	    points = grid.points[:,idx]
	    normals = grid.normals[:,idx]
	    dS = grid.dS[idx]
	    den = density[idx]
	    panelcorrection = 0.0
	    rmin = Inf
	    for i=1:n
		rvec = points[:,i] .- trg
		rnorm = norm(rvec)
		rmin = min(rmin, rnorm)
	    end
	    h = sum(dS)
	    if rmin < h
		# Correction from panel panel_idx
		edges = grid.edges[:, panel_idx]
		za = edges[1] + 1im*edges[2]
		zb = edges[3] + 1im*edges[4]
		zt = trg[1] + 1im*trg[2]
		zj = points[1,:] .+ 1im*points[2,:]
	        nuj = -normals[1,:] .-1im*normals[2,:]
	        zpwj = dS .* nuj ./ 1im
		(wcorrL, wcorrC)  =  wLCinit(za, zb, zt, zj, nuj, zpwj)
		#dummy 133.7
		if slp == 1 && dlp != 1
		    (GS, GL, GC) = slp_kernel_split(133.7)
		elseif dlp == 1
		    (GS, GL, GC) = dlp_kernel_split(133.7)
		else
		    @error "choose slp or dlp"
		end
		panelcorrection = GC*sum(wcorrC.*den) + GL*sum(wcorrL.*dS.*den)
            end
            trgcorrection += panelcorrection
        end
        correction[idx_trg] = trgcorrection
    end
    return correction
end

function layer_potential_correction_Op(grid::DiscreteCurve,
                                       density::Array{Float64},
                                       targets::Array{Float64};
				       slp::Int64 = 0,
				       dlp::Int64 = 0,
                                       interior::Int64=1)

    dlim = 1.2
    ifleft = 1#interior
    
    numpanels = grid.numpanels
    ngl = Int64(grid.panelorder)
    myind = zeros(Int64,ngl)

    panlen = zeros(Float64,numpanels)
    
    a = zeros(Complex{Float64},1)
    b = zeros(Complex{Float64},1)
    cc = zeros(Complex{Float64},1)

    z = grid.points[1,:] .+ 1im.*grid.points[2,:]
    nz = grid.normals[1,:] .+ 1im.*grid.normals[2,:]
    nt = size(targets,2)
    zt = targets[1,:] .+ 1im.*targets[2,:]
    dS = grid.dS
    correction = zeros(Float64,nt,1)

    U = zeros(Complex{Float64},ngl,ngl)
    v1 = zeros(Complex{Float64},length(density))
    
    for k=1:numpanels
        myind[:] .= (k-1)*ngl + 1:k*ngl
        edges = grid.edges[:, k]
        a = edges[1] + 1im*edges[2]
	b = edges[3] + 1im*edges[4]
        cc = (b-a)/2.0
        U[:,:] .= wLCHS_src_precomp(a,b,z[myind])
        #        v1[myind]=U*(density[myind]*cc.*conj.(nz[myind]))*(-1/pi/2)

        #        v1[myind]=U*(density[myind])
        v1[myind]=transpose(U)*(density[myind])

        #        v1[myind,:]=U
        panlen[k] = sum(grid.dS[myind])
    end
    
    for k=1:nt
        dr=abs.(z .- zt[k])
        dr=reshape(dr,(ngl,numpanels))
        dr=minimum(dr,dims=1)
        d=vec(dr)./panlen
        ind=findall(d .<dlim)
        for kk in ind
            myind[:] .= (kk-1)*ngl + 1:kk*ngl
            edges = grid.edges[:, kk]
            a = edges[1] + 1im*edges[2]
	    b = edges[3] + 1im*edges[4]
            cc = (b-a)/2.0
            zsc=z[myind]
            sol=density[myind]
            nuj = .-nz[myind]
	    zpwj = dS[myind] .* nuj ./ 1im
            CauC=wCinit(a,b,zt[k],zsc,ifleft);
            #            wcorrL, wcmpC, A, p=wLCinit(a, b, zt[k], zsc, nuj, zpwj)
            GC = -1/(2*pi)
            #            CauC = p[1:end-1]
            
            # dot product, O(p) cost for computing the correction,
            # then subtract the smooth quadrature contribution
            #            correction[k]=correction[k].+imag(transpose(LogC)*v1[myind])[1].+(-1/pi/2)*(log.(abs.((zsc .-zt[k])./cc))'*sol)[1]
            #            corrk = dot(imag.(v1[myind,:]*CauC),sol) .-dot(imag.(zpwj ./ (zsc .- zt[k])),sol)
            #            corrk = imag(sum((A\p[1:end-1]).*sol)) -sum(imag.(zpwj ./ (zsc .- zt[k])) .*sol)
            #            U[:,:] .= wLCHS_src_precomp(a,b,z[myind])
            #            corrk = imag(transpose(U*CauC)*sol) -sum(imag.(zpwj ./ (zsc .- zt[k])) .*sol)
            #            corrk = imag(sum(v1[myind].*CauC)) -sum(imag.(zpwj ./ (zsc .- zt[k])) .*sol)
            corrk = imag(sum(v1[myind].*CauC)) -sum(imag.(zpwj ./ (zsc .- zt[k])) .*sol)

            corrk = corrk*GC
            correction[k]=correction[k].+ corrk
        end
    end
    
    
    
    return correction
end

function layer_potential(grid::DiscreteCurve,
                         density::Array{Float64},
                         target::Array{Float64};
			 slp::Int64,
			 dlp::Int64,
			 S::Matrix{Float64},
			 exterior::Bool=false)
    N = size(target,2)
    M = grid.curvenum[end] - 1
    u = zeros(N)
    for idx_trg = 1:N
        trg = target[:,idx_trg]
        utrg = 0.0
        for ipan = 1:grid.numpanels
            up, uc = panel_quad(grid, density[1:end-(M+exterior)*(M>0)], ipan, trg; slp=slp, dlp=dlp)
            utrg += up+uc
        end
        u[idx_trg] = utrg
    end
    

    if (M > 0) & (dlp == 1)			
	for k = 1:M+exterior
	    u .+= density[end-M+k-exterior] .* log.(sqrt.((target[1,:] .- S[1,k]).^2 .+ (target[2,:] .- S[2,k]).^2))
	end
    end
    return u .+ exterior * sum(density[1:end-(M+exterior)*(M>0)] .* grid.dS) / 2 / pi
end

# DLP on curve
function layer_potential_self(grid::DiscreteCurve,
                              density
                              )
    npoints = length(grid.weights)
    u = zeros(npoints)
    for i=1:npoints
        ui = 0
        zi = grid.points[:,i]
        for j=1:i-1
            ui += quad_kernel(density[j], grid.dS[j], grid.normals[:,j], grid.points[:,j], zi)
        end
        for j=i+1:npoints
            ui += quad_kernel(density[j], grid.dS[j], grid.normals[:,j], grid.points[:,j], zi)
        end
        # Self
        (GS0, GL0, GC0) = kernel_split(0)        
        ui += (GS0 -grid.curvature[i]/2*GC0)*density[i]*grid.dS[i]
        u[i] = ui
    end
    return u
end

function tovec(inputtype::Tuple{Vector{Float64}, Matrix{Float64}, Matrix{Float64}})
    return inputtype[1]
end

function system_matvec(grid::DiscreteCurve,S;exterior=false)
    M = maximum(grid.curvenum)-1
    iprec = 5
    source = grid.points
    ifcharge = 0
    charge = zeros(grid.numpoints)
    ifdipole = 1
    dipstr = zeros(grid.numpoints)
    dipvec = grid.normals .* grid.dS'
    ifpot = 1
    ifgrad = 0
    ifhess = 0

    if exterior # exterior Laplace problem
	if M > 0 # multiply connected
	    logSk = zeros(Float64,grid.numpoints,M + 1)
	    sumdens = zeros(Float64,M,grid.numpoints)
	    for k = 1:M
		logSk[:,k] = log.(sqrt.((grid.points[1,:] .- S[1,k]).^2 .+ (grid.points[2,:] .- S[2,k]).^2))
		sumdens[k,findall(grid.curvenum .== k)] .= grid.dS[findall(grid.curvenum .== k)]
	    end
	    logSk[:,M+1] = log.(sqrt.((grid.points[1,:] .- S[1,M+1]).^2 .+ (grid.points[2,:] .- S[2,M+1]).^2))
	    return x -> vcat(.-x[1:end-M-(M>0)]/2 .- (1/(2*pi))*tovec(bieps2d.CurveDiscretization.rfmm2dpartself(iprec, source, ifcharge, charge, ifdipole, collect(x[1:end-M-(M>0)]), dipvec, ifpot, ifgrad, ifhess)) .+ x[1:end-M-(M>0)].*grid.dS.*grid.curvature / (4*pi) .+ sum(x[end-M:end]' .* logSk,dims=2) .+ sum(x[1:end-M-(M>0)] .* grid.dS)/2/pi, sumdens * x[1:end-M-(M>0)],sum(x[end-M:end]))				 
	else # simply connected
	    return x -> .-x/2 .- (1/(2*pi))*tovec(bieps2d.CurveDiscretization.rfmm2dpartself(iprec, source, ifcharge, charge, ifdipole, collect(x[1:end]), dipvec, ifpot, ifgrad, ifhess)) .+ x[1:end].*grid.dS.*grid.curvature / (4*pi) .+ sum(x .* grid.dS)/2/pi		 
	end
    else # interior Laplace
	if M > 0 # multiply connected
	    logSk = zeros(Float64,grid.numpoints,M + exterior)
	    sumdens = zeros(Float64,M,grid.numpoints) 
	    for k = 1:M
		logSk[:,k] = log.(sqrt.((grid.points[1,:] .- S[1,k]).^2 .+ (grid.points[2,:] .- S[2,k]).^2))
		sumdens[k,findall(grid.curvenum .== k+1)] .= grid.dS[findall(grid.curvenum .== k+1)]
	    end
	    return x -> vcat(.-x[1:end-M]/2 .- (1/(2*pi))*tovec(bieps2d.CurveDiscretization.rfmm2dpartself(iprec, source, ifcharge, charge, ifdipole, collect(x[1:end-M]), dipvec, ifpot, ifgrad, ifhess)) .+ x[1:end-M].*grid.dS.*grid.curvature / (4*pi) .+ sum(x[end-M+1:end]' .* logSk,dims=2), sumdens * x[1:end-M]) 
	else # simply connected
	    return x -> .-x/2 .- (1/(2*pi))*tovec(bieps2d.CurveDiscretization.rfmm2dpartself(iprec, source, ifcharge, charge, ifdipole, collect(x), dipvec, ifpot, ifgrad, ifhess)) .+ x.*grid.dS.*grid.curvature / (4*pi)	 
	end
    end

end

# Only for interior
function system_matrix(grid::DiscreteCurve,S::Matrix{Float64};exterior=false)
    # cormat = correction_matrix(grid) # Not needed for Laplace
    dlpmat = assemble_matrix(grid)
    if exterior
	dlpmat .+= repeat(copy(grid.dS'),grid.numpoints,1)/2/pi
    end
    M = grid.curvenum[end]
    if M == 1
	return -I * 1/2 + dlpmat # + cormat
    else
	if !exterior
	    d = 1
	else
	    d = 0
	end  
	N = grid.numpoints
	B = zeros(Float64,N,M-d)
	C = zeros(Float64,M-1,N)
	D = zeros(Float64,M-1,M-d)
	for k = 1:M-d
	    B[:,k] .= log.(sqrt.((grid.points[1,:] .- S[1,k]).^2 .+ (grid.points[2,:] .- S[2,k]).^2))
	end
	temp = cumsum(vcat(1,16*numpanelsv))
	
	for i = 1:M-1
	    idx = temp[i+d]:(temp[i+1+d]-1)
	    temp_row = zeros(Float64,N);
	    temp_row[idx] .= dcurve.dS[idx]
	    C[i,:] = C[i,:] + temp_row;
	end
	if exterior
	    return vcat(hcat(-I * 1/2 + dlpmat,B),hcat(C,D),hcat(zeros(Float64,1,N),ones(Float64,1,M)))
	else
	    return vcat(hcat(-I * 1/2 + dlpmat,B),hcat(C,D))
	end
    end
end

function system_matrix_neumann(grid::DiscreteCurve)
    # cormat = correction_matrix(grid) # Not needed for Laplace
    dlpmat = assemble_matrix_neumann(grid)
    return I * 1/2 - dlpmat # + cormat
end

function quad_kernel(den, dS, n, z, zt; slp::Int64 = 0, dlp::Int64 = 0)
    dot(a, b) = a[1]*b[1] + a[2]*b[2]
    norm2(a) = dot(a,a)
    r = (z .- zt)
    normr = sqrt(norm2(r))
    if slp == 1 && dlp != 1
	(GS, GL, GC) = slp_kernel_split(133.7)
    elseif dlp == 1
	(GS, GL, GC) = dlp_kernel_split(133.7)
    else
	@error "choose slp or dlp"
    end
    return (GS + GL*log(normr) + GC*dot(r, n)/norm2(r))*den*dS
end


function dlp_kernel_split(r)
    GS = 0.0
    GL = 0.0
    GC = -1 / (2*pi)
    return (GS, GL, GC)
end

function slp_kernel_split(r)
    GS = 0.0
    GL = -1 / (2*pi)
    GC = 0.0
    return (GS, GL, GC)
end

function assemble_matrix(grid::DiscreteCurve)
    npoints = grid.numpoints
    A = zeros(npoints, npoints)
    (GS0, GL0, GC0) = dlp_kernel_split(0)        
    for i=1:npoints
        zi = grid.points[:,i]
        for j=1:i-1
            A[i,j] = quad_kernel(1, grid.dS[j], grid.normals[:,j], grid.points[:,j], zi; slp = 0, dlp = 1)
        end
        for j=i+1:npoints
            A[i,j] = quad_kernel(1, grid.dS[j], grid.normals[:,j], grid.points[:,j], zi; slp = 0, dlp = 1)
        end
        # Self
        A[i,i] = (GS0 - grid.curvature[i]/2*GC0)*grid.dS[i]
    end
    return A
end

function assemble_matrix_neumann(grid::DiscreteCurve)
    npoints = grid.numpoints
    A = zeros(npoints, npoints)
    (GS0, GL0, GC0) = dlp_kernel_split(0)
    for i=1:npoints
        zi = grid.points[:,i]
	ni = grid.normals[:,i]
        for j=1:i-1
            A[i,j] = quad_kernel(1, grid.dS[j], ni, grid.points[:,j], zi; slp = 0, dlp = 1) - GC0*grid.dS[j]
        end
        for j=i+1:npoints
            A[i,j] = quad_kernel(1, grid.dS[j], ni, grid.points[:,j], zi; slp = 0, dlp = 1) - GC0*grid.dS[j]
        end
        A[i,i] = (GS0 + (grid.curvature[i]/2 - 1)*GC0)*grid.dS[i]
    end
    return A
end

function panel_quad(grid::DiscreteCurve, density::Array{Float64}, panel_idx::Int64, xt::Array{Float64}; slp::Int64, dlp::Int64)
    n = grid.panelorder
    idx = n*(panel_idx-1) .+ (1:n)
    # Slice out panel
    points = grid.points[:,idx]
    normals = grid.normals[:,idx]
    dS = grid.dS[idx]
    den = density[idx]
    up = 0.0
    rmin = Inf
    if slp == 1 && dlp != 1
	(GS, GL, GC) = slp_kernel_split(133.7)
    elseif dlp == 1
	(GS, GL, GC) = dlp_kernel_split(133.7)
    else
	@error "choose slp or dlp"
    end
    for i=1:n
        rvec = points[:,i] .- xt
        rnorm = norm(rvec)
        rmin = min(rmin, rnorm)	
        ui = ( GS + GL*log(rnorm) +
               GC * dot(rvec, normals[:,i]) / rnorm^2 ) * den[i]*dS[i]
        up += ui
    end
    h = sum(dS)
    uc  = 0.0
    if rmin < h
        # Correct
        edges = grid.edges[:, panel_idx]
        za = edges[1] + 1im*edges[2]
        zb = edges[3] + 1im*edges[4]
        zt = xt[1] + 1im*xt[2]
        zj = points[1,:] .+ 1im*points[2,:]
        nuj = -normals[1,:] .- 1im*normals[2,:]
        zpwj = dS .* nuj ./ 1im # outward directed normal
        (wcorrL, wcorrC)  =  wLCinit(za, zb, zt, zj, nuj, zpwj)
        uc = GC*sum(wcorrC.*den) +
            GL*sum(wcorrL.*den.*dS) 
    end
    return up, uc
end
