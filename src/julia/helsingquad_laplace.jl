import bieps2d.DiscreteCurve
import GSL
using FastGaussQuadrature

"""
    Assemble block diagonal matrix for a curve discretized using
    GL panels. Basically implements eq (30) of Helsing & Holst, A. (2015).
    """
function correction_matrix(grid::DiscreteCurve)
    glpoints, glweights = gausslegendre(grid.panelorder)
    n = grid.panelorder    
    # Build correction matrix
    M = zeros(grid.numpoints, grid.numpoints)
    WfrakL_self = WfrakLcomp(0, 1, glpoints);
    for panel_idx = 1:grid.numpanels
        ta, tb = grid.t_edges[:, panel_idx]
        point_idx = n*(panel_idx-1) .+ (1:n);        
        # Correct self and nearest neighbors
        for d = -1:1
            nb_panel_idx = 1 .+ mod(panel_idx+d-1, grid.numpanels);
            nb_point_idx = n*(nb_panel_idx-1) .+ (1:n);
            ta, tb = grid.t_edges[:, panel_idx]            
            nb_ta, nb_tb = grid.t_edges[:, nb_panel_idx]
            # Correct for panels not being of equal
            # length _in parametrization_            
            alpha = (tb-ta) / (nb_tb-nb_ta)
            trans = -d*(1+alpha)
            scale = alpha
            if d==0
                glpoints_scaled = glpoints
                WfrakL = WfrakL_self
            else
                glpoints_scaled = @. -d*(1+alpha*(1-d*glpoints))
                WfrakL = WfrakLcomp(trans, scale, glpoints);
            end
            # Assemble block (including diagonal)
            for j=1:n            
                for i=1:n
                    i_idx = n*(panel_idx-1) + i
                    j_idx = n*(nb_panel_idx-1) + j
                    nb_wcorr = WfrakL[i, j]/glweights[j] -
                        log(abs(glpoints_scaled[i] - glpoints[j]));
                    zi = grid.points[:, i_idx];
                    zj = grid.points[:, j_idx];
                    r = zi.-zj
                    r = sqrt(r[1]^2+r[2]^2 )                
                    (GS, GL, GC) = kernel_split(r)
                    M[i_idx, j_idx] = GL*nb_wcorr*grid.dS[j_idx];
                end
            end
            # Correct diagonal
            if d==0
                for i=1:n
                    i_idx = n*(panel_idx-1) + i
                    si = grid.dS[i_idx] / grid.weights[i_idx]
                    wcorr = WfrakL[i,i]/glweights[i] +
                        log(abs((tb-ta)*si/2))
                    (GS, GL, GC) = kernel_split(0)
                    M[i_idx, i_idx] = GL*wcorr*grid.dS[i_idx];
                end
            end            
        end # End neighbor loop
    end # End panel loop
    return M
end


"""
    The choice of input arguments trans=0 and scale=1 gives WL
    for ri and rj on the same quadrature panel γp. The choice
    trans=±2 and scale=1 gives WL for ri on a neighboring panel
    γp±1, assuming it is equal in parameter length. The input
    argument tfrak is a column vector whose entries contain the
    canonical nodes ti.
            
    Code by Johan Helsing
    """
function WfrakLcomp(trans,scale,tfrak)
    npt = length(tfrak)
    A = ones(npt, npt)
    for k=2:npt
        A[:,k] = tfrak .* A[:,k-1]
    end
    tt = trans .+ scale*tfrak
    Q = zeros(npt, npt)
    p = zeros(1,npt+1)
    c = (1 .- (-1).^(1:npt)) ./ (1:npt)
    for j = 1:npt
        p[1] = log(abs((1-tt[j]) / (1+tt[j])))
        for k = 1:npt
            p[k+1] = p[k] * tt[j] + c[k]	   
        end
        Q[j,1:2:npt-1] = log(abs(1-tt[j]^2)) .- p[2:2:npt]
        Q[j,2:2:npt] = p[1] .- p[3:2:npt+1]
        Q[j,:] = Q[j,:] ./ (1:npt)
    end
    WfrakL=Q/A
    return WfrakL
end

function wLCinit(ra, rb, r, rj, nuj, rpwj)
    nuj = -nuj
    npt = length(rj)
    # Transform endpoints to [-1,1]
    dr = (rb - ra) / 2
    rtr = (r - (rb + ra) / 2) /dr
    rjtr = (rj .- (rb + ra) /2 ) / dr
    A = complex(ones(npt, npt));
    for k = 2:npt
	A[k,:] = rjtr .* A[k-1,:] # vandermonde transposed
    end
    p = complex(zeros(npt+1))
    q = complex(zeros(npt))
    c = (1 .- (-1).^(1:npt)) ./ (1:npt)
    p[1] = log(1-rtr) - log(-1-rtr)
    p1 = log(1-rtr) + log(-1-rtr)
    # Signs here depend on how we have defined
    # direction of curve and normals
    # INTERIOR and EXTERIOR ok
    if imag(rtr)<0 && abs(real(rtr))<1
        p[1] += 2im*pi
        p1 -= 2im*pi
    end

    for k = 1:npt
	p[k+1] = rtr*p[k] + c[k]
    end
    
    q[1:2:npt-1] = p1 .- p[2:2:npt]
    q[2:2:npt] = p[1] .- p[3:2:npt+1]
    q = q ./ (1:npt)
    wcorrL = imag(A\q * dr.* conj(nuj)) ./ abs.(rpwj)
    @. wcorrL -= log(abs((rj - r) / dr))
    wcmpC = imag(A\p[1:npt] .- rpwj ./ (rj .- r))
    return wcorrL, wcmpC, A, p
end


function wLCHS_src_precomp(a::Complex{Float64},b::Complex{Float64},zsc::Vector{ComplexF64})
    # computation about the source points, used before wLCHSnew  
    # O(p^3) operations
    #
    Ng=length(zsc)
    npt=Ng
    cc=(b-a)/2
    zsctr=(zsc .-(b+a)/2)./cc
    
    #    U = zeros(Complex{Float64},Ng,Ng)

    #    U[:,:] .= legepols(zsctr,Ng-1)
    U=legepols(zsctr,Ng-1)

    #    U = complex(ones(npt, npt));
    #    for k = 2:npt
    #			U[k,:] = zsctr .* U[k-1,:] # vandermonde transposed
    #    end
    
    U=inv(transpose(U))
    # slightly more accurate than inv    
    #    [u,s,v]=svd(U);
    #    U=v*diag(1./diag(s))*u';

    return U
    
end

function wCinit(ra::Complex{Float64}, rb::Complex{Float64}, r::Complex{Float64}, rj::Vector{ComplexF64},ifleft::Int64)

    #=
    npt = length(rj)

    # Transform endpoints to [-1,1]
    dr = (rb - ra) / 2
    rtr = (r - (rb + ra) / 2) /dr
    rjtr = (rj .- (rb + ra) /2 ) / dr
    p = complex(zeros(npt+1))
    c = (1 .- (-1).^(1:npt)) ./ (1:npt)
    p[1] = log(1-rtr) - log(-1-rtr)
    p1 = log(1-rtr) + log(-1-rtr)
    # Signs here depend on how we have defined
    # direction of curve and normals
    # INTERIOR and EXTERIOR ok
    if imag(rtr)<0 && abs(real(rtr))<1
    p[1] += 2im*pi
    p1 -= 2im*pi
    end

    for k = 1:npt
    p[k+1] = rtr*p[k] + c[k]
    end
    wcorrC = p[1:npt]
    =#
    ####
    
    zsc = rj
    a = ra
    b = rb
    ztg = r
    
    Ng=length(zsc);
    cc=(b-a)/2;
    ztgtr=(ztg-(b+a)/2)/cc;

    p=zeros(Complex{Float64},Ng+1,1);
    #
    gam=-1im
    loggam=-0.5*1im*pi # log(gam)
    if ifleft==1
        gam=1im;
        loggam=0.5*1im*pi;
    end
    p[1]=loggam + log((1-ztgtr)./(gam*(-1-ztgtr)));
    p[1] = log(1-ztgtr) - log(-1-ztgtr)
    p1 = log(1-ztgtr) + log(-1-ztgtr)
    # Signs here depend on how we have defined
    # direction of curve and normals
    # INTERIOR and EXTERIOR ok
    if imag(ztgtr)<0 && abs(real(ztgtr))<1
        p[1] += 2im*pi
        p1 -= 2im*pi
    end

    p111=log(abs((1-ztgtr).*(1+ztgtr)));
    
    p[2]=2+ztgtr*p[1];
    for k=1:Ng-1
        p[k+2]=( (2*k+1)*ztgtr*p[k+1]-k*p[k] )/(k+1);
    end

    wcorrC=p[1:end-1];

    return wcorrC
end


function wLinit(a::Complex{Float64},b::Complex{Float64},ztg::Complex{Float64},zsc::Vector{ComplexF64},ifleft::Int64)
    # *** ztgtr is target vector in transformed plane ***
    # added a new input argument "ifleft"
    # ifleft=1: the target is on the left side of the source panel
    # ifleft=0: the target is on the right side of the source panel
    # the curve is assumed to be counterclockwise oriented.
    # Thus, for a closed curve, ifleft=1 is equivalent to saying that
    # the target is in the interior domain, and ifleft=0 means that
    # the target is in the exterior domain.
    # But this technique works for open arcs as well, this is why we
    # use left/right instead of interior/exterior.
    
    # use Legendre polynomials instead of monomials
    # O(p) cost!!!
    Ng=length(zsc);
    cc=(b-a)/2;
    ztgtr=(ztg-(b+a)/2)/cc;

    p=zeros(Complex{Float64},Ng+1,1);
    q=zeros(Complex{Float64},Ng,1);
    #
    gam=-1im
    loggam=-0.5*1im*pi # log(gam)
    if ifleft==1
        gam=1im;
        loggam=0.5*1im*pi;
    end
    p[1]=loggam + log((1-ztgtr)./(gam*(-1-ztgtr))); 

    p111=log(abs((1-ztgtr).*(1+ztgtr)));
    
    p[2]=2+ztgtr*p[1];
    for k=1:Ng-1
        p[k+2]=( (2*k+1)*ztgtr*p[k+1]-k*p[k] )/(k+1);
    end
    q[1]=p111-2-ztgtr*p[1] # p111 only affects the value of q(1) if we use
    # Legendre polynomials instead of monomials!
    #    q(1)=real(q(1)); % one could just use the real part of q(1)
    for k=1:Ng-1
        q[k+1]=(p[k]-p[k+2])/(2*k+1);
    end
    
    wcorrL=q;

    return wcorrL
end
