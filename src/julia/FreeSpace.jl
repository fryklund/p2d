__precompile__()
module FreeSpace

using FINUFFT
using FFTW
using SpecialFunctions

export fs_poisson
export per_poisson

# TODO: Add precomputation of kernel
## FREESPACE SOLVER WITH NUFFT, ASSUMES ndgrid
"Solves the Poisson equation, defined as Lu = f"
function fs_poisson(F::Matrix{Float64}, L::Float64, xnu=[], ynu=[])
    # Prep oversampled grid 
    sf = 1+sqrt(2) # Minimum oversampling
    M = size(F)
    @assert length(M)>1 && M[1]==M[2] "Grid must be square"
    N = M[1]
    Nf = Int(ceil(sf*N))
    if mod(Nf,2) != 0
	Nf+=1  #Even grids are always faster (right?)
    end
    sf = Nf/N # Effective oversampling
    # Pad and transform
    k1, k2 = k_vectors([Nf, Nf], [L*sf, L*sf])
    R = sqrt(2)*L
    logR = log(R)
    Fpad = zeros(Nf, Nf)
    Fpad[1:N, 1:N] = F
    Fhat = fftshift( fft(Fpad) )
    # Multiply with truncated greens function    
    Uhat = Fhat # compute in-place
    mid = div(Nf,2) + 1 # grid is even
    Fzero = Fhat[mid, mid]
    k1sq, k2sq = k1.^2, k2.^2
    for i = 1:Nf
	for j = 1:Nf
	    K2 = k1sq[i] + k2sq[j]
	    K = sqrt(K2)
	    J0 = besselj(0, K*R)
	    J1 = besselj(1, K*R)
	    GR = 2*pi/K2*(J0 + K*R*J1*logR - 1)
	    Uhat[i,j] *= GR
        end
    end
    # Correct at k=0
    Uhat[mid, mid] = Fzero*(pi*R^2*(-1 + 2*log(R)))/2
    Uhat *= 1/(2*pi)
    # Transform back and truncate
    U = ifft(ifftshift(Uhat))
    U = real( U[1:N,1:N] )
    
    if length(xnu)==0
	return U
    end
    # NUFFT for off-grid points
    @assert length(xnu)==length(ynu)
    origin = pi/sf
    scale = 2*pi/(sf*L)
    xn = origin .+ xnu*scale
    yn = origin .+ ynu*scale
    unu = complex(zeros(length(xnu)))
    nufft2d2!(xn, yn, unu, 1, 10*eps(), Uhat/(N*sf)^2)
    unu = real(unu)
    return U, unu
end

## PERIODIC SOLVER WITH NUFFT
function per_poisson(F, L, xnu=[], ynu=[])
    M = size(F)
    @assert length(M)>1 && M[1]==M[2] "Grid must be square (for now)"
    Nf = M[1]
    # Pad and transform
    k1, k2 = k_vectors([Nf, Nf], [L, L])
    Fhat = fftshift( fft(F) )
    # Multiply with greens function    
    Uhat = Fhat # compute in-place
    for i = 1:Nf
	for j = 1:Nf
	    K2 = k1[i]^2 + k2[j]^2
	    GR = K2==0 ? 0 : -1/K2            
	    Uhat[i,j] *= GR
        end
    end
    # Transform back and truncate
    U = real(ifft(ifftshift(Uhat)))
    
    if length(xnu)==0
	return U
    end
    # NUFFT for off-grid points
    @assert length(xnu)==length(ynu)
    scale = 2*pi/L
    # First rescale from [L/2,L/2] to [-pi,pi]    
    xnu *= scale
    ynu *= scale
    # Then swap [-pi,0] and [0,pi]
    xnu = -pi + mod(xnu, 2*pi)
    ynu = -pi + mod(ynu, 2*pi)

    unu = complex(zeros(length(xnu)))
    nufft2d2!(xnu, ynu, unu, 1, eps(), Uhat/Nf^2)
    unu = real(unu)
    return U, unu
end


## HELPERS
function ndgrid(x, y)
    nx = length(x)
    ny = length(y)
    X = zeros(nx, ny)
    Y = zeros(nx, ny)
    for i = 1:nx
	for j = 1:ny
	    X[i,j] = x[i]
	    Y[i,j] = y[j]
        end
    end
    return X,Y
end

function k_vectors(M,box)
    function k_vec(M,L)
	if mod(M,2)==0
	    MM = M/2;
	    k = (2*pi/L)*collect(-MM:(MM-1));
        elseif mod(M-1,2)==0
	    MM = (M-1)/2;
	    k = (2*pi/L)*collect(-MM:MM);
        else
	    error("k-vectors not computed");
        end
        return k
    end
    k1 = k_vec(M[1], box[1])
    k2 = k_vec(M[2], box[2])    
    return k1, k2
end

end # module
