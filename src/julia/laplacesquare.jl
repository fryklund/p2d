using bieps2d
using LinearAlgebra
using FastGaussQuadrature
using SparseArrays
import IterativeSolvers
include("helsingquad_laplace.jl")

function laplacesquare()
# ********************************************************
# *** Solves Laplace equation with basic RCIP inside a polygon    *
# ********************************************************
# *** User specified quantities ***************
lambda = -1/2;
nvertices = 4;
panels_per_edge = 5; # must have at least 5 panels per edge
boxlen = 0.5
theta = pi/4
# *********************************************

# Gauss-Legendre points and weights on the interval [-1,1]
T, W = gausslegendre(16)
nq = length(W);
npan = panels_per_edge * nvertices; # number of coarse panels
np = npan * nq;
nsub = [5, 10, 20];

# *** Pbc and PWbc prolongation matrices ***
IP, IPW = IPinit(T,W)
Pbc = Matrix(blockdiag(sparse(Diagonal(ones(16))),sparse(IP) ,sparse(IP),sparse(Diagonal(ones(16)))))
PWbc = Matrix(blockdiag(sparse(Diagonal(ones(16))),sparse(IPW) ,sparse(IPW),sparse(Diagonal(ones(16)))))	 


# determine "star" indices, i.e. those close to a corner
starind = Array{Any}(undef,nvertices)
for i = 1 : nvertices
    if i == 1
        starind[i] = vcat(np - 2*nq + 1: np, 1:2*nq)
    else
        start = (i - 1) * nq * panels_per_edge - 2 * nq + 1
        starind[i] = collect(start : start + 4 * nq - 1)
    end
end

# initialize geometry, Laplace DLP matrix and right hand side
x, y, xp, yp, tangent_x, tangent_y, curv, jac, thetas, s_coarse, panels_x, panels_y = geom_init(npan, T, W, nvertices, 0, boxlen,theta)
	
K = LaplaceKernelDL(x, y, tangent_x, tangent_y, curv, jac)
rhs = exact_solution(x,y);

# solve for actual density function on the coarse grid
rho_coarse = IterativeSolvers.gmres(lambda*I + K, rhs; reltol=eps())	 

# cretate K^\circ and K^*
Kcirc = deepcopy(K);
Kstar = zeros(size(K))
for i = 1 : nvertices
    Kcirc[starind[i], starind[i]] .= 0.0
    Kstar[starind[i], starind[i]] .= K[starind[i], starind[i]]
end
# *** Recursion for the R matrix ***

if !isempty(nsub)
    rho_hat = zeros(Float64,length(rho_coarse), length(nsub))
    for i = 1 : length(nsub)
        R=(1/lambda)*Matrix(Diagonal(ones(np)))
        Rblock = Rcomp(thetas,lambda,T,W,Pbc,PWbc,np,starind, false, nsub[i])
        
        for k = 1 : nvertices
            R[starind[k], starind[k]] .= Rblock[starind[k], starind[k]]
        end
        rho_tilde = IterativeSolvers.gmres(I + Kcirc * R,rhs; reltol=eps(),maxiter=np)	 

        rho_hat[:,i] = R * rho_tilde
    end
end

R=(1/lambda)*Matrix(Diagonal(ones(np)))
Rblock = Rcomp(thetas,lambda,T,W,Pbc,PWbc,np,starind,true,nsub)

for k = 1 : nvertices
    R[starind[k], starind[k]] = Rblock[starind[k], starind[k]]
end

# solve for transformed density on coarse grid 
rho_tilde = IterativeSolvers.gmres(I + Kcirc * R,rhs; reltol=eps(),maxiter=np)	 
	 
rho_hat_rec = R * rho_tilde

# reconstruct fine density
nsub_fine = 10
xf, yf, xpf, ypf, tangent_xf, tangent_yf, temp1, jacf, temp2, s_fine, panels_xf, panels_yf = geom_init(npan, T, W, nvertices, nsub_fine, boxlen,theta)
 
rho_fine = reconstruct_fine_density(lambda, thetas, Pbc, T, W, R, starind, nsub_fine, npan, rho_tilde)
## Post-processing 
     
# no subdivisions
M = 100;
x_target = range(-boxlen*0.99, boxlen*0.99, length = M)
y_target = x_target
X_target, Y_target = bieps2d.ndgrid(x_target, y_target)	 
        
u_dlp_refined = evaluateLaplaceDLP(X_target[:], Y_target[:], xf, yf, tangent_xf, tangent_yf, jacf, rho_fine)

u_dlp_refined = helsing_quad(u_dlp_refined, rho_fine, xf, yf, xpf, ypf, jacf, vcat(panels_xf, panels_xf[1]), vcat(panels_yf, panels_yf[1]), X_target[:], Y_target[:])

u_error = abs.((u_dlp_refined .- exact_solution(X_target[:], Y_target[:]))./exact_solution(X_target[:], Y_target[:]))

u_error[findall(u_error .== 0.0)] .= 1e-16;


pcolor(X_target, Y_target, reshape(log10.(u_error), size(X_target)));
#heatmap(x, y, reshape(log10.(u_error), size(X_target)))	 
return u_error, u_dlp_refined,x,y
end

###########################################################################	 
function LaplaceKernelDL(x_src, y_src, tangent_x, tangent_y, curv, jac)
# D = LaplaceDLmatrix(x, y, tangent_x, tangent_y, curv, jac), generate
# double-layer potential matrix for the Laplace equation

N = length(x_src)
normal_x = -tangent_y
normal_y = tangent_x

rx = x_src .- x_src'
ry = y_src .- y_src'
rdotn = rx.* repeat(normal_x,1,N)' .+ ry.*repeat(normal_y,1,N)'
rho2 = (rx.^2 + ry.^2)

D = rdotn .* repeat(jac,1,N)' ./ rho2
D[1:N+1:N^2] .= -0.5 * curv .* jac

D = D /(2*pi);
return D
end # LaplaceDLmatrix

###########################################################################	 
function evaluateLaplaceDLP(x_target, y_target, x_src, y_src,tangent_x, tangent_y, jac, rho)

M = length(x_target)

normal_x = -tangent_y
normal_y = tangent_x

rx = x_target .- x_src'	 
ry = y_target .- y_src'
rdotn = rx.* repeat(normal_x,1,M)' .+ ry.*repeat(normal_y,1,M)'
rho2 = (rx.^2 + ry.^2)

	 
D = rdotn .* repeat(jac,1,M)' ./ rho2

u = D*rho/(2*pi);

end

###########################################################################	 
function Rcomp(thetas, lambda, T, W, Pbc, PWbc, np, starind, recursion, nsub)

R = zeros(np,np)
TOL = 1e-14
inner_indices = 17 : 80

for k = 1 : length(thetas)
    
    if recursion
        # compute R using fixed point iteration
        x, y, tangent_x, tangent_y, curv, jac = geom_wedge_init(T, W,thetas[k])
        
        K = LaplaceKernelDL(x, y, tangent_x, tangent_y, curv, jac)
        
        R0 = PWbc' * ((lambda * I + K) \ Pbc)
        
        K[inner_indices, inner_indices] .= 0
        diff = 1
        iter = 0
		  R1  = R0
        while diff > TOL
            
            FR = (1/lambda)*Matrix(Diagonal(ones(96)))
            FR[inner_indices, inner_indices] .= R0
            R1 = PWbc' * ((inv(FR) + K) \ Pbc)
            
            diff = norm(R1 - R0, 2)
            R0 = R1
            
            iter = iter + 1
        end
        
#        disp(['corner ', num2str(k), ' of ', num2str(length(thetas)), ': ', ...
        #    num2str(iter), ' iterations']);
        
        R[starind[k], starind[k]] .= R1
        
    else # compute R normally, except that the geometry and K are scale invariant
        x, y, tangent_x, tangent_y, curv, jac = geom_wedge_init(T, W, thetas[k]);
        
        K = LaplaceKernelDL(x, y, tangent_x, tangent_y, curv, jac);
        
        R0 = PWbc' *((lambda * I + K) \ Pbc)
        
        K[inner_indices, inner_indices] .= 0.0
        FR = (1/lambda)*Matrix(Diagonal(ones(96)))
        
        for i = 1 : nsub + 1
            FR[inner_indices, inner_indices] .= R0
            R0 = PWbc' * ((inv(FR) + K) \ Pbc)
        end
        
        R[starind[k], starind[k]] .= R0
    end
end
return R
end

###########################################################################

function geom_init(npan, T, W, nvertices, nsub, boxlen,theta, center)


# L shaped domain
xv = [0, 2, 2, 1, 1, 0];
yv = [0, 0, 1, 1, 2, 2];

# Square domain
xvo = [center[1]-boxlen/2.0, center[1]+boxlen/2.0, center[1]+boxlen/2.0, center[1]-boxlen/2.0]
yvo = [center[2]-boxlen/2.0, center[2]-boxlen/2.0, center[2]+boxlen/2.0, center[2]+boxlen/2.0]


	 
xv = xvo.*cos(theta) .- yvo.*sin(theta)
yv = yvo.*cos(theta) .+ xvo.*sin(theta)	 
# reorder the points to prevent edges from overlapping
# meanx = mean(xv);
# meany = mean(yv);
# angles = atan2( (yv-meany),(xv-meanx));
# [~, sortIndices] = sort(angles);
# xv = xv(sortIndices);
# yv = yv(sortIndices);

if nsub == 0
    panels_per_edge = div(npan , nvertices)
else
    panels_per_edge = div(npan , nvertices) + 2 * nsub
end

x = Vector{Float64}(undef,0)
y = Vector{Float64}(undef,0)
xp = Vector{Float64}(undef,0)
yp = Vector{Float64}(undef,0)
tangent_x = Vector{Float64}(undef,0)
tangent_y = Vector{Float64}(undef,0)
curv = Vector{Float64}(undef,0)
jac = Vector{Float64}(undef,0)
s = Vector{Float64}(undef,0)
thetas = zeros(Float64,nvertices, 1)
edge_lengths = zeros(Float64,nvertices,1)
panels_x = Vector{Float64}(undef,0)
panels_y = Vector{Float64}(undef,0)

# compute the lengths of all the edges
for i = 1 : nvertices
    if i < nvertices
        next_index = i + 1
    else
        next_index = 1
    end
    edge_lengths[i] = sqrt((xv[i] - xv[next_index])^2 + (yv[i] - yv[next_index])^2)
end

minimum_edge = minimum(edge_lengths)
if nsub ==  0
    corner_panel_length = minimum_edge/panels_per_edge
else
    corner_panel_length = minimum_edge/(panels_per_edge - 2*nsub)
end

# parameterize each edge
for i = 1 : nvertices
    
    if i < nvertices
        next_index = i + 1
    else
        next_index = 1
    end
    
    if i == 1
        previous_index = nvertices
    else
        previous_index = i - 1
    end
   
    # compute interior angle
    v1x = xv[previous_index] - xv[i]
    v1y = yv[previous_index] - yv[i]    
    theta1 = atan(v1y, v1x)
    
    v2x = xv[next_index] - xv[i]
    v2y = yv[next_index] - yv[i] 
    theta2 = atan(v2y, v2x)
    #thetas(i) = wrapTo2Pi(theta1 - theta2)
    thetas[i] = atan(sin(theta1 - theta2),cos(theta1 - theta2))             
    if thetas[i]< 0
        thetas[i] = abs(thetas[i]) + 2 * (pi - abs(thetas[i]))
    end
    
    s_start = 0
    s_end = edge_lengths[i]
    
    for k = 1 : panels_per_edge
        
        panels_x = vcat(panels_x, (xv[i] * (s_end - s_start) + xv[next_index] * s_start) / s_end)
        panels_y = vcat(panels_y, (yv[i] * (s_end - s_start) + yv[next_index] * s_start) / s_end)
        
        # first and last two coarse panels on each edge must be the same size
        d_tmp = corner_panel_length #% / edge_lengths(i);
        if nsub == 0
           corner_panels = [1, 2, panels_per_edge-1, panels_per_edge]
        else
            corner_panels = vcat(collect(1 : 2 + nsub), collect(panels_per_edge - nsub - 1 : panels_per_edge))
        end
        
        if maximum(k .== corner_panels)
            
            if nsub == 0
                panel_length = d_tmp
                s_panel = T * panel_length / 2 .+ s_start .+ panel_length / 2
                s_start = s_start + panel_length
                
            else
                if k == 1 || k == 2
                    panel_length = d_tmp / (2 ^ nsub)
                else
						  if k < panels_per_edge / 2
                    panel_length = d_tmp / (2 ^ (nsub + 2 - k))
                    
                    else
								if k == panels_per_edge || k == panels_per_edge-1
                            panel_length = d_tmp / (2 ^ nsub)
                        else
                           panel_length = d_tmp / (2 ^ (nsub + 2 - panels_per_edge + k - 1))
                        end
                    end
                end
                
                s_panel = T * panel_length / 2 .+ s_start .+ panel_length / 2
                s_start = s_start + panel_length
            end
        else
            
            if nsub == 0
                panel_length = (s_end - 4*d_tmp) / (panels_per_edge - 4)
            else
                panel_length = (s_end - 4*d_tmp) / (panels_per_edge - 2 * (nsub + 2))
            end
            s_panel = T * panel_length / 2 .+ s_start .+ panel_length / 2
            s_start = s_start .+ panel_length
        end

        Wcoa = W * panel_length / 2
        
        # each edge is parameterized from s=0 to s = edge length
        x_panel = (xv[i] * (s_end .- s_panel) + xv[next_index] * s_panel) / s_end
        y_panel = (yv[i] * (s_end .- s_panel) + yv[next_index] * s_panel) / s_end
        
        xp_panel = (ones(size(s_panel)) * xv[next_index] .- xv[i]) / s_end;
        yp_panel = (ones(size(s_panel)) * yv[next_index] .- yv[i]) / s_end;

        x = append!(x, x_panel)
        y = append!(y, y_panel)
        xp = append!(xp, xp_panel)
        yp = append!(yp, yp_panel)
        
        xpp_panel = zeros(size(yp_panel))
        ypp_panel = zeros(size(yp_panel))
        
        jac_panel = sqrt.(xp_panel.^2 + yp_panel.^2)
    
        tangent_x = append!(tangent_x, -xp_panel ./ jac_panel)
        tangent_y = append!(tangent_y, -yp_panel ./ jac_panel)
    
        curv = append!(curv, (xp_panel .* ypp_panel .- yp_panel .* xpp_panel)./ jac_panel.^3)
    
        # include quadrature weights in Jacobian
        jac = append!(jac, jac_panel .* Wcoa)   
        
        s = append!(s, s_panel)
    end
end
return x, y, xp, yp, tangent_x, tangent_y, curv, jac, thetas, s, panels_x, panels_y    

end

###########################################################################	 
function  geom_wedge_init(T,W, theta)

s = vcat(T/2 .-1.5, T/4 .-0.75, T/4 .-0.25, T/4 .+0.25, T/4 .+0.75, T/2 .+1.5)
W = vcat(W/2, W/4, W/4, W/4, W/4, W/2)

x = zeros(size(s))
y = zeros(size(s))
xp = zeros(size(s))
yp = zeros(size(s))

xv = vcat(cos.(theta), 0, 1)
yv = vcat(sin.(theta), 0, 0)

# the entire wedge is parameterized from s=-2 to s = 2
x[1 : floor(Int64,end/2)] .= -0.5 * (xv[2] * (-2 .- s[1 : floor(Int64,end/2)]) .+ xv[1] * s[1 : floor(Int64,end/2)])
y[1 : floor(Int64,end/2)] .= -0.5 * (yv[2] * (-2 .- s[1 : floor(Int64,end/2)]) .+ yv[1] * s[1 : floor(Int64,end/2)])

xp[1 : floor(Int64,end/2)] .= -0.5 * (xv[1] - xv[2])
yp[1 : floor(Int64,end/2)] .= -0.5 * (yv[1] - yv[2])


x[floor(Int64,end/2) + 1 : end] .= 0.5 * (xv[2] * (2 .- s[floor(Int64,end/2) + 1:end]) .+ xv[3] .* s[floor(Int64,end/2) + 1:end])
y[floor(Int64,end/2) + 1 : end] .= 0.5 * (yv[2] * (2 .- s[floor(Int64,end/2) + 1:end]) + yv[3] .* s[floor(Int64,end/2) + 1:end])

xp[floor(Int64,end/2) + 1 : end] .= 0.5 * (xv[3] - xv[2])
yp[floor(Int64,end/2) + 1 : end] .= 0.5 * (yv[3] - yv[2])

xpp = zeros(length(W),1)
ypp = zeros(length(W),1)

jac = sqrt.(xp.^2 + yp.^2)

tangent_x = .-xp ./ jac
tangent_y = .-yp ./ jac

curv = (xp .* ypp .- yp .* xpp) ./ jac.^3

jac = jac .* W

return x, y, tangent_x, tangent_y, curv, jac	 
end

##########################################################################
function reconstruct_fine_density(lambda, thetas, Pbc, T, W, R, starind, nsub, npan, rho_tilde)

nvertices = length(thetas)

rho_fine = zeros(length(rho_tilde) + 2 * nsub * 16 * nvertices, 1)
inner_indices = 17:80
npan = div(npan, nvertices)

for k = 1 : nvertices
    
    Rblock = R[starind[k],starind[k]]
    rho_tilde_c = rho_tilde[starind[k]]
    
    x, y, tangent_x, tangent_y, curv, jac = geom_wedge_init(T, W, thetas[k])
  
    Kcirc = LaplaceKernelDL(x, y, tangent_x, tangent_y, curv, jac) 
    Kcirc[inner_indices, inner_indices] .= 0

    FR = lambda * I + Kcirc;
    FR[inner_indices, inner_indices] .= inv(Rblock)
    C = (I - Kcirc * inv(FR) ) *  Pbc
    
    rho_fine_left = Vector{Float64}(undef,0)
    rho_fine_right = Vector{Float64}(undef,0)
    
    for i = 1 : nsub
        rho_vec_b = C * rho_tilde_c
       
        rho_tilde_c = rho_vec_b[17:80]
        
        rho_fine_left = append!(rho_fine_left, rho_vec_b[1:16] / lambda)
        rho_fine_right = prepend!(rho_fine_right,rho_vec_b[81:96] / lambda)  
        
    end

    rho_fine_c = Rblock * rho_tilde_c
    
    rho_fine_left = append!(rho_fine_left, rho_fine_c[1 : floor(Int64,end/2)])
    rho_fine_right = prepend!(rho_fine_right,rho_fine_c[floor(Int64,end/2) + 1 : end])
    
    # place into correct indices
    starind_left = starind[k][1 : floor(Int64,end/2)]
    starind_right = starind[k][floor(Int64,end/2)+1:end]
    
    # read off density on Gamma^\circ
    start_indice = (k - 1) * (16 * (2*(nsub + 2) + npan - 4)) + length(rho_fine_left) + 1
	 rho_fine[start_indice : start_indice + 16 * (npan - 4) - 1] .= rho_tilde[starind_right[end] + 1 : starind_right[end] + 16 * (npan - 4)] / lambda
    

    # add right corner
    rho_fine[start_indice - length(rho_fine_right) : start_indice - 1] .= rho_fine_right;
    
    # add left corner
    if k > 1
        start_indice = start_indice - 2 * length(rho_fine_left)   
    else
        start_indice = length(rho_fine) - length(rho_fine_left) + 1
    end
    
    rho_fine[start_indice : start_indice + length(rho_fine_left) - 1] .= rho_fine_left
    
end
return rho_fine
end

##########################################################################
function exact_solution(x, y)
u_exact = x.^2 - y.^2 .+ 1;
end

##########################################################################

function IPinit(T,W)
# *** create prolongation matrices ***
A = ones(16,16)
AA = ones(32,16)
T2 = vcat(T .- 1,T .+ 1) ./ 2
W2 = vcat(W,W) ./ 2
for k = 2 : 16
    A[:,k] .= A[:,k-1] .* T
    AA[:,k] = AA[:,k-1] .* T2
end
IP = AA / A
IPW = IP .* (W2 * ( 1 ./ W)')
return IP,IPW
end
##########################################################################
function helsing_quad(u, rho, x, y, xp, yp, jac, panels_x, panels_y, x_target, y_target)
# Originally written by Sara Palsson, modified by Lukas Bystricky

X16, W16 = gausslegendre(16)
X32, W32 = gausslegendre(32)

IP1 = [0.708233680592340
   0.417460345661405
   0.120130621067993
  -0.023738421449022
  -0.024711509929699
   0.008767765249452
   0.010348492268217
  -0.004882937574295
  -0.005817138599704
   0.003371610049761
   0.003906389536232
  -0.002688491450644
  -0.002982620437980
   0.002396613365728
   0.002527899637498
  -0.002352088012027
  -0.353398976603946
   0.128914102924785
   0.490438143959320
   0.447688855448316
   0.159581518743026
  -0.043199604388389
  -0.045054276136530
   0.019843278050097
   0.022653307118776
  -0.012766166617303
  -0.014509184948602
   0.009852414829341
   0.010828396616511
  -0.008645914067315
  -0.009083716164910
   0.008436024529321
   0.254788814089210
  -0.079438651130554
  -0.180021675391748
   0.112706994698110
   0.452990321322730
   0.457103093490146
   0.172045630255837
  -0.057268690116309
  -0.057207007970563
   0.029864275385776
   0.032348060459762
  -0.021279595795857
  -0.022894435002697
   0.018024807626862
   0.018775629092083
  -0.017365908001573
  -0.192642764476253
   0.057539598371389
   0.118550200216509
  -0.060558741914225
  -0.137550413480163
   0.110511198044240
   0.438506321611140
   0.462013534571532
   0.176750576258897
  -0.070000077178178
  -0.066362764802900
   0.040455161093420
   0.041534393253997
  -0.031758507171589
  -0.032516100034128
   0.029834063376996
   0.143303354000051
  -0.042055381151416
  -0.083623389175777
   0.040136114296907
   0.081653505573269
  -0.052756328748486
  -0.114499477585301
   0.111212881525809
   0.430368039226389
   0.466243258020893
   0.179333130796333
  -0.083730103755132
  -0.075715752328325
   0.054025580681878
   0.053262695336761
  -0.048049159001486
  -0.099719237840432
   0.029015527255332
   0.056745671351501
  -0.026507281530925
  -0.051684172094834
   0.031210872338250
   0.060395080215009
  -0.046871396211720
  -0.096541871337564
   0.111392998543353
   0.422933391767461
   0.472656340478166
   0.183945672706357
  -0.102439013196790
  -0.090682596162259
   0.078322264361212
   0.058956626916838
  -0.017080869741123
  -0.033131760949494
   0.015276769255016
   0.029215868945318
  -0.017147768479290
  -0.031798751860556
   0.023087392278153
   0.042463097189794
  -0.039147738182975
  -0.077430953866654
   0.106967836431448
   0.409850258260916
   0.488757091888883
   0.202162218471372
  -0.145558364176374
  -0.019521496677808
   0.005645327810182
   0.010912188921697
  -0.005004288804177
  -0.009495119079648
   0.005510772494078
   0.010056981232185
  -0.007134062523268
  -0.012669001886025
   0.011041839978672
   0.019781931058369
  -0.022233561830741
  -0.044565913068780
   0.079639340872343
   0.355553969823584
   0.596733166923931]
	 
IP2 =  [0.713862126485003
   0.415861464999546
   0.117139053388567
  -0.022430941452515
  -0.022386727521234
   0.007526833242539
   0.008309785375195
  -0.003613499292735
  -0.003898339148532
   0.002002775905536
   0.002001361056109
  -0.001144934539207
  -0.001000441823818
   0.000579622749829
   0.000369122977751
  -0.000114841089527
  -0.373111737631011
   0.134514697868894
   0.500919671211399
   0.443106180943123
   0.151429254240996
  -0.038845347375560
  -0.037895234846542
   0.015381407522509
   0.015901484842618
  -0.007943124788689
  -0.007786257681468
   0.004394915660383
   0.003804467366061
  -0.002190252675426
  -0.001389346807558
   0.000431437040723
   0.293533407753495
  -0.090449199150776
  -0.200637543016583
   0.121726728257118
   0.469050569386606
   0.448514982681450
   0.157904965796408
  -0.048439925399146
  -0.043818636110253
   0.020276192880273
   0.018942511378929
  -0.010357973256273
  -0.008777345506291
   0.004982616916114
   0.003133611585860
  -0.000969126893520
  -0.254321612548312
   0.075074608907870
   0.151405998292083
  -0.074948908347047
  -0.163209724781879
   0.124257459362683
   0.461191598958519
   0.447810530211124
   0.155140021647846
  -0.054461091185942
  -0.044531484146797
   0.022565176432982
   0.018247128138763
  -0.010060054357776
  -0.006218741515135
   0.001907870729615
   0.231294304516010
  -0.067085064623796
  -0.130570952180064
   0.060729794130518
   0.118450523305909
  -0.072521831781564
  -0.147226860414845
   0.131787043059898
   0.461828826830738
   0.443484454902563
   0.147123228142680
  -0.057098466541441
  -0.040667821628640
   0.020922699023661
   0.012453895329684
  -0.003756646627295
  -0.217123906984604
   0.062438843010710
   0.119528551263843
  -0.054106792084372
  -0.101143929931585
   0.057878893205722
   0.104762346987468
  -0.074928255602672
  -0.139758055668850
   0.142936730014770
   0.468072149819296
   0.434818901723193
   0.133282875813318
  -0.053518478918945
  -0.028603957732716
   0.008260758005463
   0.208787542905641
  -0.059782988522232
  -0.113508058889059
   0.050717913029473
   0.092991730208347
  -0.051720793838675
  -0.089713332858577
   0.060028276446192
   0.099980675207689
  -0.081702601145486
  -0.139379433799503
   0.160051371016460
   0.483006808606042
   0.415312218104070
   0.103715923253338
  -0.024969801606650
  -0.204900209448852
   0.058561762926483
   0.110802967056481
  -0.049241305347759
  -0.089574268926796
   0.049263741070694
   0.084095332698592
  -0.054976265993162
  -0.088410558594981
   0.068301146401525
   0.105538302999532
  -0.098599012554463
  -0.155663999454775
   0.200570302178176
   0.540640169981230
   0.303399903673815]
	 
u_spec = u
z = x .+ 1im * y
zp = xp .+ 1im*yp
z_target = x_target .+ 1im*y_target

panels_z = panels_x .+ 1im*panels_y
npan = length(panels_z) - 1

for i = 1:length(z_target) #loop through all target points
    oldsum_total = 0
    for k = 1:npan
        
        panel_mid = (panels_z[k+1] + panels_z[k])/2
        panel_length = panels_z[k+1] - panels_z[k]
            
       if abs(z_target[i]-panel_mid) < abs(panel_length) #Check if z too close to any panel
            tz = zeros(16,1)
				tzp = zeros(16,1)
				tjac = zeros(16,1)
				trho = zeros(16,1)
            taupan = zeros(16,1)
            orig32 = zeros(32,1)
				tW32 = zeros(32,1)
            
            #  Calculate tau0
            tau0 = 2*(z_target[i]-panel_mid)/panel_length
            
            lg1 = log(1-tau0)
            lg2 = log(-1-tau0)
            
            j = collect(1:16)
            indj = (k-1)*16 .+ j
            tz = z[indj]
            
            panel_mid = (panels_z[k+1] + panels_z[k])/2
            panel_length = panels_z[k+1] - panels_z[k]
            taupan = 2*(tz .- panel_mid) / panel_length  #map all nodes on the panel to [-1,1]
            
            #Check if the point nz is between the panel and the real axis
            if (real(tau0) > -1) & (real(tau0) < 1)
                if imag(tau0) > 0 #above real axis, check if enclosed by axis and panel
                    furthercheck = false
                    for j=1:16 #go through all nodes on panel
                        if imag(taupan[j]) > imag(tau0)
                            furthercheck = true
                            break
                        end
                    end
                    
                    # if real(tau0) is below at least one of the
                    # quadrature nodes
                    if furthercheck
                        
                        #15 degree interpolation at real(tau0)
                        tmpT = real(taupan)
                        tmpb = imag(taupan)
                        
                        p = vandernewtonT(tmpT,tmpb,16)
                        
                        kk = collect(0:15)
                        test = sum(p.*real(tau0).^kk)
                        
                        if test > imag(tau0)
                            #tau0 is enclosed by panel and real axis
                            lg1 = lg1 - pi*1im
                            lg2 = lg2 + pi*1im
                        end
                    end
                else
						  if imag(tau0) < 0 #below the real axis, check enclosed
                        furthercheck = false
                        for j=1:16 #go through all nodes on panel
                            if imag(taupan[j]) < imag(tau0)
                                furthercheck = true
                                break
                            end
                        end
                        
                        # if real(tau0) is above at least one of the
                        # quadrature nodes
                        if furthercheck
                            
                            #16 degree interpolation at real(tau0)
                            tmpT = real.(taupan)
                            tmpb = imag.(taupan)
                            
                            p = vandernewtonT(tmpT,tmpb,16)
                            
                            kk = collect(0:15)
                            test = sum(p.*real(tau0).^kk)
                            
                            if test < imag(tau0)
                                #tau0 is enclosed by panel and real axis
                                lg1 = lg1 + pi*1im
                                lg2 = lg2 - pi*1im
                            end
                        end
                    end
                end
            end
            
            p32 = zeros(Complex{Float64},32,1)
            p32[1] = lg1-lg2
            # Calculate old contribution to u from panel
            
            tzp = zp[indj]
            trho = rho[indj]
            tjac = jac[indj]
            txp = xp[indj]
            typ = yp[indj]
            
            # original 16 point contribution to u from panel
            oldsum = -1/(2*pi)*sum(tjac.*trho.*imag.(tzp./(tz .- z_target[i]))./sqrt.(txp.^2 + typ.^2))
            oldsum_total = oldsum_total + oldsum
            
            #16 point quadrature of 1/(z - z_target) over panel
            testsum = sum(tjac.*tzp./(tz .- z_target[i])./sqrt.(txp.^2 + typ.^2))
            
            if abs(p32[1]-testsum) > 1e-13 #Standard 16-GL not good enough!
                # Interpolate to 32-point GL quadrature
                trho32 = IPmultR(trho,IP1,IP2)
                tz32 = IPmultR(tz,IP1,IP2)
                tzp32 = IPmultR(tzp,IP1,IP2)
                plen = tjac[1] / W16[1]
                txp32 = IPmultR(txp,IP1,IP2)
                typ32 = IPmultR(typ,IP1,IP2)
                tjac32 = W32.*plen
                
                #32 point quadrature of 1/(z - z_target) over panel
                testsum32 = sum(tjac32.*tzp32./(tz32 .- z_target[i])./sqrt.(txp32.^2 + typ32.^2))
                
                if abs(p32[1]-testsum32) < 1e-13 # 32 GL suffices!
                    
                    newsum = -1/(2*pi)*sum(tjac32.*trho32.* imag(tzp32./(tz32 .- z_target[i]))./sqrt.(txp32.^2 + typ32.^2))
                    
                    u_spec[i] = u_spec[i] + (newsum-oldsum)
                else #32 GL not enough, use interpolatory quadrature instead
                    # Use interpolatory quadrature
                    
                    taupan32 = IPmultR(taupan,IP1,IP2)
                    
                    if (minimum(imag.(taupan32)) < 0) & (maximum(imag.(taupan32)) > 0)
                        taupan32 = real(taupan32)
                    end
                        
                    signc = -1
                    for j=1:31 #Calculate pk
                        p32[j+1] = tau0*p32[j] + (1-signc)/(j)
                        signc = -signc
                    end
                    
                    p32coeff = vandernewton(taupan32,p32,32)
                    newsum2 = -1/(2*pi)*sum(imag(p32coeff.*trho32))
                    
                    u_spec[i] = u_spec[i] + (newsum2 - oldsum)
                    
                end
            end
        end
    end
end
return u_spec
end

##########################################################################
function vandernewton(T,b,n)

for k=1:n-1
    for i=n:-1:k+1
        b[i] = b[i] - T[k]*b[i-1]
    end
end

for k=n-1:-1:1
    for i=k+1:n
        b[i] = b[i] / (T[i] - T[i-k])
    end
    for i=k:n-1
        b[i] = b[i] - b[i+1]
    end
end
return b	 
end

##########################################################################
function vandernewtonT(T,b,n)
x = T
c = b
for k=1:n-1
    for i=n:-1:k+1
        c[i] = (c[i]-c[i-1])/(x[i]-x[i-k])
    end
end
a = c
for k=n-1:-1:1
    for i=k:n-1
        a[i] = a[i]-x[k]*a[i+1]
    end
end
return a
end

##########################################################################
function IPmultR(f16,IP1,IP2)
f32 = zeros(Complex{Float64},32,1)
for i=1:16
   t1 = 0.0
   t2 = 0.0
   ptr = i
   for j=1:8
      t1 = t1 + IP1[ptr] * (f16[j]+f16[17-j]) #17
      t2 = t2 + IP2[ptr] * (f16[j]-f16[17-j]) #17
      ptr = ptr + 16
   end
   f32[i] = t1+t2
   f32[33-i] = t1-t2 #33
end
return f32
end