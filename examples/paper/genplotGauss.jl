using LinearAlgebra
using MATLAB
using rbfqr
include("../../dev/chebyshev2dinterpolation.jl")
include("../../dev/chebexps.jl")
include("../../dev/legetens.jl")
include("../../dev/volumetree_pux.jl")
include("../../dev/haltonbox.jl")

function setCheb(order)
    if order == 0
	T = (x) -> ones(size(x))
    elseif order == 1
	T = (x) -> x
    elseif order == 2
	T = (x) -> 2*x.^2 .- 1
    elseif order == 3
	T = (x) -> 4*x.^3 .- 3*x
    elseif order == 4
	T = (x) -> 8*x.^4 .- 8*x.^2 .+ 1
    elseif order == 5
	T = (x) -> 16*x.^5 .- 20*x.^3 .+ 5*x
    elseif order == 6
	T = (x) -> 32*x.^6 .- 48*x.^4 .+ 18*x.^2 .- 1
    elseif order == 7
	T = (x) -> 64*x.^7 .- 112*x.^5 .+ 56*x.^3 .- 7*x
    elseif order == 8
	T = (x) -> 128*x.^8 .- 256*x.^6 .+ 160*x.^4 .- 32*x.^2 .+ 1
    elseif order == 9
	T = (x) -> 256*x.^9 .- 576*x.^7 .+ 432*x.^5 .- 120*x.^3 .+ 9*x
    elseif order == 10
	T = (x) -> 512*x.^10 .- 1280*x.^8 .+ 1120*x.^6 .- 400*x.^4 .+ 50*x.^2 .- 1
    elseif order == 11
	T = (x) -> 1024*x.^11 .- 2816*x.^9 .+ 2816*x.^7 .- 1232*x.^5 .+ 220*x.^3 .- 11*x
    end
    return T
end

function distmesh(ax,bx,ay,by,eta)
    mat"addpath('src/external/distmesh')"
    mat"$p= distmesh2d(@dpoly,@huniform, $eta,[-1,-1; 1,1],[1 1; -1 1; -1 -1; 1 -1; 1 1],[1 1; -1 1; -1 -1; 1 -1; 1 1]);"
    p = hcat(p[:,1]*(bx - ax)/2 .+ (bx + ax) / 2, p[:,2] *(by - ay)/2 .+ (by + ay)/2)
    return p
end

boxlen = 2^(1)
# First kind
T0(x) = 1.0; 
T1(x) = x;
T2(x) = 2*x.^2 .- 1;
T3(x) = 4*x.^3 .- 3*x;
T4(x) = 8*x.^4 .- 8*x.^2 .+ 1
T5(x) = 16*x.^5 .- 20*x.^3 .+ 5*x
T6(x) = 32*x.^6 .- 48*x.^4 .+ 18*x.^2 .- 1
T7(x) = 64*x.^7 .- 112*x.^5 .+ 56*x.^3 .- 7*x

#--- Select a function
param = 1
#F2d(x,y) = exp(-(param*abs(x-y))^2)
#F2d(x,y) = T6(x/(boxlen/2)) .* T0(y/(boxlen/2))

# Number of lines to intersect B0
Nrand = 1000
boxlen0 = boxlen/3
v0 = -boxlen0/2 .+ rand(Nrand,4)*(boxlen0/2 + boxlen0/2)
#y0v = -boxlen0/2 .+ rand(Nrand)*(boxlen0/2 + boxlen0/2)
#x1v = -boxlen0/2 .+ rand(Nrand)*(boxlen0/2 + boxlen0/2)
#y1v = -boxlen0/2 .+ rand(Nrand)*(boxlen0/2 + boxlen0/2)

# Number of points to try for each line
Nrandeval = 1000
xrandeval = hcat((rand(Nrandeval) .- 0.5)*boxlen*0.95,(rand(Nrandeval) .- 0.5)*boxlen*0.95)

# Delauny triangulation
xkdens = 0.286
#xkdens = 0.5
ax,bx,ay,by = -boxlen/2,boxlen/2,-boxlen/2,boxlen/2
xkdel = distmesh(ax,bx,ay,by,xkdens)
nxkdel = size(xkdel,1)

# Distribution of Gaussians:

# Chebyshev tensor gird
norderrbf = 8
norderrbf2 = norderrbf
Bgridrbf = Matrix{Float64}(undef,2,norderrbf^2)
Bxqrbf = Vector{Float64}(undef,norderrbf)
Bwtsrbf = Vector{Float64}(undef,norderrbf)
Bumatrbf = Matrix{Float64}(undef,norderrbf,norderrbf)	
Bvmatrbf = Matrix{Float64}(undef,norderrbf,norderrbf)
Bityperbf = 2 # What type of calculation to be performed, see function chebexps
chebexps(Bityperbf,norderrbf,Bxqrbf,Bumatrbf,Bvmatrbf,Bwtsrbf)
Bxqrbf = Bxqrbf / (2/boxlen)
mesh2d(Bxqrbf,norderrbf,Bxqrbf,norderrbf,Bgridrbf)
xkB = hcat(vec(Bgridrbf[1,:]),vec(Bgridrbf[2,:]))
nxkB = size(xkB,1)

# Uniform distribution
norderuni = 8
Bgriduni = Matrix{Float64}(undef,2,norderuni^2)
Bxquni = Vector{Float64}(undef,norderuni)
Bwtsuni = Vector{Float64}(undef,norderuni)
Bumatuni = Matrix{Float64}(undef,norderuni,norderuni)	
Bvmatuni = Matrix{Float64}(undef,norderuni,norderuni)
Bitypeuni = 2 # What type of calculation to be performed, see function chebexps
Bxquni = collect(range(-1,1,norderuni))
Bxquni = Bxquni / (2/boxlen)
mesh2d(Bxquni,norderuni,Bxquni,norderuni,Bgriduni)
xkuni = hcat(vec(Bgriduni[1,:]),vec(Bgriduni[2,:]))
nxkuni = size(xkuni,1)

# Combination of unifrom and Delauny
idxx = findall(abs.(abs.(Bgriduni[1,:]) .- 1) .< 1e-14)
idxy = findall(abs.(abs.(Bgriduni[2,:]) .- 1) .< 1e-14)
idxuni = sort(union(idxx,idxy))
idxx = findall(abs.(abs.(xkdel[:,1]) .- 1) .< 1e-2)
idxy = findall(abs.(abs.(xkdel[:,2]) .- 1) .< 1e-2)
idxdel = setdiff(collect(1:nxkdel),sort(union(idxx,idxy)))
xkunion = vcat(copy(Bgriduni[:,idxuni])',xkdel[idxdel,:])

xkhalton = 2*(halton(round(Int64,26/pi*7.7),2) .- 0.5)
nxkhalton = size(xkhalton,1)

# Staggered grid

ug = deepcopy(Bgriduni);
h = abs(Bgriduni[1,1] - Bgriduni[1,2])
for i=1:2:(norderuni-1)
    idx = (i*norderuni+1):((i+1)*norderuni)
    ug[1,idx] .+= h/2
end

idx = findall(ug[1,:] .<= 1.0)
ug = ug[:,idx]

#xk = xkdel
#nxk = nxkdel

xk = copy(ug')
nxk = size(xk,1)

#xk = xkhalton
#nxk = nxkhalton

println(" #Gaussian basis functions = ", nxk)

norderrbf = 12
maxcheborder = norderrbf - 1

Bgrid = Matrix{Float64}(undef,2,norderrbf^2)
Bxq = Vector{Float64}(undef,norderrbf)
Bwts = Vector{Float64}(undef,norderrbf)
Bumat = Matrix{Float64}(undef,norderrbf,norderrbf)	
Bvmat = Matrix{Float64}(undef,norderrbf,norderrbf)
Bitype = 2 # What type of calculation to be performed, see function chebexps
chebexps(Bitype,norderrbf,Bxq,Bumat,Bvmat,Bwts)
Bxq = Bxq / (2/boxlen)
mesh2d(Bxq,norderrbf,Bxq,norderrbf,Bgrid)

norder = 8

B0grid = Matrix{Float64}(undef,2,norder^2)
B0xq = Vector{Float64}(undef,norder)
B0wts = Vector{Float64}(undef,norder)
B0umat = Matrix{Float64}(undef,norder,norder)	
B0vmat = Matrix{Float64}(undef,norder,norder)
B0itype = 2 # What type of calculation to be performed, see function chebexps
chebexps(B0itype,norder,B0xq,B0umat,B0vmat,B0wts)
B0xq = B0xq / (2/boxlen0)
mesh2d(B0xq,norder,B0xq,norder,B0grid)

# RBF-QR MATRIX
A =  rbfqr_diffmat_2d_fast(copy(Bgrid'), [0.0,0.0], boxlen * 1.00000000000001/sqrt(2), xk, 1e-5)[1]
Acomp =  rbfqr_diffmat_2d_fast(copy(hcat(Bgrid,B0grid)'), [0.0,0.0], boxlen * 1.00000000000001/sqrt(2), xk, 1e-5)[1]

II = ones(Float64,nxk)
II = Diagonal(II)

AcompII =  rbfqr_diffmat_2d_fast(copy(hcat(Bgrid,B0grid,xk')'), [0.0,0.0], boxlen * 1.00000000000001/sqrt(2), xk, 1e-5)[1]

compBgrid = hcat(Bgrid,B0grid)

maxerrorv = zeros(Nrand,Nrandeval)

for kk = 1:Nrand
    x0 = v0[kk,1]
    y0 = v0[kk,2]
    x1 = v0[kk,3]
    y1 = v0[kk,4]

    dx = x1 - x0
    dy = y1 - y0

    k = dy/dx

    m = y0 - k*x0

    normal = [-dy,dx]

    P0 = [x0[1],y0[1]]

    idxB0in = []
    idxB0out = []
    println("kk = ",kk)
    for n = 1:norder^2
	if dot(normal,B0grid[:,n] - P0) <= 0
	    push!(idxB0in,n)
	else
	    push!(idxB0out,n)
	end
    end
    if length(idxB0in)==0
	# continue
    end

    idxBin = []
    idxBout = []

    for n = 1:norderrbf^2
	if dot(normal,Bgrid[:,n] - P0) <= 0
	    push!(idxBin,n)
	else
	    push!(idxBout,n)
	end
    end



    idxxkin = []
    idxxkout = []

    for n = 1:nxk
	if dot(normal,xk[n,:] - P0) <= 0
	    push!(idxxkin,n)
	else
	    push!(idxxkout,n)
	end
    end


    idxcompBin = []
    idxcompBout = []

    for n = 1:(norder^2 + norderrbf^2)
	if dot(normal,compBgrid[:,n] - P0) <= 0
	    push!(idxcompBin,n)
	else
	    push!(idxcompBout,n)
	end
    end

    idxcompBuniin = []
    idxcompBuniout = []

    for n = 1:(norderuni^2)
	if dot(normal,Bgriduni[:,n] - P0) <= 0
	    push!(idxcompBuniin,n)
	else
	    push!(idxcompBuniout,n)
	end
    end	 

    
    II = zeros(Float64,nxk)
    II[idxxkin] .= 1.0
    II = Diagonal(II)
    II = II[idxxkin,:]

    idxxrandin = []
    for i = 1:Nrandeval
 	d = dot(normal,xrandeval[i,:] - P0)
        dist2xk = sqrt.((xrandeval[i,1].-xk[idxxkout,1]).^2 .+ (xrandeval[i,2].-xk[idxxkout,2]).^2)
        idx = findall(dist2xk .<= h*1.05)
	if (d <= -0.05) & (length(idx) == 0)  
	    push!(idxxrandin,i)	 
	end            
    end

    nidxrandin = length(idxxrandin)
    maxerrmn = -1
    for i = 1:nidxrandin
        println("i = ",i)
        for orderm = 0:7
            for ordern = 0:(7-orderm)
	        Tm = setCheb(orderm)
                Tn	= setCheb(ordern)
                F2d(x,y) = Tm(x / (boxlen/2)) .* Tn(y / (boxlen/2))
                Fxk = F2d.(xk[:,1],xk[:,2])
                FBgrid = F2d.(Bgrid[1,:],Bgrid[2,:])
                FBgriduni = F2d.(Bgriduni[1,:],Bgriduni[2,:])	 
                FcompBgrid = F2d.(compBgrid[1,:],compBgrid[2,:])

                CVand = 0.0
                C = AcompII[vcat(idxcompBin,norderrbf^2 .+ norder^2 .+ idxxkin),:] \ (vcat(FcompBgrid[idxcompBin],Fxk[idxxkin]) .- CVand)

                F = Acomp*C
                
                #F[idxBin] .= FBgrid[idxBin]	 
                F = F[1:norderrbf^2]

                
                FBcheb = copy(transpose(reverse(reshape(F,(norderrbf,norderrbf)))))

                chebcoefs = interpspec2D_DCT_compcoeff(norderrbf, norderrbf, FBcheb)
	        
	 	global err = abs(F2d(xrandeval[idxxrandin[i],1],xrandeval[idxxrandin[i],2]) - eval_func_2D(chebcoefs,xrandeval[idxxrandin[i],1],xrandeval[idxxrandin[i],2],norderrbf,-boxlen/2,boxlen/2,norderrbf,-boxlen/2,boxlen/2))
               
                if err > maxerrmn
                    maxerrmn = err
                    maxerrorv[kk,i] = err
                end
	    end

            
        end
	
    end
end

Plots.plot([0;Nrand+1],log10.([0.5e-10;0.5e-10]),lc=:red,legend = false,lw=5.0,ylim=(-16, 0))

for i=1:Nrand
    idx = findall(maxerrorv[:,i].>0)
    vals = maxerrorv[idx,i]
    p = Plots.scatter!(i*ones(length(idx)),log10.(vals), mc=:black, ms=2,legend = false)
    display(p)
end
