using LinearAlgebra
using MATLAB
using rbfqr
using Plots
import Random

Random.seed!(0)

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
xrandeval = hcat((rand(Nrandeval) .- 0.5)*boxlen0*0.95,(rand(Nrandeval) .- 0.5)*boxlen0*0.95)

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

diagh = sqrt((xk[1,1]-xk[9,1])^2 + (xk[1,2]-xk[9,2])^2)

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

errorv = zeros(sum(1:maxcheborder+1),2)
mnv = zeros(sum(1:maxcheborder+1),2)
itr = 1
ik = 0
maxerrormat3 = zeros(Nrand,Nrandeval)
for orderm = 0:7
    for ordern = 0:(7-orderm)
	mnv[itr,:] .= [orderm,ordern]
        ERROR = -1
        ERRORL2 = -1
        Tm = setCheb(orderm)
        Tn	= setCheb(ordern)
        F2d(x,y) = Tm(x / (boxlen/2)) .* Tn(y / (boxlen/2))
        Fxk = F2d.(xk[:,1],xk[:,2])
        FBgrid = F2d.(Bgrid[1,:],Bgrid[2,:])
        FBgriduni = F2d.(Bgriduni[1,:],Bgriduni[2,:])	 
        FcompBgrid = F2d.(compBgrid[1,:],compBgrid[2,:])
	maxCerror = -1
	condmax = -1
	maxcoeff = -1
	maxinterperror = -1
	maxcheberror = -1
	
        for kk = 1:Nrand
            locmaxCerror = -1
            loccondmax = -1
            locmaxcoeff = -1
            locmaxinterperror = -1
            locmaxcheberror = -1
            # Define line y = kx + m
            #x0 = 0.0
            #y0 = -boxlen/3
            #x1 = boxlen/2
            #y1 = boxlen/2


            #x0 = x0v[kk]
            #y0 = y0v[kk]
            #x1 = x1v[kk]
            #y1 = y1v[kk]

            x0 = v0[kk,1]
            y0 = v0[kk,2]
            x1 = v0[kk,3]
            y1 = v0[kk,4]
            #=
	    x0 = -boxlen
	    x1 = boxlen 
	    y0 = boxlen * 1.1
	    y1 = boxlen * 1.1
            =#	 
            dx = x1 - x0
            dy = y1 - y0

            k = dy/dx

            m = y0 - k*x0

            normal = [-dy,dx]

            P0 = [x0[1],y0[1]]

            #--- The RBF-QR method is used for approximating a function and its 
            #--- first and second derivative (in one dimension)


     

            idxB0in = []
            idxB0out = []

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

            #println("# points inside = ", length(idxBin))

            #II = ones(Float64,nxk)
            II = zeros(Float64,nxk)
            II[idxxkin] .= 1.0
            II = Diagonal(II)
            II = II[idxxkin,:]

            #II = ones(Float64,length(idxxkin))
            #II = Diagonal(II)

            #Ain = A[idxBin,idxxkin]
            #Aout = A[idxBout,idxxkin]

            Ain = A[idxBin,:]
#            Aout = A[idxBout,:]

#            Aincomp = Acomp[idxcompBin,:]
            #Aoutcomp = Acomp[idxcompBout,:]	 
            #Ain = Aincomp

            F = zeros(norderrbf^2)

            AII = Matrix(vcat(Ain,II))
#            AIIcomp = Matrix(vcat(Aincomp,II))	 
#            AIIcomp = Matrix(vcat(A,II))

            #println("COND in", cond(Aincomp))
            #println("COND II", cond(AIIcomp))	 
            #Amap =  rbfqr_diffmat_2d_fast(copy(Bgrid'), [0.0,0.0], boxlen * 1.00000000000001/sqrt(2), xk[idxxkin,:], eps())[1]

#            nAC = 1

#            chebypoly_x = chebypoly.(Bgrid[1,:],-boxlen/2,boxlen/2,nAC)
#            chebypoly_y = chebypoly.(Bgrid[2,:],-boxlen/2,boxlen/2,nAC)
#=
            Chx = zeros(nAC,norderrbf^2)
            Chy = zeros(nAC,norderrbf^2)
            for k = 1:norderrbf^2
	        Chx[:,k] .= vec(chebypoly_x[k])
	        Chy[:,k] .= vec(chebypoly_y[k])
            end


            AC = zeros(norderrbf^2,nAC^2)
            order = nAC

            col = 1
            for i = 1:nAC
	        for j = 1:nAC
		    AC[:,col] .= (Chx[i,:]) .* (Chy[j,:])
                    #		global	col += 1
		    col += 1
	        end
            end
            Fdatacheb = F2d.(Bgrid[1,idxBin],Bgrid[2,idxBin])
            CVand = AC[idxBin,:]\Fdatacheb
            CVand = CVand * 0.0
            =#
            CVand = 0.0
            #M = vcat(hcat(A,AC),hcat(AC',zeros(nAC^2,nxk + nAC^2 - norderrbf^2)))

            #Min = vcat(hcat(Ain,AC[idxBin,:]),hcat(AC[idxBin,:]',zeros(nAC^2,nAC^2 + nxk - length(idxBin))))
            #FM = vcat(FBgrid[idxBin],zeros(nAC^2))
            #CM = Min \ FM

            #Mout = vcat(hcat(A[idxBout,:],AC[idxBout,:]),hcat(AC[idxBout,:]',zeros(norderrbf^2,norderrbf^2 + nxk - length(idxBout))))

            C = AII \ vcat(FBgrid[idxBin],Fxk[idxxkin])
            #C = AIIcomp \ (vcat(FcompBgrid[idxcompBin],Fxk[idxxkin]) .- CVand)
            #C = AcompII[vcat(idxcompBin,norderrbf^2 .+ norder^2 .+ idxxkin),:] \ (vcat(FcompBgrid[idxcompBin],Fxk[idxxkin]) .- CVand)

            #C = AIIcomp \ vcat(FcompBgrid,Fxk)
            #C = Aincomp \ FcompBgrid[idxcompBin]
            #C = Ain \ FBgrid[idxBin]
            #F[idxBout] .= Aout * (Ain\FBgrid[idxBin])
#            F = Acomp * C .+ CVand
            #F[idxBout] = Aoutcomp * C
            #F .= FBgrid
            #F[idxcompBin] .= FcompBgrid[idxcompBin]
            #F = Acomp*C
            #F = Acomp * Fxk
            F = A * C
            #F[idxBin] .= FBgrid[idxBin]	 
            F = F[1:norderrbf^2]

            #IIuni = zeros(Float64,128)
            #IIuni[idxBin] .= 1.0
            #IIuni = Diagonal(IIuni)
            #iIIuni = IIuni[idxBin,:]
	    
            #Cga = Matrix(vcat(Aga[idxBin,:],IIuni)) \ vcat(FBgrid[idxBin],FBgriduni[idxcompBuniin])
	    
            #F = Aga * Cga
            #C[idxxkin] .= Fxk[idxxkin]
            #println("Interp error ", norm(Aincomp * C - FcompBgrid[idxcompBin]))
            Aincomp = Acomp[idxcompBin,:]
            locmaxinterperror = norm(Aincomp * C .+ CVand- FcompBgrid[idxcompBin]	 )
            #println("Coeff error ", norm(C[idxxkin] - Fxk[idxxkin]))
	    normcoefferr = norm(C[idxxkin] - Fxk[idxxkin])
	    condAI = cond(AcompII)
            #	 println("COND AI = ", condAI)
            if	 locmaxCerror < 	 normcoefferr
	        locmaxCerror =  normcoefferr
            end
            if	 loccondmax < 	 condAI
	        loccondmax =  condAI
            end

            if maxinterperror < locmaxinterperror
	        maxinterperror = locmaxinterperror
            end
            #println("Coeff error ", norm(C - Fxk[idxxkin]))
            #F = M * CM
            #F = F[1:norderrbf^2]
            #F[idxBin] = (Min * CM)[1:length(idxBin)]
            #=
            nAC = 2

            XY = hcat(compBgrid,copy(xk'))

            idxXYin = []
            idxXYout = []

            for n = 1:(norder^2 + norderrbf^2 + nxk)
	    if dot(normal,XY[:,n] - P0) <= 0
	    push!(idxXYin,n)
	    else
	    push!(idxXYout,n)
	    end
            end

            chebypoly_x = chebypoly.(XY[1,:],-boxlen/2,boxlen/2,nAC)
            chebypoly_y = chebypoly.(XY[2,:],-boxlen/2,boxlen/2,nAC)

            Chx = zeros(nAC,norder^2 + norderrbf^2 + nxk)
            Chy = zeros(nAC,norder^2 + norderrbf^2 + nxk)
            for k = 1:(norder^2 + norderrbf^2 + nxk)
	    Chx[:,k] .= vec(chebypoly_x[k])
	    Chy[:,k] .= vec(chebypoly_y[k])
            end


            AC = zeros(norder^2 + norderrbf^2 + nxk,nAC^2)
            order = nAC

            col = 1
            for i = 1:nAC
	    for j = 1:nAC
	    AC[:,col] .= (Chx[i,:]) .* (Chy[j,:])
	    col += 1
	    end
            end


            Min = hcat(Matrix(vcat(Aincomp,II)),AC[idxXYin,:])#,hcat(AC[idxXYin,:]',zeros(nAC^2,nAC^2 + nxk - length(idxXYin))))

            Cfoo = Min\vcat(FcompBgrid[idxcompBin],Fxk[idxxkin])
            =#
            #F = hcat(Acomp,AC[1:(norder^2 + norderrbf^2),:]) * Cfoo
            #F[idxBout] = Aoutcomp * C
            #F .= FBgrid
            #F[idxcompBin] .= FcompBgrid[idxcompBin]
            #F = Acomp*C
            #F = Acomp * Fxk
            #F = A * C
            #F[idxBin] .= FBgrid[idxBin]	 
            #F = F[1:norderrbf^2]
	    
            FBcheb = copy(transpose(reverse(reshape(F,(norderrbf,norderrbf)))))

            chebcoefs = interpspec2D_DCT_compcoeff(norderrbf, norderrbf, FBcheb)

            foo = sqrt(sum(abs.(chebcoefs .* reverse(diagm(ones(norderrbf)),dims=2)).^2)/norderrbf)

            #foo = sqrt(abs(chebcoefs[1,8])^2 + abs(chebcoefs[2,7])^2 + abs(chebcoefs[3,6])^2 + abs(chebcoefs[4,5])^2 + abs(chebcoefs[5,4])^2 + abs(chebcoefs[6,3])^2 + abs(chebcoefs[7,2])^2  + abs(chebcoefs[8,1])^2)
            if maxcheberror < foo
	        maxcheberror = foo
            end

            #xyeval = [0.0001,-0.011]
            #println("ERROR eval = ",abs(F2d(xyeval[1],xyeval[2]) - eval_func_2D(chebcoefs,xyeval[1],xyeval[2],norderrbf,-boxlen/2,boxlen/2,norderrbf,-boxlen/2,boxlen/2)))

            #for k = 1:length(idxBin)
            #	finterp = eval_func_2D(chebcoefs,Bgrid[1,idxBin[k]],Bgrid[2,idxBin[k]],norderrbf,-0.5,0.5,norderrbf,-0.5,0.5)
            #	println("ERROR = ", finterp .- F[idxBin[k]])
            #end
            maxerror = -1
            #Nrand = 1000
            #xrand = hcat((rand(Nrand) .- 0.5)*boxlen*0.95,(rand(Nrand) .- 0.5)*boxlen*0.95)
            idxxrandin = []
            #idxxrandout = []
            #errlocal = zeros(length(idxB0in))
            #errlocal = zeros(length(idxxkin))
            errlocal = []
            #for i = 1:length(idxxkin)#Nrandeval#length(idxB0in)# #
            #for i = 1:length(idxB0in)# # Nrandeval
	    for i = 1:Nrandeval
                #	err = abs(F2d(B0grid[1,idxB0in[i]],B0grid[2,idxB0in[i]])- eval_func_2D(chebcoefs,B0grid[1,idxB0in[i]],B0grid[2,idxB0in[i]],norderrbf,-boxlen/2,boxlen/2,norderrbf,-boxlen/2,boxlen/2))

                # err = abs(Fxk[idxxkin[i]]- eval_func_2D(chebcoefs,xk[idxxkin[i],1],xk[idxxkin[i],2],norderrbf,-boxlen/2,boxlen/2,norderrbf,-boxlen/2,boxlen/2))


                #	 if err > 1e-10
                #		  	errlocal[i] = err
                #	println(i, " ERROR xk = ",err)
	        #	  break
                #	 end
                #	  if err > maxerror
                #			 maxerror = err
                #		end

	        d = dot(normal,xrandeval[i,:] - P0)
                #	 println(d)
                dist2xk = sqrt.((xrandeval[i,1].-xk[idxxkout,1]).^2 .+ (xrandeval[i,2].-xk[idxxkout,2]).^2)
                idx = findall(dist2xk .<= diagh)
	        if (d <= 0) & (length(idx) == 0)
                    
                    #if (xrandeval[i,2] > topline)|| (xrandeval[i,2] < bottomline)		  
                    #		  println("Max err = ", ERROR)
		    push!(idxxrandin,i)
	 	   global err = abs(F2d(xrandeval[i,1],xrandeval[i,2]) - eval_func_2D(chebcoefs,xrandeval[i,1],xrandeval[i,2],norderrbf,-boxlen/2,boxlen/2,norderrbf,-boxlen/2,boxlen/2))
		    push!(errlocal,err)
#                    		  println("error = ", err)
                    if err > maxerrormat3[kk,i]
                        maxerrormat3[kk,i] = err
                    end
	            if err > maxerror
			maxerror = err
		    end
      
	        end

                #		 err = abs(F2d(xrand[i,1],xrand[i,2]) - eval_func_2D(chebcoefs,xrand[i,1],xrand[i,2],norderrbf,-boxlen/2,boxlen/2,norderrbf,-boxlen/2,boxlen/2))
                #	 	println("Error at xk: ", err)
	        #	  push!(idxxrandin,i)
                #		 if err > maxerror
                #			 maxerror = err
                #		end
            end
	    maxCerror = locmaxCerror
	    condmax = loccondmax
            #	 println(kk,", max error ", maxerror)
	    if isempty(errlocal)
		l2errorloc = eps()
	    else
		l2errorloc = sqrt(sum(abs.(errlocal).^2)/length(errlocal))
	    end
            if maxerror > ERROR
		ERROR = maxerror
	        global ik = kk
            end
            if l2errorloc > ERRORL2
		ERRORL2 = l2errorloc
	        #global ik = kk
            end
            #println("Max err = ", ERROR)
            #end
            #println("Max err = ", ERROR)
            #println("Max err * boxlen= ", maxerror * boxlen)
        end
        println(" - m = ", orderm, " - n = ", ordern)	 
        println("Max err = ", ERROR)
        println("L2 err = ", ERRORL2)
        println("C err = ", maxCerror)
        println("Cond AI = ", condmax)
        println("Interp error at Cheb = ",maxinterperror)
        println("Cheb coeff error = ", maxcheberror)
        errorv[itr,1] = ERROR
        errorv[itr,2] = ERRORL2
        global itr = itr + 1
    end
end

#=
Plots.plot([0;Nrand+1],([0.5e-10;0.5e-10]),lc=:red,legend = false,lw=5.0,ylim=(1e-15, 1e-9),yaxis=:log)

for i=1:Nrand
    idx = findall(maxerrormat3[:,i].>0)
    vals = maxerrormat3[idx,i]
    p = Plots.scatter!(i*ones(length(idx)),vals, mc=:black, ms=2,legend = false,yaxis=:log,ylabel = "Maxmimum error",xlabel = "Experiment number")
    display(p)
end
=#

#=
boxlen0 = boxlen/3
ax = -boxlen0/2
ay = -boxlen0/2
bx = boxlen0/2
by = boxlen0/2

x = [ax,bx,bx,ax,ax]
y = [ay,ay,by,by,ay]
plot(x,y)
scatter!(xk[idxxkin,1],xk[idxxkin,2])
plot!([-boxlen/2,boxlen/2,boxlen/2,-boxlen/2,-boxlen/2],[-boxlen/2,-boxlen/2,boxlen/2,boxlen/2,-boxlen/2])
=#
#println("Min error = ", toterr)
##################################################################
#=
order = norderrbf
numCoeffs = div((order+2)*(order+1),2)
scalexk = 1.0./ maximum(abs.(Bgrid[1,idxBin]));
scaleyk = 1.0./ maximum(abs.(Bgrid[2,idxBin]));
xsk=Bgrid[1,idxBin]*scalexk
ysk=Bgrid[2,idxBin]*scaleyk
Ak = zeros(length(idxBin), numCoeffs)
col = 1
for i = 0:order
for j = 0:(order-i)
Ak[:,col] = (xsk[:] .^ i) .* (ysk[:] .^ j)
global	col += 1
end
end
println(size(Ak))
#@show cond(Ak)

Fk = F2d.(Bgrid[1,idxBin], Bgrid[2,idxBin])
Ck = Ak \ Fk
# Scale coeff
col = 1
for i = 0:order
for j = 0:(order-i)
Ck[col] *=	 (scalexk .^ i) .* (scaleyk .^ j)
global	col += 1
end
end
# Do all points to look at interpolation error
Fe = zeros(norderrbf * norderrbf)
column = 1
for xpower = 0:order
for ypower = 0:(order-xpower)
global	Fe +=  (Ck[column] .* Bgrid[1,:].^xpower .* Bgrid[2,:].^ypower)
global	column += 1;
end
end
=#
#=
Fdatacheb = F2d.(Bgrid[1,idxBin],Bgrid[2,idxBin])

chebypoly_x = chebypoly.(Bgrid[1,:],-boxlen/2,boxlen/2,norderrbf)
chebypoly_y = chebypoly.(Bgrid[2,:],-boxlen/2,boxlen/2,norderrbf)

Chx = zeros(norderrbf,norderrbf^2)
Chy = zeros(norderrbf,norderrbf^2)
for k = 1:length(idxBin)
Chx[:,k] .= vec(chebypoly_x[k])
Chy[:,k] .= vec(chebypoly_y[k])
end


AC = zeros(norderrbf^2,norderrbf^2)
order = norderrbf

col = 1
for i = 1:order
for j = 1:order
AC[:,col] .= (Chx[i,:]) .* (Chy[j,:])
global	col += 1
end
end

CVand = AC[idxBin,:]\Fdatacheb
Fdatacheb = F2d.(Bgrid[1,idxBin],Bgrid[2,idxBin])


for i = 1:nxk
println("ERROR eval = ",abs(Fxk[i] - eval_func_2D(reshape(CVand,(norderrbf,norderrbf)),xk[i,1],xk[i,2],norderrbf,-boxlen/2,boxlen/2,norderrbf,-boxlen/2,boxlen/2)))
end

=#
#=
Fdatacheb = F2d.(Bgrid[1,idxBin],Bgrid[2,idxBin])

chebypoly_x = chebypoly.(0.0,-boxlen/2,boxlen/2,norderrbf)
chebypoly_y = chebypoly.(0.0,-boxlen/2,boxlen/2,norderrbf)

Chx = zeros(norderrbf,1)
Chy = zeros(norderrbf,1)
for k = 1:1
Chx[:,k] .= vec(chebypoly_x[k])
Chy[:,k] .= vec(chebypoly_y[k])
end


AC = zeros(norderrbf^2,1)
order = norderrbf

col = 1
for i = 1:order
for j = 1:order
AC[col,:] = (Chx[i,:]) .* (Chy[j,:])
global	col += 1
end
end

CVand = AC\Fdatacheb


=#
