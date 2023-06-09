using FMMLIB2D

using LinearAlgebra
using Random
#using InlineTest

Random.seed!(0)

# TODO: Compute gradient and hessian

function direct_self(source, charge, dipvec, dipstr)
    pot = zero(charge)
    N = size(source, 2)
    for i=1:N
        for j=1:N
            if i == j
                continue
            end
            r1 = source[1,j]-source[1,i]
            r2 = source[2,j]-source[2,i]
            rsq = r1^2 + r2^2
            rdotq = r1*dipvec[1,j] + r2*dipvec[2,j]            
            pot[i] += charge[j]*log(rsq)/2 + dipstr[j]*rdotq/rsq
        end
    end
    return pot
end

function direct_targ(source, charge, dipvec, dipstr, target)
    M = size(target, 2)
    N = size(source, 2)
    pot = zeros(typeof(charge[1]), M)    
    for i=1:M
        for j=1:N
            r1 = source[1,j]-target[1,i]
            r2 = source[2,j]-target[2,i]
            rsq = r1^2 + r2^2
            rdotq = r1*dipvec[1,j] + r2*dipvec[2,j]            
            pot[i] += charge[j]*log(rsq)/2 + dipstr[j]*rdotq/rsq
        end
    end
    return pot
end    

@testset "Laplace FMM 2D (complex)" begin
    iprec = 5
    N = 40
    M = 30
    source = rand(2, N)
    charge = rand(N) + 1im*rand(N)
    dipstr = rand(N) + 1im*rand(N)
    dipvec = rand(2, N)
    target = rand(2, M)
    ifcharge = 1
    ifdipole = 1
    ifpot = 1
    ifgrad = 1
    ifhess = 1
    ifpottarg = 1
    ifgradtarg = 1
    ifhesstarg = 1
    refpottarg = direct_targ(source, charge, dipvec, dipstr, target)
    refpot = direct_self(source, charge, dipvec, dipstr)
    @testset "Direct interface: lfmm2dpartself" begin
        pot, grad, hess = 
            lfmm2dpartself(iprec, source, ifcharge, charge, ifdipole, dipstr, dipvec, ifpot, ifgrad, ifhess)
        pot_relerr = norm(refpot-pot, Inf) / norm(refpot, Inf)
        @test pot_relerr < 1e-15
    end
    
    @testset "Direct interface: lfmm2dparttarg" begin
        pot, grad, hess, pottarg, gradtarg, hesstarg = 
            lfmm2dparttarg(iprec, source, ifcharge, charge, ifdipole, dipstr, dipvec, ifpot, ifgrad, ifhess,
                           target, ifpottarg, ifgradtarg, ifhesstarg)
        pot_relerr = norm(refpot-pot, Inf) / norm(refpot, Inf)
        pottarg_relerr = norm(refpottarg-pottarg, Inf) / norm(refpottarg, Inf)    
        @test pot_relerr < 1e-15
        @test pottarg_relerr < 1e-15
    end
    @testset "Keyword interface" begin
        U = lfmm2d(source=source, charge=charge, dipstr=dipstr, dipvec=dipvec)
        @test norm(U.pot-refpot, Inf) / norm(refpot, Inf) < 1e-14
        U = lfmm2d(source=source, charge=charge, dipstr=dipstr, dipvec=dipvec, target=target)        
        @test norm(U.pottarg-refpottarg, Inf) / norm(refpot, Inf) < 1e-14
    end
    @testset "Tolerance test" begin
        N = 2000
        source = rand(2, N)
        charge = rand(N) + 1im*rand(N)
        dipstr = rand(N) + 1im*rand(N)
        dipvec = rand(2, N)        
        refpot = direct_self(source, charge, dipvec, dipstr)
        for tolexp=-14:-1
            TOL = 10.0^tolexp
            U = lfmm2d(source=source, charge=charge, tol=TOL, dipstr=dipstr, dipvec=dipvec)
            relerr = norm(U.pot-refpot) / norm(refpot)
            @test relerr < TOL
        end
    end
end

@testset "Laplace FMM 2D (real)" begin
    iprec = 5
    N = 40
    M = 30
    source = rand(2, N)
    charge = rand(N)
    dipstr = rand(N)
    dipvec = rand(2, N)
    target = rand(2, M)
    ifcharge = 1
    ifdipole = 1
    ifpot = 1
    ifgrad = 1
    ifhess = 1
    ifpottarg = 1
    ifgradtarg = 1
    ifhesstarg = 1
    refpottarg = direct_targ(source, charge, dipvec, dipstr, target)
    refpot = direct_self(source, charge, dipvec, dipstr)
    @testset "Direct interface: rfmm2dpartself" begin
        pot, grad, hess = 
            rfmm2dpartself(iprec, source, ifcharge, charge, ifdipole, dipstr, dipvec, ifpot, ifgrad, ifhess)
        pot_relerr = norm(refpot-pot, Inf) / norm(refpot, Inf)
        @test pot_relerr < 1e-15
    end    
    @testset "Direct interface: rfmm2dparttarg" begin
        pot, grad, hess, pottarg, gradtarg, hesstarg = 
            rfmm2dparttarg(iprec, source, ifcharge, charge, ifdipole, dipstr, dipvec, ifpot, ifgrad, ifhess,
                           target, ifpottarg, ifgradtarg, ifhesstarg)
        pot_relerr = norm(refpot-pot, Inf) / norm(refpot, Inf)
        pottarg_relerr = norm(refpottarg-pottarg, Inf) / norm(refpottarg, Inf)    
        @test pot_relerr < 1e-15
        @test pottarg_relerr < 1e-15
    end
    @testset "Keyword interface" begin
        U = rfmm2d(source=source, charge=charge, dipstr=dipstr, dipvec=dipvec)
        @test norm(U.pot-refpot, Inf) / norm(refpot, Inf) < 1e-14
        U = rfmm2d(source=source, charge=charge, dipstr=dipstr, dipvec=dipvec, target=target)        
        @test norm(U.pottarg-refpottarg, Inf) / norm(refpot, Inf) < 1e-14
    end
    @testset "Tolerance test" begin
        N = 2000
        source = rand(2, N)
        charge = rand(N)
        dipstr = rand(N)
        dipvec = rand(2, N)        
        refpot = direct_self(source, charge, dipvec, dipstr)
        for tolexp=-14:-1
            TOL = 10.0^tolexp
            U = rfmm2d(source=source, charge=charge, tol=TOL, dipstr=dipstr, dipvec=dipvec)
            relerr = norm(U.pot-refpot) / norm(refpot)
            @test relerr < TOL
        end
    end    
end
