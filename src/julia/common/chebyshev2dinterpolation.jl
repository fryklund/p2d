using FFTW
using SpecialFunctions

function meshgrid(xin::Vector{Float64},yin::Vector{Float64})
    nx =length(xin)
    ny = length(yin)
    xout = zeros(ny,nx)
    yout = zeros(ny,nx)
    for jx = 1:nx
        for ix = 1:ny
            xout[ix,jx] = xin[jx]
            yout[ix,jx] = yin[ix]
        end
    end
    return xout, yout
end


## Chebyshev nodes
# Given an interval (a,b), calculates the associated n Chebyshev nodes.

function chebynodes(a::Float64, b::Float64, n::Int64)

    nodes = zeros(n)

    @inbounds for ii = 1:1:n
        nodes[ii] = 0.5 * (b + a) - 0.5 * (a - b) * cos(((ii - 1) * pi) / (n - 1))
    end
    return nodes
end

# Chebyshevnodes of first kind

function chebynodesfirst(a::Float64, b::Float64, n::Int64)

    nodes = zeros(n)
    h = pi / (2*n)

    @inbounds for ii = 1:1:n
	t = (2*ii - 1) * h
	nodes[ii] = 0.5 * (b + a) - 0.5 * (a - b) * cos(t)
    end
    return nodes
end


## 2D Chebyshev grid

# Calculates the [X,Y] grid obtained by the mesh of two vx, vy vectors.

function chebynodes_grid(a1::Float64, b1::Float64, n1::Int64, a2::Float64, b2::Float64, n2::Int64)

    vx = chebynodes(a1,b1,n1)
    vy = chebynodes(a2,b2,n2)

    X,Y = meshgrid(vx,vy);

    return X,Y
end

function chebynodesfirst_grid(a1::Float64, b1::Float64, n1::Int64, a2::Float64, b2::Float64, n2::Int64)

    vx = chebynodesfirst(a1,b1,n1)
    vy = chebynodesfirst(a2,b2,n2)

    X,Y = meshgrid(vx,vy);

    return X,Y
end

## 2D Spectral interpolation using FFT
# n1,n2: number of Chebyshev nodes
# a1,a2: intervall [a1,b1], [a2,b2]
# b1,b2: intervall [a1,b1], [a2,b2]
# f: function to be interpolated, form f(x,y)

function interpspec2D_FFT(n1::Int64,a1::Float64,b1::Float64,n2::Int64,a2::Float64,b2::Float64,f)

    c = zeros(n2,n1)
    coefficients = zeros(n2,n1)

    # Chebyshev grid calculation
    X,Y = chebynodes_grid(a1,b1,n1,a2,b2,n2)

    # Evaluation of f on Chebyshev nodes
    fcheby = f.(X,Y)

    # First dimension - coefficients for [k]
    for k = 1:n2
        c[k,:] = coeff_fft(fcheby[k,:])
    end

    # Second dimension - coefficients for [k,l]
    for l=1:n1
        coefficients[:,l] = coeff_fft(copy(c[:,l]))
    end

    return coefficients
end

function interpspec2D_DCT(n1::Int64,a1::Float64,b1::Float64,n2::Int64,a2::Float64,b2::Float64,f)

    c = zeros(n2,n1)
    coefficients = zeros(n2,n1)

    # Chebyshev grid calculation
    X,Y = chebynodesfirst_grid(a1,b1,n1,a2,b2,n2)

    # Evaluation of f on Chebyshev nodes
    fcheby = f.(X,Y)

    # First dimension - coefficients for [k]
    for k = 1:n2
        c[k,:] = coeff_dct(fcheby[k,:])
    end

    # Second dimension - coefficients for [k,l]
    for l=1:n1
        coefficients[:,l] = coeff_dct(copy(c[:,l]))
    end

    return coefficients
end


function interpspec2D_FFT_compcoeff(n1::Int64, n2::Int64, fcheby::Matrix{Float64})
    c = zeros(n2,n1)
    coefficients = zeros(n2,n1)

    # First dimension - coefficients for (k)
    for k=1:n2
        c[k,:] = coeff_fft(fcheby[k,:])   
    end

    # Second dimension - coefficients for (k,l)
    for l=1:n1
        coefficients[:,l] = coeff_fft(copy(c[:,l]))
    end
    return coefficients
end

function interpspec2D_DCT_compcoeff(n1::Int64, n2::Int64, fcheby::Matrix{Float64})
    c = zeros(n2,n1)
    coefficients = zeros(n2,n1)

    # First dimension - coefficients for (k)
    for k=1:n2
        c[k,:] = coeff_dct(fcheby[k,:])   
    end
    #	 N = length(value)
    #coefficients = sqrt(2/N) * dct(value)
    #coefficients[1] = coefficients[1] / sqrt(2)
    #return  coefficients
    #	 N = size(fcheby,2)
    #cc = sqrt(2/N) * dct(fcheby,[2])
    #cc[:,1] .= cc[:,1] / sqrt(2)
    #println(norm(c - cc))
    
    # Second dimension - coefficients for (k,l)
    for l=1:n1
        coefficients[:,l] = coeff_dct(copy(c[:,l]))
    end
    return coefficients
end


function diffx_interpspec2D_DCT_compcoeff(n1::Int64, n2::Int64, c::Matrix{Float64})
    dcout = zeros(n2,n1)
    # First dimension - coefficients for (k)
    for k=1:n2
        dcout[:,k] .= chebdiff(c[:,k],n2-1)
    end
    return collect(dcout')
end

function diffy_interpspec2D_DCT_compcoeff(n1::Int64, n2::Int64, c::Matrix{Float64})
    dcout = zeros(n2,n1)
    # First dimension - coefficients for (k)
    for l=1:n1
        dcout[l,:] .= chebdiff(c[l,:],n1-1)
    end

    return collect(dcout')
end


function diffxy_interpspec2D_DCT_compcoeff(n1::Int64, n2::Int64, c::Matrix{Float64})
    dc = zeros(n2,n1)
    dcout = zeros(n2,n1)
    # First dimension - coefficients for (k)
    for k=1:n2
        dc[:,k] .= chebdiff(c[:,k],n2-1)
    end
    #	 N = length(value)
    #coefficients = sqrt(2/N) * dct(value)
    #coefficients[1] = coefficients[1] / sqrt(2)
    #return  coefficients
    #	 N = size(fcheby,2)
    #cc = sqrt(2/N) * dct(fcheby,[2])
    #cc[:,1] .= cc[:,1] / sqrt(2)
    #println(norm(c - cc))
    
    # Second dimension - coefficients for (k,l)
    for l=1:n1
        dcout[l,:] .= chebdiff(dc[l,:],n1-1)
    end
    return collect(dcout')
end

function diff_interpspec2D_DCT_compcoeff(n1::Int64, n2::Int64, c::Matrix{Float64})
    dc = zeros(n2,n1)
    dcout = zeros(n2,n1)
    # First dimension - coefficients for (k)
    for k=1:n2
        dc[:,k] .= chebdiff(c[:,k],n2-1)
    end
    #	 N = length(value)
    #coefficients = sqrt(2/N) * dct(value)
    #coefficients[1] = coefficients[1] / sqrt(2)
    #return  coefficients
    #	 N = size(fcheby,2)
    #cc = sqrt(2/N) * dct(fcheby,[2])
    #cc[:,1] .= cc[:,1] / sqrt(2)
    #println(norm(c - cc))
    
    # Second dimension - coefficients for (k,l)
    for l=1:n1
        dcout[l,:] .= chebdiff(dc[l,:],n1-1)
    end
    return dcout,dc
end

function DCT_compcoeff_2d(n1::Int64, n2::Int64, fcheby::Matrix{Float64})

    coefficients = zeros(n2,n1)

    # First dimension - coefficients for (k)
    #for k=1:n2
    #   c[k,:] = coeff_dct(fcheby[k,:])   
    #end
    #	 N = length(value)
    #coefficients = sqrt(2/N) * dct(value)
    #coefficients[1] = coefficients[1] / sqrt(2)
    #return  coefficients
    #	 N = size(fcheby,2)
    coefficients = sqrt(2/n1) * dct(fcheby,[2])
    coefficients[:,1] .= coefficients[:,1] / sqrt(2)
    #println(norm(c - cc))
    
    # Second dimension - coefficients for (k,l)
    #for l=1:n1
    #   coefficients[:,l] = coeff_dct(copy(c[:,l]))
    #end

    coefficients = sqrt(2/n2) * dct(coefficients,[1])
    coefficients[1,:] .= coefficients[1,:] / sqrt(2)
    return coefficients
end



## Chebyshev polynomials
# Calculates the polynomials functions of N degree, interval [a,b]

@inline function chebypoly(x::Float64,a::Float64,b::Float64,N::Int64)

    # Necessary transformation to ensure that x lies in [-1,1]
    x = (2 * (x-a) / (b-a) ) - 1;

    # Initialization
    chebypolys = zeros(N,length(x))

    # T_1(x) = 1
    chebypolys[1,:] .= 1

    # T_2(x) = x
    if N > 1
        chebypolys[2,:] .= x
    end

    # for n>2, T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    if N > 2
        @inbounds for k = 3:N
            @fastmath chebypolys[k,:] .= 2 * x.* chebypolys[k-1,:] .- chebypolys[k-2,:]
        end
    end
    return chebypolys
end

## Calculation of Chebyshev coefficients using the DCT method

function coeff_dct(value::Vector{Float64})
    # x =-cos(pi/N*((0:N-1)'+1/2))
    #value = value[end:-1:1]
    N = length(value)
    coefficients = sqrt(2/N) * dct(value)
    coefficients[1] = coefficients[1] / sqrt(2)
    return  coefficients
end

## Calculation of Chebyshev coefficients using the FFT method

function coeff_fft(value::Vector{Float64})

    N = length(value)-1
    coeff = real(fft(vcat(value,value[N:-1:2])))
    coefficients = vcat(coeff[1]/(2*N), coeff[2:N]/N, coeff[N+1]/(2*N))
    return  coefficients
end

## Evaluation of a 2D function at the point (x,y)

# To use with the coefficients obtained after running interspec2D_FFT.

@inline function eval_func_2D(coefficients::Matrix{Float64},x::Float64,y::Float64,n1::Int64,a1::Float64,b1::Float64,n2::Int64,a2::Float64,b2::Float64)
    chebypoly_x = chebypoly(x,a1,b1,n1)
    chebypoly_y = chebypoly(y,a2,b2,n2)
    z = (chebypoly_y' * coefficients * chebypoly_x)[1]
    return z
end




################################################
using Random

function test()
    ## MAIN - SPECTRAL INTERPOLATION - FFT - 2D

    # This script interpolates a function generated by random coefficients.

    # It then proceeds to compare the original coefficients with the
    # interpolated ones.

    ## Parameters

    # Axis 1
    n1 = 100	
    a1 = -5.0
    b1 = 5.0

    # Axis 2
    n2 = 100
    a2 = -3.0
    b2 = 3.0
    
    ## PART ZERO - DEFINITION OF THE FUNCTION

    # Definition of the function to be interpolated
    coeff_sim = randn(n2,n1)
    # Grid to be interpolated
    X,Y = chebynodes_grid(a1, b1, n1, a2, b2, n2)
    ## PART ONE - INTERPOLATION AND EVALUATION

    println("Evaluation of the function...")
    f_interp = zeros(size(X,1), size(X,2))
    for j = 1:size(X,2)
        for i=1:size(X,1)
            f_interp[i,j] = eval_func_2D(coeff_sim,X[i,j],Y[i,j],n1,a1,b1,n2,a2,b2)
        end
    end

    ## PART TWO - INTERPOLATION

    println("Coefficients calculation ...")
    coefficients = interpspec2D_FFT_compcoeff(n1,n2,f_interp)

    ## PART THREE - COMPARISON OF COEFFICIENTS

    println("Coefficients error...")
    error = sum(sum(abs.(coefficients .- coeff_sim)))


end

function test_dct()
    ## MAIN - SPECTRAL INTERPOLATION - FFT - 2D

    # This script interpolates a function generated by random coefficients.

    # It then proceeds to compare the original coefficients with the
    # interpolated ones.

    ## Parameters

    # Axis 1
    n1 = 100	
    a1 = -5.0
    b1 = 5.0

    # Axis 2
    n2 = 100
    a2 = -3.0
    b2 = 3.0
    
    ## PART ZERO - DEFINITION OF THE FUNCTION

    # Definition of the function to be interpolated
    coeff_sim = randn(n2,n1)
    # Grid to be interpolated
    X,Y = chebynodesfirst_grid(a1, b1, n1, a2, b2, n2)
    ## PART ONE - INTERPOLATION AND EVALUATION

    println("Evaluation of the function...")
    f_interp = zeros(size(X,1), size(X,2))
    for j = 1:size(X,2)
        for i=1:size(X,1)
            f_interp[i,j] = eval_func_2D(coeff_sim,X[i,j],Y[i,j],n1,a1,b1,n2,a2,b2)
        end
    end

    ## PART TWO - INTERPOLATION

    println("Coefficients calculation ...")
    coefficients = interpspec2D_DCT_compcoeff(n1,n2,f_interp)

    ## PART THREE - COMPARISON OF COEFFICIENTS

    println("Coefficients error...")
    error = sum(sum(abs.(coefficients .- coeff_sim)))


end


function example()
    ## MAIN - SPECTRAL INTERPOLATION - FFT - 2D

    # This script evaluates a function f = @(x,y) on the grid [T1,T2] for the
    # points defined by t1,t2.

    ## Parameters

    # Definition of the function to be interpolated
    @. f(x,y) = sin(x + y + eps()) / (x + y + eps())
    #@. f(x,y) = sin(x*pi/100) * sin(y*pi/100)
    #@. f(x,y) = erfc(abs(x)*5.5) * erfc(abs(y)*5.5)
    # Axis 1
    n1 = 200
    a1 = -1.0
    b1 = 1.0
    delta1 = .01

    # Axis 2
    n2 = 200
    a2 = -1.0
    b2 = 1.0
    delta2 = 0.5

    # Grid to be interpolated
    t1 = collect(a1:delta1:b1)
    t2 = collect(a2:delta2:b2)
    T1, T2 = meshgrid(t1,t2)

    # PART ONE - INTERPOLATION AND EVALUATION

    # Step one : interpolation of the function.
    println("Coefficients calculation ...")
    coefficients = interpspec2D_FFT(n1,a1,b1,n2,a2,b2,f);

    # Step two : Evaluation of the interpolated function on the vector t.
    println("Evaluation of the function...")
    f_interp = zeros(length(t2),length(t1))
    for j=1:length(t2)
        for i=1:length(t1)
            f_interp[j,i] = eval_func_2D(coefficients,T1[j,i],T2[j,i],n1,a1,b1,n2,a2,b2)
        end
    end

    ## PART TWO - OPTIONAL
    # Evaluation of f on the Chebyshev nodes.

    X,Y = chebynodes_grid(a1,b1,n1,a2,b2,n2)
    fcheby = f(X,Y)

    # Evaluation of f on the grid [T1,T2].
    f_t = f(T1,T2)
    # We now want to compare our interpolated function with the original function.


    ## PART THREE - INTERPOLATION ERROR

    println("Interpolation error")
    error = maximum(abs.(f_t .- f_interp))
    println(error)
    return f_interp, f_t
end

function example_differentiate()
    ## MAIN - SPECTRAL INTERPOLATION - FFT - 2D

    # This script evaluates the derivative of a function f = @(x,y) on the grid [T1,T2] for the
    # points defined by t1,t2.

    ## Parameters

    # Definition of the function to be interpolated
    @. f(x,y) = sin(x + y + eps()) / (x + y + eps())
    #@. f(x,y) = sin(x*pi/100) * sin(y*pi/100)
    #@. f(x,y) = erfc(abs(x)*5.5) * erfc(abs(y)*5.5)
    # Axis 1
    n1 = 200
    a1 = -2.23
    b1 = 12.0
    delta1 = .01

    # Axis 2
    n2 = 200
    a2 = -3.02
    b2 = 0.02
    delta2 = 0.5

    # Grid to be interpolated
    t1 = collect(a1:delta1:b1)
    t2 = collect(a2:delta2:b2)
    T1, T2 = meshgrid(t1,t2)

    # PART ONE - INTERPOLATION AND EVALUATION

    # Step one : interpolation of the function.
    println("Coefficients calculation ...")
    coefficients = interpspec2D_FFT(n1,a1,b1,n2,a2,b2,f);

    # Step two : differentiate the function
    
    # Step three : Evaluation of the interpolated function on the vector t.
    println("Evaluation of the function...")
    f_interp = zeros(length(t2),length(t1))
    for j=1:length(t2)
        for i=1:length(t1)
            f_interp[j,i] = eval_func_2D(coefficients,T1[j,i],T2[j,i],n1,a1,b1,n2,a2,b2)
        end
    end

    ## PART TWO - OPTIONAL
    # Evaluation of f on the Chebyshev nodes.

    X,Y = chebynodes_grid(a1,b1,n1,a2,b2,n2)
    fcheby = f(X,Y)

    # Evaluation of f on the grid [T1,T2].
    f_t = f(T1,T2)
    # We now want to compare our interpolated function with the original function.


    ## PART THREE - INTERPOLATION ERROR

    println("Interpolation error")
    error = maximum(abs.(f_t .- f_interp))
    println(error)
    return f_interp, f_t
end

function example_dct()
    ## MAIN - SPECTRAL INTERPOLATION - FFT - 2D

    # This script evaluates a function f = @(x,y) on the grid [T1,T2] for the
    # points defined by t1,t2.

    ## Parameters

    # Definition of the function to be interpolated
    @. f(x,y) = sin(x + eps()) / (x + eps()) + sin(y + eps()) / (y + eps())
    #@. f(x,y) = sin(x*pi/100) * sin(y*pi/100)
    #@. f(x,y) = x * (y-0.1) ^6
    # Axis 1
    n1 = 14
    a1 = -1.0
    b1 = 1.0
    delta1 = .01

    # Axis 2
    n2 = 14
    a2 = -1.0
    b2 = 1.0
    delta2 = 0.5

    # Grid to be interpolated
    t1 = collect(a1:delta1:b1)
    t2 = collect(a2:delta2:b2)
    T1, T2 = meshgrid(t1,t2)

    # PART ONE - INTERPOLATION AND EVALUATION

    # Step one : interpolation of the function.
    println("Coefficients calculation ...")
    coefficients = interpspec2D_DCT(n1,a1,b1,n2,a2,b2,f);

    # Step two : Evaluation of the interpolated function on the vector t.
    println("Evaluation of the function...")
    f_interp = zeros(length(t2),length(t1))
    for j=1:length(t2)
        for i=1:length(t1)
            f_interp[j,i] = eval_func_2D(coefficients,T1[j,i],T2[j,i],n1,a1,b1,n2,a2,b2)
        end
    end

    ## PART TWO - OPTIONAL
    # Evaluation of f on the Chebyshev nodes.

    #X,Y = chebynodesfirst_grid(a1,b1,n1,a2,b2,n2)
    #fcheby = f(X,Y)

    # Evaluation of f on the grid [T1,T2].
    f_t = f(T1,T2)

    # We now want to compare our interpolated function with the original function.


    ## PART THREE - INTERPOLATION ERROR

    println("Interpolation error")
    error = maximum(abs.(f_t .- f_interp))
    println(error)
    return f_interp, f_t
end
