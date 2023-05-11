module AnalyticDomains

# Curve parametrization
struct CurveParam
    tau
    dtau
    d2tau
    normal
    tangent
    orientation
    center
end

export starfish, CurveDesc, kite, harmonicext

function ellipse(;a = 1, b = 1, center=zeros(Float64,2,1),exterior=false)
    if exterior
	s = -1.0
    else
	s = 1.0
    end
    desc(t) = a * cos(t*s) + b * sin(t*s)*1im + (center[1] + 1im*center[2])
    diff1(t) =  - s * a * sin(t*s) + s * b * cos(t*s)*1im + eps()
    diff2(t) =  - a * cos(t*s) - b * sin(t*s)*1im + eps()
    tang(t) = diff1(t) / abs(diff1(t))
    normal(t) = .-1im*tang(t)
    orientation = s
    return CurveParam(
        desc,
        diff1,
        diff2,
        normal,
        tang,
        orientation,
	center
    )
end

function starfish(;n_arms = 5, amplitude = 0.3, radius = 1.0, center=zeros(Float64,2,1), exterior=false)
    if exterior
	s = -1.0
    else
	s = 1.0
    end
    desc(t) = radius*((1 + amplitude*cos(n_arms*t)).*exp(1im*s*t)) + (center[1] + 1im*center[2])
    diff1(t) = radius*(exp(t*1im*s)*(amplitude*cos(n_arms*t) + 1)*1im*s - amplitude*n_arms*exp(t*1im*s)*sin(n_arms*t))
    diff2(t) = radius*(- exp(t*1im*s)*(amplitude*cos(n_arms*t) + 1) - amplitude*n_arms*exp(t*1im*s)*sin(n_arms*t)*2*1im*s - amplitude*n_arms^2*exp(t*1im*s)*cos(n_arms*t))
    tang(t) = diff1(t) / abs(diff1(t))
    normal(t) = .-1im*tang(t)

    orientation = s
    return CurveParam(
        desc,
        diff1,
        diff2,
        normal,
        tang,
        orientation,
	center
    )
end



function harmonicext(a,b,c0,c,d,R,exterior=false,center=zeros(Float64,2,1))
    function csindcos(j,c,d,R,s,t)
	(c * sin(j*t) + d * cos(j * t)) * R * exp(s * 1im * t)
    end
    function dcsindcos(j,c,d,R,s,t)
	(c * j * cos(j*t) - d * j * sin(j*t)) * R * exp(s * 1im*t) + s * 1im * (c * sin(j*t) + d * cos(j*t)) * R * exp(s * 1im*t)
    end
    function d2csindcos(j,c,d,R,s,t)
	(-c * j^2 * sin(j*t) - d * j^2 * cos(j*t)) * R * exp(s * 1im*t) + 		s * 1im * (c * j * cos(j*t) - d * j * sin(j*t)) * R * exp(s * 1im*t)+ s * 1im * (c * j * cos(j*t) - d * j * sin(j*t)) * R * exp(s * 1im*t) - (c * sin(j*t) + d * cos(j*t)) * R * exp(s * 1im*t)
    end
    
    if exterior
	s = -1.0
    else
	s = 1.0
    end
    desc0(t) = a + 1im * b + c0 * R * exp(s * 1im * t)
    diff10(t) =  s * 1im * c0  * R * exp(s * 1im * t) + eps()
    diff20(t) =   - c0  * R * exp(s * 1im*t) + eps()
    
    descj(t) = csindcos(1,c[1],d[1],R,s,t) + csindcos(2,c[2],d[2],R,s,t) + csindcos(3,c[3],d[3],R,s,t) + csindcos(4,c[4],d[4],R,s,t) + csindcos(5,c[5],d[5],R,s,t) + csindcos(6,c[6],d[6],R,s,t) + csindcos(7,c[7],d[7],R,s,t) + csindcos(8,c[8],d[8],R,s,t) + csindcos(9,c[9],d[9],R,s,t) + csindcos(10,c[10],d[10],R,s,t)
    diff1j(t) = dcsindcos(1,c[1],d[1],R,s,t) + dcsindcos(2,c[2],d[2],R,s,t) + dcsindcos(3,c[3],d[3],R,s,t) .+ dcsindcos(4,c[4],d[4],R,s,t) .+ dcsindcos(5,c[5],d[5],R,s,t) .+ dcsindcos(6,c[6],d[6],R,s,t) .+ dcsindcos(7,c[7],d[7],R,s,t) .+ dcsindcos(8,c[8],d[8],R,s,t) + dcsindcos(9,c[9],d[9],R,s,t) + dcsindcos(10,c[10],d[10],R,s,t)

    diff2j(t) = d2csindcos(1,c[1],d[1],R,s,t) + d2csindcos(2,c[2],d[2],R,s,t) + d2csindcos(3,c[3],d[3],R,s,t) .+ d2csindcos(4,c[4],d[4],R,s,t) .+ d2csindcos(5,c[5],d[5],R,s,t) .+ d2csindcos(6,c[6],d[6],R,s,t) .+ d2csindcos(7,c[7],d[7],R,s,t) .+ d2csindcos(8,c[8],d[8],R,s,t) + d2csindcos(9,c[9],d[9],R,s,t) + d2csindcos(10,c[10],d[10],R,s,t)
    
    desc(t) = desc0(t) + descj(t)
    diff1(t) = diff10(t) + diff1j(t)
    diff2(t) = diff20(t) + diff2j(t)	 
    tang(t) = diff1(t) / abs(diff1(t))
    normal(t) = .-1im*tang(t)
    orientation = s
    return CurveParam(
	desc,
	diff1,
	diff2,
	normal,
	tang,
	orientation,
	center
    )
end

function kite(;exterior=false,center=zeros(Float64,2,1))
    if exterior
        s = -1.0
    else
        s = 1.0
    end
    desc(t) = cos(t) + 0.35 * cos(2*t) - 0.35 + 1im * 0.7 * sin(t)
    diff1(t) = -sin(t) - 0.7 * sin(2*t) + 1im * 0.7 * cos(t)
    diff2(t) = -cos(t) - 1.4 * cos(2*t) - 1im * 0.7 * sin(t)
    tang(t) = diff1(t) / abs(diff1(t))
    normal(t) = .-1im*tang(t)
    orientation = s
    return CurveParam(
        desc,
        diff1,
        diff2,
        normal,
        tang,
	orientation,
	center
    )
end

function saw(;radius = 1.0,  N = 7, b = 2, c = 0.5, center=zeros(Float64,2,1), exterior=false)
    if exterior
        s = -1.0
    else
        s = 1.0
    end
    desc(t) = radius * (((b + c * sin(s*N*t))) * cos(s*t + c*sin(N*s*t)) + 1im*(((b + c * sin(s*N*t))) * sin(s*t + c*sin(s*N*t)))) + (center[1] + 1im*center[2])
    diff1(t) = radius * s * (cos(c * sin(N * s * t) + s * t) + 1im * sin(c * sin(N * s * t) + s * t)) * (1im * b * c * N * cos(N * s * t) + 1im * b + 1im * c^2 * N * sin(N * s * t) * cos(N * s * t) + 1im * c * sin(N * s * t) + c * N * cos(N * s * t))
    
    diff2(t) =  -1/4 * radius * s^2 * (cos(c * sin(N * s * t) + s * t) + 1im * sin(c * sin(N * s * t) + s * t)) * (2 *(b - 3 * 1im) * c^2 * N^2 * cos(2 * N * s * t) + 2 * b * c^2 * N^2 + 4 * 1im * b * c * N^2 * sin(N * s * t) + 8 * (b - 1im) * c * N * cos(N * s * t) + 4 * b + c^3 * N^2 * sin(N * s * t) + c^3 * N^2 * sin(3 * N * s * t) - 2 * 1im * c^2 * N^2 + 4 * c^2 * N * sin(2 * N * s * t) + 4 * c * N^2 * sin(N * s * t) + 4 * c * sin(N * s * t))
    tang(t) = diff1(t) / abs(diff1(t))
    normal(t) = .-1im*tang(t)
    orientation = s
    return CurveParam(
        desc,
        diff1,
        diff2,
        normal,
        tang,
        orientation,
	center
    )
end


function rectangle(;width = 1, height = 1, order = 10, exterior=false, center=zeros(Float64,2,1))
    @assert iseven(order)
    a = width
    b = height
    p = order;
    desc(t) = (a*cos(t) + b*sin(t)*1im)/(cos(t)^p + sin(t)^p)^(1/p) + (center[1] + 1im*center[2])
    _diff1(t) = -(a*cos(t)*sin(t)^p - b*cos(t)^p*sin(t)*1im)/(cos(t)*sin(t)*(cos(t)^p + sin(t)^p)^((p + 1)/p))    
    _diff2(t) = (-(a*cos(t)^(p + 1)*sin(t)^p + b*cos(t)^(2*p + 2)*sin(t)*2im - 2*a*cos(t)^3*sin(t)^(2*p) - 2*a*cos(t)^(p + 3)*sin(t)^p - b*cos(t)^p*sin(t)^(p + 1)*1im - b*cos(t)^(2*p)*sin(t)*2im + b*cos(t)^(p + 2)*sin(t)^(p + 1)*2im + a*p*cos(t)^(p + 1)*sin(t)^p + b*p*cos(t)^p*sin(t)^(p + 1)*1im)/(cos(t)^2*sin(t)^2*(cos(t)^p + sin(t)^p)^((2*p + 1)/p)))

    # Fudge because derivative expression is NaN at t=0
    diff1(t) = _diff1(t + eps(t))
    diff2(t) = _diff2(t + eps(t))

    
    tang(t) = diff1(t) / abs(diff1(t))
    normal(t) = .-1im*tang(t)
    orientation = 1
    return CurveDesc(
        desc,
        diff1,
        diff2,
        normal,
        tang,
	orientation,
	center
    )
end

end # module
