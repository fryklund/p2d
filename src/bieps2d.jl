module bieps2d

using FMMLIB2D
using LinearAlgebra
using LinearMaps
using IterativeSolvers
using PolygonOps
using StaticArrays
using OffsetArrays
using Octavian

include("julia/AnalyticDomains.jl")
using .AnalyticDomains

include("julia/CurveDiscretization.jl")
using .CurveDiscretization

include("julia/FreeSpace.jl")
include("../tests/bieps2dTests.jl")

import .CurveDiscretization
export CurveDiscretization
export AnalyticDomains
export FreeSpace
export LinearSolvers
export LinearAlgebra
export IterativeSolvers
export PolygonOps
export StaticArrays

end # module
