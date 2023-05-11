#using bieps2d
using ReTest

include("bieps2dTests.jl")

retest(bieps2d, bieps2dTests)
