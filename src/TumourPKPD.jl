module TumourPKPD

using LinearAlgebra, Statistics, Random
using OrdinaryDiffEq
using SciMLBase
using LogExpFunctions: log1pexp

include("pk.jl")
include("pd.jl")
# Workers add: include("data.jl"), include("hierarchical.jl"),
# include("simulate.jl"), include("validate.jl")

export Dose, dosing_schedule, PKModel, OneCmt, TwoCmt, conc
export Simeoni, NoDrug, PKExposure, simulate_tumour, simeoni_rhs!

end
