# Full suite. test_pd_mtk.jl (ModelingToolkit cross-check) is slow; set TUMOURPKPD_FAST=1 to skip it.
using Test
@testset "TumourPKPD" begin
    include("test_pk.jl")
    include("test_pd.jl")
    get(ENV, "TUMOURPKPD_FAST", "0") == "1" || include("test_pd_mtk.jl")
    include("test_data.jl")
    include("test_hierarchical.jl")
    include("test_validate.jl")
end
