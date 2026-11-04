# One OS process per chain. Chains inside one Julia process scale badly for this model (the ODE solves
# allocate heavily and share a single GC: 4 concurrent tasks ran ~4x slower per gradient each), four
# single-threaded processes run at full speed. A script that includes this file re-invokes itself with
# `--chain-only=<k> --chain-out=<file>`; the child samples chain k and saves it with `save_chain_parts`.
using JLD2, MCMCChains

function save_chain_parts(path, chain)
    data, names, internals = TumourPKPD.chain_to_parts(chain)
    jldsave(path; data, names, internals)
end
function load_chain_parts(path)
    d = JLD2.load(path)
    return TumourPKPD.chain_from_parts(d["data"], d["names"], d["internals"])
end

function save_init(path, nt::NamedTuple, k2_centered::AbstractVector{Bool})
    jldsave(path; keys = collect(String.(keys(nt))), vals = collect(Any, values(nt)), k2_centered = collect(Bool, k2_centered))
end
"Returns (start NamedTuple, k2_centered mask) written by `save_init`."
function load_init(path)
    d = JLD2.load(path)
    return NamedTuple{Tuple(Symbol.(d["keys"]))}(Tuple(d["vals"])), collect(Bool, d["k2_centered"])
end

"""
    run_chain_procs(script, args, nchains; maxprocs, project, threads=1) -> (chain::Chains, wall_seconds)

Runs `julia --project=project -t threads script args... --chain-only=k --chain-out=tmp_k` for k in 1:nchains, at most
`maxprocs` at a time, and concatenates the saved chains (`chainscat`).
"""
function run_chain_procs(script, args, nchains; maxprocs = nchains, project = Base.active_project(), threads = 1)
    tmp = mktempdir()
    t0 = time()
    sem = Base.Semaphore(max(maxprocs, 1))
    @sync for k in 1:nchains
        @async begin
            Base.acquire(sem)
            try
                out = joinpath(tmp, "chain$(k).jld2")
                cmd = `$(Base.julia_cmd()) --project=$(dirname(project)) -t $threads $script $args --chain-only=$k --chain-out=$out`
                run(cmd)
            finally
                Base.release(sem)
            end
        end
    end
    chains = [load_chain_parts(joinpath(tmp, "chain$(k).jld2")) for k in 1:nchains]
    wall = time() - t0
    rm(tmp; recursive = true, force = true)
    return chainscat(chains...), wall
end
