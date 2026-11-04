# Build the job matrices for .github/workflows/fits.yml from the workflow inputs.
#
#   julia scripts/ci_matrix.jl [--drugs=a,b] [--designs=primary,late21] [--nchains=4] [--nadapt=500]
#                              [--parts=2] [--draws=250] [--seglen=250] [--burnin=25] [--units=N]
#
# Inputs come from the command line, else from the environment (IN_DRUGS, IN_DESIGNS, IN_NCHAINS,
# IN_NADAPT, IN_PARTS, IN_DRAWS, IN_SEGLEN, IN_BURNIN, IN_UNITS), else from the defaults below. Standard library only, so it runs
# without instantiating the project. With $GITHUB_OUTPUT set it appends the outputs there; without it
# (local dry run) it prints them. Invalid input is an error: the values end up in artifact names, paths and
# shell commands, so drug names are restricted to [A-Za-z0-9._-] and checked against data/regimens.toml.
#
# Chain K and part P are 1-based.

using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DESIGNS = ["primary", "late21"]
const MAX_MATRIX = 256        # GitHub limit on jobs per matrix
const DEFAULTS = Dict("drugs" => "binimetinib,BKM120,BYL719", "designs" => "primary,late21", "nchains" => "4",
    "nadapt" => "500", "parts" => "2", "draws" => "250", "seglen" => "250", "burnin" => "25", "units" => "")

function parse_args(args)
    kv = Dict{String,String}()
    for a in args
        m = match(r"^--([a-z]+)=(.*)$", a)
        m === nothing && error("unrecognised argument $(repr(a)); use --name=value")
        haskey(DEFAULTS, m[1]) || error("unknown option --$(m[1])")
        kv[m[1]] = m[2]
    end
    kv
end

function setting(kv, name)
    haskey(kv, name) && return strip(kv[name])
    e = get(ENV, "IN_" * uppercase(name), nothing)
    (e === nothing || isempty(strip(e))) && return DEFAULTS[name]
    strip(e)
end

function splitlist(s)
    xs = [strip(x) for x in split(s, ',') if !isempty(strip(x))]
    unique(xs)
end

function intopt(s, name; lo, hi)
    v = tryparse(Int, s)
    (v === nothing || v < lo || v > hi) && error("$name must be an integer in $lo:$hi, got $(repr(s))")
    v
end

function known_drugs()
    f = joinpath(ROOT, "data", "regimens.toml")
    isfile(f) || return nothing
    collect(keys(TOML.parsefile(f)["drugs"]))
end

function build(kv)
    drugs = splitlist(setting(kv, "drugs"))
    designs = splitlist(setting(kv, "designs"))
    isempty(drugs) && error("drugs is empty")
    isempty(designs) && error("designs is empty")
    for d in drugs
        occursin(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", d) || error("invalid drug name $(repr(d))")
    end
    kd = known_drugs()
    if kd !== nothing
        for d in drugs
            d in kd || error("unknown drug $(repr(d)); data/regimens.toml has $(join(sort(kd), ", "))")
        end
    end
    for s in designs
        s in DESIGNS || error("unknown design $(repr(s)); allowed: $(join(DESIGNS, ", "))")
    end
    nchains = intopt(setting(kv, "nchains"), "nchains"; lo = 1, hi = 16)
    parts = intopt(setting(kv, "parts"), "parts"; lo = 1, hi = 32)
    nadapt = intopt(setting(kv, "nadapt"), "nadapt"; lo = 1, hi = 100_000)
    draws = intopt(setting(kv, "draws"), "draws_per_part"; lo = 1, hi = 100_000)
    # seglen > 0 splits warmup into two jobs per chain (segments of seglen iterations); only two segments are supported.
    # seglen >= nadapt needs one segment only, so it is the same as a single warmup job.
    seglen = intopt(setting(kv, "seglen"), "seglen"; lo = 0, hi = 100_000)
    seglen > 0 && nadapt > 2 * seglen && error("nadapt ($nadapt) must be <= 2 * seglen ($(2 * seglen)): only two warmup segments are supported")
    segmented = seglen > 0 && seglen < nadapt
    burnin = intopt(setting(kv, "burnin"), "burnin"; lo = 0, hi = 100_000)
    us = setting(kv, "units")
    units = isempty(us) ? 0 : intopt(us, "units"; lo = 1, hi = 100_000)

    pairs = [(d, s) for d in drugs for s in designs]
    init = [(drug = d, design = s) for (d, s) in pairs]
    warmup = [(drug = d, design = s, chain = k) for (d, s) in pairs for k in 1:nchains]
    sample = [(drug = d, design = s, chain = k, part = p) for (d, s) in pairs for k in 1:nchains for p in 1:parts]
    for (name, m) in (("init", init), ("warmup", warmup), ("sample", sample))
        length(m) <= MAX_MATRIX || error("$name matrix has $(length(m)) jobs, limit is $MAX_MATRIX")
    end
    (; drugs, designs, nchains, parts, nadapt, draws, seglen, segmented, burnin, units, init, warmup, sample, merge = init)
end

# Compact JSON for the flat NamedTuple rows used above (strings and integers only).
jsonval(x::Integer) = string(x)
jsonval(x::AbstractString) = "\"" * x * "\""      # validated: no characters that need escaping
jsonrow(r) = "{" * join(("\"$k\":" * jsonval(v) for (k, v) in pairs(r)), ",") * "}"
matrix_json(rows) = "{\"include\":[" * join(jsonrow.(rows), ",") * "]}"

function outputs(b)
    [
        "matrix_init" => matrix_json(b.init),
        "matrix_warmup" => matrix_json(b.warmup),
        "matrix_sample" => matrix_json(b.sample),
        "matrix_merge" => matrix_json(b.merge),
        "drugs" => join(b.drugs, ","),
        "designs" => join(b.designs, ","),
        "nchains" => string(b.nchains),
        "parts" => string(b.parts),
        "nadapt" => string(b.nadapt),
        "draws" => string(b.draws),
        "seglen" => string(b.seglen),
        "segmented" => string(b.segmented),
        "burnin" => string(b.burnin),
        "units" => b.units == 0 ? "" : string(b.units),
    ]
end

function main(args)
    b = build(parse_args(args))
    out = outputs(b)
    if haskey(ENV, "GITHUB_OUTPUT")
        open(ENV["GITHUB_OUTPUT"], "a") do io
            for (k, v) in out
                println(io, k, "=", v)
            end
        end
    else
        for (k, v) in out
            println(k, "=", v)
        end
    end
    println(stderr, "ci_matrix: ", length(b.init), " init/merge, ", length(b.warmup), " warmup, ",
        length(b.sample), " sample jobs (drugs=", join(b.drugs, ","), " designs=", join(b.designs, ","),
        " chains=", b.nchains, " parts=", b.parts,
        " warmup=", b.segmented ? "2 segments of $(b.seglen)" : "1 job", ")")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
