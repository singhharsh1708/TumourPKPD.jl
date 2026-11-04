# Guard run before every actions/upload-artifact step in fits.yml.
#
#   julia scripts/ci_check_upload.jl --kind=stage --max-mb=N file.enc...
#   julia --project=. scripts/ci_check_upload.jl --kind=fit --max-mb=N <drug>_<design>.jld2 <drug>_<design>_diagnostics.md
#
# The repository is public and artifacts are downloadable, so nothing derived from PDXE may be uploaded in clear
# except the merged fit.
#
# kind=stage  (init / warm / chain state and the cleaned tables handed between jobs): every file must be a `.enc`
#             file starting with the OpenSSL "Salted__" header (i.e. it really went through `openssl enc`) and be
#             at most N MB. A plain `.jld2`, `.csv`, ... is refused.
# kind=fit    (merged fit, in clear): only `<drug>_<design>.jld2` and `.md` files, at most N MB, no PDXE column names
#             in the bytes, and for the `.jld2` a structural check of the saved PosteriorDraws (needs `--project=.`):
#               - the file holds exactly one key, "pd";
#               - the struct has exactly the expected fields (a new field must be reviewed and added here);
#               - `diagnostics` has only allow-listed keys, with the expected kinds of values: no matrices, no
#                 per-unit coordinate vectors (z-coordinates, baselines), no raw chain arrays (top-level float vectors
#                 are limited to the 64-entry map_lps / mode_lps; dictionaries may hold no float vectors);
#               - baselines are latent (their draws are not constant): constant columns would be exactly the
#                 observed day-0 volumes.
#             What the merged fit does contain (and this check cannot judge): per-PDX posterior draws of lambda0, k2 and
#             of the latent baselines (posterior medians land within about 10% of the observed day-0 volumes, they
#             are not equal to them), model ids, tumour types, and per-curve last-observation days in
#             diagnostics["cutoffs"].
# Exit status 1 on any violation. Stage kind is standard library only.

const FIT_EXT = (".jld2", ".md")
const MARKERS = ["volume_mm3", "pct_tvol_diff", "body_weight_g", "Days Post T0", "% TVol Difference",
    "Volume (mm3)", "tumor_type_raw", "treatment_raw"]

const PD_FIELDS = [:drug, :models, :doses, :pk, :cref, :μ_λ0, :ω_λ0, :μ_k2, :ω_k2, :logλ1, :logk1, :σ, :λ0, :k2,
    :w0_control, :w0_treated, :w0_control_draws, :w0_treated_draws, :diagnostics]

# top-level diagnostics keys. kind: :scalar (number, string, bool), :strings, :bools, :ints, :floats (short vector
# of sampler summaries), :dictlist (short vector of dicts, checked like :dict), :dict (checked recursively: scalars, strings, short integer/string vectors, nested dicts)
const DIAG_KEYS = Dict{String,Symbol}(
    "baseline_sd" => :scalar, "chain_start_modes" => :ints, "cross_mode_rhat" => :scalar, "cutoffs" => :cutoffs,
    "draws_per_chain" => :scalar, "dropped_units" => :strings, "fit_window_day" => :scalar, "k2_centered" => :bools,
    "k2_centered_models" => :strings, "latent_w0" => :scalar, "lod" => :scalar, "map_lps" => :floats,
    "max_observed_day" => :scalar, "max_rhat" => :scalar, "max_tree_depth_hits" => :scalar,
    "mean_acceptance" => :scalar, "mean_n_steps" => :scalar, "mean_step_size" => :scalar,
    "mean_tree_depth" => :scalar, "median_followup_day" => :scalar, "min_ess_bulk" => :scalar,
    "min_ess_tail" => :scalar, "min_ebfmi" => :scalar,
    "mode_lps" => :floats, "n_distinct_modes" => :scalar, "n_divergences" => :scalar, "nchains" => :scalar,
    "nu" => :scalar, "rhat_by_warmup_chain" => :scalar, "runtime_s" => :scalar, "seed" => :scalar,
    "fingerprint" => :scalar, "n_start_modes" => :scalar, "excluded_modes" => :dictlist,
    "settings" => :dict, "split" => :dict, "stage" => :dict, "train_models" => :strings, "tumour_types" => :dict)
const MAX_SHORT = 64          # longest numeric vector allowed in :floats / dict values

# byte-level substring search (Base has no substring search for byte vectors)
function hasbytes(needle, hay::Vector{UInt8})
    n = length(needle)
    n == 0 && return true
    first = needle[1]
    i = 1
    last = length(hay) - n + 1
    while i <= last
        j = findnext(==(first), hay, i)
        (j === nothing || j > last) && return false
        all(k -> hay[j + k - 1] == needle[k], 1:n) && return true
        i = j + 1
    end
    false
end

function check_stage(f, max_mb, bad)
    endswith(f, ".enc") || push!(bad, "$f: stage files must be encrypted (.enc)")
    sz = filesize(f)
    sz <= max_mb * 1024^2 || push!(bad, "$f: $(round(sz / 1024^2; digits = 1)) MB exceeds $max_mb MB")
    head = open(io -> read(io, 8), f)
    String(head) == "Salted__" || push!(bad, "$f: not an OpenSSL salted file (missing Salted__ header)")
end

function check_diag_value(path, v, bad)
    if v isa AbstractDict
        for (k, x) in v
            check_diag_value("$path.$k", x, bad)
        end
    elseif v isa AbstractMatrix || (v isa AbstractArray && ndims(v) > 1)
        push!(bad, "diagnostics $path is a matrix $(size(v))")
    elseif v isa AbstractVector
        if eltype(v) <: AbstractDict
            for (i, x) in enumerate(v)
                check_diag_value("$path[$i]", x, bad)
            end
        elseif eltype(v) <: AbstractString || eltype(v) <: Integer || eltype(v) <: Bool
            length(v) <= 5000 || push!(bad, "diagnostics $path is a long vector ($(length(v)))")
        elseif eltype(v) <: Real
            push!(bad, "diagnostics $path is a floating-point vector of length $(length(v)); not allowed inside dictionaries")
        else
            push!(bad, "diagnostics $path has unexpected element type $(eltype(v))")
        end
    elseif v isa Tuple
        (length(v) <= 8 && all(x -> x isa Real, v)) || push!(bad, "diagnostics $path is a tuple of unexpected content")
    elseif !(v isa Union{Number,AbstractString,Bool,Nothing,Symbol})
        push!(bad, "diagnostics $path has unexpected type $(typeof(v))")
    end
end

function check_fit_structure(f, bad)
    @eval using TumourPKPD, JLD2
    M = TumourPKPD
    root = normpath(joinpath(@__DIR__, ".."))
    isdefined(M, :PosteriorDraws) || Base.include(M, joinpath(root, "src", "hierarchical.jl"))
    ks = Base.invokelatest(() -> JLD2.jldopen(io -> collect(keys(io)), f))
    ks == ["pd"] || push!(bad, "$f: top-level keys $(ks), expected exactly [\"pd\"]")
    pd = Base.invokelatest(M.load_fit, f)
    flds = collect(fieldnames(M.PosteriorDraws))
    flds == PD_FIELDS || push!(bad, "$f: PosteriorDraws fields changed: $(flds) (review, then update PD_FIELDS)")
    dg = pd.diagnostics
    for (k, v) in dg
        if !haskey(DIAG_KEYS, k)
            push!(bad, "$f: diagnostics key $(repr(k)) is not on the allow-list (review, then add it to DIAG_KEYS)")
            continue
        end
        kind = DIAG_KEYS[k]
        if kind == :scalar
            v isa Union{Number,AbstractString,Bool,Nothing} || push!(bad, "diagnostics[$k] should be a scalar, is $(typeof(v))")
        elseif kind == :cutoffs
            v isa AbstractDict && all(kk -> occursin(r"^[^|]+\|(treated|control)$", kk), keys(v)) &&
                all(x -> x isa Real, values(v)) || push!(bad, "diagnostics[cutoffs] is not a \"<model>|treated/control\" => day map")
        elseif kind == :strings
            v isa AbstractVector{<:AbstractString} || push!(bad, "diagnostics[$k] should be a vector of strings")
        elseif kind == :bools
            v isa AbstractVector{Bool} || push!(bad, "diagnostics[$k] should be a vector of Bool")
        elseif kind == :ints
            v isa AbstractVector{<:Integer} || push!(bad, "diagnostics[$k] should be a vector of integers")
        elseif kind == :floats
            (v isa AbstractVector{<:Real} && length(v) <= MAX_SHORT) || push!(bad, "diagnostics[$k] should be a short numeric vector")
        elseif kind == :dict
            v isa AbstractDict || push!(bad, "diagnostics[$k] should be a Dict")
        elseif kind == :dictlist
            (v isa AbstractVector && all(x -> x isa AbstractDict, v) && length(v) <= MAX_SHORT) ||
                push!(bad, "diagnostics[$k] should be a short vector of Dicts")
        end
        k in ("settings", "split", "stage", "tumour_types", "excluded_modes") && check_diag_value(k, v, bad)
    end
    # latent baselines only: constant draws would be the observed day-0 volumes themselves
    get(dg, "latent_w0", false) === true || push!(bad, "$f: latent_w0 is not true; fixed baselines are the observed day-0 volumes")
    for (nm, W) in (("w0_control_draws", pd.w0_control_draws), ("w0_treated_draws", pd.w0_treated_draws))
        size(W, 1) > 1 || push!(bad, "$f: $nm has a single draw")
        any(j -> all(==(W[1, j]), @view(W[:, j])), axes(W, 2)) && push!(bad, "$f: $nm has constant columns (exact observed volumes)")
        size(W, 2) == length(pd.models) || push!(bad, "$f: $nm has $(size(W, 2)) columns for $(length(pd.models)) models")
    end
end

function check(files, kind, max_mb)
    bad = String[]
    for f in files
        if !isfile(f)
            push!(bad, "$f: missing")
            continue
        end
        if kind == "stage"
            check_stage(f, max_mb, bad)
            continue
        end
        ext = lowercase(splitext(f)[2])
        ext in FIT_EXT || push!(bad, "$f: extension $ext is not allowed for a merged fit")
        sz = filesize(f)
        sz <= max_mb * 1024^2 || (push!(bad, "$f: $(round(sz / 1024^2; digits = 1)) MB exceeds $max_mb MB"); continue)
        bytes = read(f)
        for m in MARKERS
            hasbytes(codeunits(m), bytes) && push!(bad, "$f: contains data column name $(repr(m))")
        end
        ext == ".jld2" && check_fit_structure(f, bad)
    end
    bad
end

function main(args)
    max_mb = 500.0
    kind = ""
    files = String[]
    for a in args
        if (m = match(r"^--max-mb=([0-9.]+)$", a)) !== nothing
            max_mb = parse(Float64, m[1])
        elseif (m = match(r"^--kind=(stage|fit)$", a)) !== nothing
            kind = m[1]
        else
            push!(files, a)
        end
    end
    isempty(kind) && error("--kind=stage|fit is required")
    isempty(files) && error("no files given")
    bad = check(files, kind, max_mb)
    if isempty(bad)
        println("upload check ok (", kind, "): ", length(files), " file(s): ", join(basename.(files), ", "))
    else
        foreach(b -> println(stderr, "upload check FAILED: ", b), bad)
        exit(1)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
