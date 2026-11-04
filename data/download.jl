# Download the PDXE supplement (Gao et al. 2015, Nat Med 21:1318, doi 10.1038/nm.3954) and verify checksums.
#
#   julia --project=. data/download.jl [dir] [--only=xlsx|pdf]
#
# dir defaults to $TUMOURPKPD_RAW_DIR, else ~/Work/tumour-pkpd/downloads/pdxe if it exists, else data/cache/raw.
# Raw files are never committed (data/cache is gitignored).

using Downloads, SHA

const BASE = "https://static-content.springer.com/esm/art%3A10.1038%2Fnm.3954/MediaObjects/"
# name => (url, sha256 recorded on first download 2026-11-04)
const FILES = Dict(
    "nm3954_S2.xlsx" => (BASE * "41591_2015_BFnm3954_MOESM10_ESM.xlsx",
        "c4b9a6903a4d1f76e3ddca4199039776d56bb99970aa5b7abe4f3abd732a0c6d"),
    "nm3954_supp.pdf" => (BASE * "41591_2015_BFnm3954_MOESM9_ESM.pdf",
        "c2730752c3b492b5f5ad442e946e115f224fcb4ffc0adee882a16ed3ede4a512"),
)

function default_raw_dir()
    haskey(ENV, "TUMOURPKPD_RAW_DIR") && return ENV["TUMOURPKPD_RAW_DIR"]
    legacy = joinpath(homedir(), "Work", "tumour-pkpd", "downloads", "pdxe")
    isdir(legacy) && return legacy
    return joinpath(@__DIR__, "cache", "raw")
end

sha256_file(f) = open(io -> bytes2hex(sha256(io)), f)

function fetch_all(dir::AbstractString = default_raw_dir(); only::Union{Nothing,String} = nothing)
    mkpath(dir)
    for (name, (url, want)) in sort(collect(FILES); by = first)
        only !== nothing && !endswith(name, only) && continue
        f = joinpath(dir, name)
        if isfile(f) && sha256_file(f) == want
            @info "ok (already present, checksum matches)" f
            continue
        end
        @info "downloading" url f
        tmp = f * ".part"
        Downloads.download(url, tmp)
        got = sha256_file(tmp)
        if got != want
            rm(tmp; force = true)
            error("checksum mismatch for $name: got $got, expected $want")
        end
        mv(tmp, f; force = true)
        @info "downloaded and verified" f
    end
    return dir
end

function cli(args)
    dir = nothing
    which = nothing
    for a in args
        startswith(a, "--only=") ? (which = a[8:end]) : (dir = a)
    end
    fetch_all(dir === nothing ? default_raw_dir() : dir; only = which)
end

abspath(PROGRAM_FILE) == (@__FILE__) && cli(ARGS)
