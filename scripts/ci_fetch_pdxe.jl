# Download the PDXE workbook from the original Springer URL with retry and backoff (CI use).
#
#   TUMOURPKPD_RAW_DIR=<dir> julia scripts/ci_fetch_pdxe.jl [attempts]
#
# Wraps data/download.jl, which verifies the SHA-256 of the file and refuses a mismatch. Retries on any failure
# (network error, truncated download, checksum mismatch) with exponential backoff: 15, 30, 60, 120 s ...
# The data is never cached or uploaded by the workflows; every job that needs it fetches it here or decrypts the
# private stage artifact made by the `prep` job. Standard library only.

include(joinpath(@__DIR__, "..", "data", "download.jl"))

function fetch_with_retry(attempts::Integer)
    for i in 1:attempts
        try
            fetch_all(default_raw_dir(); only = "xlsx")
            return
        catch err
            i == attempts && rethrow()
            wait = 15 * 2^(i - 1)
            println(stderr, "download attempt $i of $attempts failed: ", sprint(showerror, err), "; retrying in $wait s")
            sleep(wait)
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && fetch_with_retry(isempty(ARGS) ? 5 : parse(Int, ARGS[1]))
