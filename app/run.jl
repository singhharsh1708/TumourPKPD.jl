# Launch the explorer on localhost.
#
#   julia --project=TumourPKPD.jl/app -t 2 TumourPKPD.jl/app/run.jl [fits_dir] [--port 8081] [--demo]
#
# fits_dir   directory with <drug>.jld2 files written by `save_fit` (default: ../fits). A drug without a file
#            falls back to a synthetic DEMO fit (written to app/demo_fits), clearly labelled in the UI.
# --demo     ignore fits_dir and use demo fits for every drug.
# --port N   default 8081.

include(joinpath(@__DIR__, "explorer.jl"))
using .Explorer, .Explorer.ExplorerCore, Bonito

function parse_args(args)
    dir = joinpath(@__DIR__, "..", "fits"); port = 8081; demo = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--port"
            port = parse(Int, args[i+1]); i += 2
        elseif a == "--demo"
            demo = true; i += 1
        else
            dir = a; i += 1
        end
    end
    return (; dir = abspath(dir), port, demo)
end

opts = parse_args(ARGS)
demodir = joinpath(@__DIR__, "demo_fits")
fits = ExplorerCore.load_fits(opts.demo ? demodir : opts.dir; demo_dir = demodir)
for d in sort(collect(keys(fits)))
    println(rpad(d, 12), fits[d].label, "  (", fits[d].path, ")")
end

app = Explorer.build_app(fits)
server = Bonito.Server(app, "127.0.0.1", opts.port)
println("\nTumourPKPD explorer: http://127.0.0.1:$(opts.port)   (Ctrl-C to stop)")
try
    wait(server.server)
catch err
    err isa InterruptException || rethrow()
finally
    close(server)
end
