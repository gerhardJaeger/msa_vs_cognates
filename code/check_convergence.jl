using Pkg
Pkg.activate(".")
Pkg.instantiate()

##

using CSV, DataFrames, Pipe

## vstat

vstat_files = []
for (root, dirs, files) in walkdir(".")
    for file in files
        if occursin(".vstat", file)
            push!(vstat_files, joinpath(root, file))
        end
    end
end

##

not_converged = []
for fn in vstat_files
    vdf = @pipe CSV.File(
        fn,
        delim="\t",
        header=2,
        missingstring="NA") |> 
            DataFrame |>
            dropmissing(_, :PSRF)
    if maximum(vdf.PSRF) > 1.1
        push!(not_converged, fn)
    end
end

@info not_converged

##

# fn = not_converged[1]
# vdf = @pipe CSV.File(
#         fn,
#         delim="\t",
#         header=2,
#         missingstring="NA") |> 
#             DataFrame |>
#             dropmissing(_, :PSRF)
