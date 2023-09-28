using Pkg
Pkg.activate(".")
Pkg.instantiate()

##

using CSV, DataFrames, Pipe

## mcmc

mcmc_files = []
for (root, dirs, files) in walkdir(".")
    for file in files
        if occursin(".mcmc", file)
            push!(mcmc_files, joinpath(root, file))
        end
    end
end

##

not_converged = []
for fn in mcmc_files
    vdf = @pipe CSV.File(
        fn,
        delim="\t",
        header=6,
        missingstring="NA") |> 
            DataFrame |>
            dropmissing(_, "AvgStdDev(s)")
    if vdf[end, "AvgStdDev(s)"] > 0.01
        push!(not_converged, fn)
    end
end

for x in not_converged
    @info x
end


##
