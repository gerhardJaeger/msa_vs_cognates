using Pkg
Pkg.activate(".")
Pkg.instantiate()

##

using CSV
using DataFrames
using Pipe
using StatsPlots
using Statistics
##


posterior_files = []
for (root, dirs, files) in walkdir(".")
    for file in files
        if occursin("1.p", file) && occursin("_", file)
            push!(posterior_files, joinpath(root, file))
        end
    end
end

##

posterior_alphas_ = []

for fn in posterior_files
    @info fn
    ds, chartype = split(split(split(fn, "/")[end], ".")[1], "_")
    alphas = @pipe CSV.File(
        fn,
        delim="\t",
        header=2,
        missingstring="NA") |> 
            DataFrame |>
            _[(nrow(_)÷ 2):end, :] |>
            select(_, "alpha") |>
            Array |> vec
    push!(posterior_alphas_, [(ds=ds, chartype=chartype, alpha=α) for α in alphas])
end

posterior_alphas = DataFrame(vcat(posterior_alphas_...))

##

@pipe posterior_alphas |>
    groupby(_, [:ds, :chartype]) |>
    combine(_, :alpha => mean => :alpha) |>
    @df _ boxplot(:chartype, :alpha, group=:chartype, legend=false, ylabel="alpha", xlabel="chartype", title="Posterior alpha")

