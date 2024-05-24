cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.instantiate()

##

using CSV
using DataFrames
using StatsPlots
using ProgressMeter
using Statistics
using Pipe
using Printf
using PrettyTables
plotlyjs()

##

datasets = @pipe readdir("phylip") |>
    split.(_, "_") |>
    first.(_) |>
    unique |>
    sort


##


function gqd_msa(ds)
    msa_tree = "phylip/$(ds)_msa.ml.tre"
    glot_tree = "../data/glottolog_trees/$(ds)_glottolog.tre"
    output = read(`qdist $msa_tree $glot_tree`, String)
    gqd = 1 - parse(Float64, split(output, "\t")[end-2])
    return gqd
end

##

function gqd_cc(ds)
    msa_tree = "phylip/$(ds)_cc.ml.tre"
    glot_tree = "../data/glottolog_trees/$(ds)_glottolog.tre"
    output = read(`qdist $msa_tree $glot_tree`, String)
    gqd = 1 - parse(Float64, split(output, "\t")[end-2])
    return gqd
end


##
results = []

for ds in datasets
    try
        push!(results, (ds=ds, gqd_msa=gqd_msa(ds), gqd_cc=gqd_cc(ds)))
    catch e
    end
end

results_df = filter(x -> !isnan(x.gqd_msa) && !isnan(x.gqd_cc), DataFrame(results))

##

@df results_df scatter(:gqd_msa, :gqd_cc, xlabel="MSA", ylabel="CC", legend=false)
plot!(0:0.1:1, 0:0.1:1, color=:black, linestyle=:dash, linewidth=2)

##

@pipe results_df |>
    stack(_, [:gqd_msa, :gqd_cc]) |>
    @df _ dotplot(:variable, :value, group=:variable, xlabel="Method", ylabel="QD", legend=false)