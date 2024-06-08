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

datasets = @pipe readdir("upgma_trees/") |>
    split.(_, "_") |>
    first.(_) |>
    unique |>
    sort


##
function gqd_upgma(ds)
    msa_tree = "upgma_trees/$(ds)_upgma.tre"
    glot_tree = "../data/glottolog_trees/$(ds)_glottolog.tre"
    output = read(`qdist $msa_tree $glot_tree`, String)
    gqd = 1 - parse(Float64, split(output, "\t")[end-2])
    return gqd
end

##

function gqd_nj(ds)
    msa_tree = "neighbour_joining_trees/$(ds)_nj.tre"
    glot_tree = "../data/glottolog_trees/$(ds)_glottolog.tre"
    output = read(`qdist $msa_tree $glot_tree`, String)
    gqd = 1 - parse(Float64, split(output, "\t")[end-2])
    return gqd
end


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
    push!(results, (
            ds=ds,
            gqd_upgma=gqd_upgma(ds),
            gqd_nj=gqd_nj(ds),
            gqd_cc=gqd_cc(ds),
            gqd_msa=gqd_msa(ds)
        )
    )
end

results_df = filter(x -> !isnan(x.gqd_upgma) && !isnan(x.gqd_nj) && !isnan(x.gqd_cc) && !isnan(x.gqd_msa), DataFrame(results))

##

@pipe results_df |>
    stack(_, [:gqd_upgma, :gqd_nj, :gqd_cc, :gqd_msa]) |>
    @df _ dotplot(:variable, :value, group=:variable, xlabel="Method", ylabel="QD", legend=false)

##

@pipe results_df |>
    stack(_, [:gqd_upgma, :gqd_nj, :gqd_cc, :gqd_msa]) |>
    @df _ boxplot(:variable, :value, group=:variable, xlabel="Method", ylabel="QD", legend=false)
@pipe results_df |>
    stack(_, [:gqd_upgma, :gqd_nj, :gqd_cc, :gqd_msa]) |>
    @df _ dotplot!(:variable, :value, group=:variable, xlabel="Method", ylabel="QD", legend=false)


##
@pipe results_df |>
    stack(_, [:gqd_upgma, :gqd_nj, :gqd_cc, :gqd_msa]) |>
    groupby(_, :variable) |>
    combine(_, :value => mean => :mean, :value => std => :std)

#     Row │ variable   mean      std      
#     │ String     Float64   Float64  
# ─────┼───────────────────────────────
#   1 │ gqd_upgma  0.249513  0.174721
#   2 │ gqd_nj     0.208609  0.164862
#   3 │ gqd_cc     0.211304  0.164284
#   4 │ gqd_msa    0.236392  0.156864


##


# Assuming df is your DataFrame and results_df is already defined
# Extract min and max for the axes
min_val = min(minimum(results_df.gqd_cc), minimum(results_df.gqd_nj))
max_val = max(maximum(results_df.gqd_cc), maximum(results_df.gqd_nj))

# Set plot dimensions to be square
plot_width = 600
plot_height = 600

# Create the plot with manually set limits and aspect ratio 1:1
@df results_df scatter(:gqd_cc, :gqd_nj, xlabel="QD CC", ylabel="QD NJ", legend=false, 
                       xlim=(min_val, max_val), ylim=(min_val, max_val), 
                       size=(plot_width, plot_height), aspect_ratio=1)

# Add the y=x line
plot!([min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, label="y=x")
