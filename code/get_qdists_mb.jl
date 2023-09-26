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
plotlyjs()

##

glottologTrees = readdir("../data/glottolog_trees")

datasets = first.(split.(glottologTrees, "_"))

##



ds = rand(datasets)


function gqd_correspondences(ds)
    correspondence_trees = open("../data/posterior_trees/$(ds)_correspondences.trees") do f
        readlines(f)
    end
    glot_tree = "../data/glottolog_trees/$(ds)_glottolog.tre"
    gqd = Vector{Float64}(undef, length(correspondence_trees))
    for (i, tree1) in enumerate(correspondence_trees)
        fn1 = tempname()
        write(fn1, tree1)
        output = read(`qdist $fn1 $glot_tree`, String)
        rm(fn1)
        gqd[i] = 1 - parse(Float64, split(output, "\t")[end-2])
    end
    return gqd
end

##


function gqd_cognates(ds)
    correspondence_trees = open("../data/posterior_trees/$(ds)_cognate_classes_cc.trees") do f
        readlines(f)
    end
    glot_tree = "../data/glottolog_trees/$(ds)_glottolog.tre"
    gqd = Vector{Float64}(undef, length(correspondence_trees))

    for (i, tree1) in enumerate(correspondence_trees)
        fn1 = tempname()
        write(fn1, tree1)
        output = read(`qdist $fn1 $glot_tree`, String)
        rm(fn1)
        gqd[i] = 1 - parse(Float64, split(output, "\t")[end-2])
    end
    return gqd
end

##

gqs = Dict{Tuple{String, String}, Vector{Float64}}()

for ds in datasets
    try
        begin
            gqs[(ds, "correspondences")] = gqd_correspondences(ds)
            gqs[(ds, "cognates")] = gqd_cognates(ds)
        end
    catch e
    end
end

##

df = DataFrame(
    dataset = String[],
    method = String[],
    gqd = Float64[]
)

for (ds, method) in keys(gqs)
    df_ = DataFrame(
        dataset = ds,
        method = method,
        gqd = gqs[(ds, method)]
    )
    append!(df, df_)
end

##

@df df boxplot(
    :dataset, 
    :gqd, 
    legend = false, 
    group = :method,
    xrotation = 45,
    alpha = 0.5,
)

##

df_correspondences = df[df.method .== "correspondences", :]
df_cognates = df[df.method .== "cognates", :]

@df df_correspondences violin(
    string.(:dataset), 
    :gqd, 
    side=:right, 
    linewidth=0, 
    label="correspondences", 
    xrotation=45,
    legend=:outerbottom
)
@df df_cognates violin!(
    string.(:dataset), 
    :gqd, 
    side=:left, 
    linewidth=0, 
    label="cognates", 
    xrotation=45
)


@df df_correspondences dotplot!(
    string.(:dataset), 
    :gqd, 
    side=:right, 
    marker=(:black,stroke(0)), 
    label="",
    markersize=0.2
)
@df df_cognates dotplot!(
    string.(:dataset), 
    :gqd, 
    side=:left, 
    marker=(:black,stroke(0)), 
    label="",
    markersize=0.2
)
title!("Generalized Quartet distances to Glottolog trees")

##
savefig("gqd.pdf")