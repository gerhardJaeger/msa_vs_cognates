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

glottologTrees = readdir("../data/glottolog_trees")

datasets = first.(split.(glottologTrees, "_")) |> sort



##


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


function gqd_combined(ds)
    correspondence_trees = open("../data/posterior_trees/$(ds)_combined.trees") do f
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
            gqs[(ds, "combined")] = gqd_combined(ds)
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

sort!(df, :dataset)

##

@df df groupedboxplot(
    :dataset, 
    :gqd, 
    legend = true, 
    group = :method,
    xrotation = 45,
    alpha = 0.5,
    markersize=.5
)

##

@df df groupedviolin(
    :dataset,
    :gqd,
    group=:method,
    xrotation=45,
    legend=:outertop
)

##

savefig("gqd.pdf")

##

@pipe df |>
    groupby(_, [:dataset, :method]) |>
    combine(_, :gqd => median => :gqd) |>
    groupby(_, :method) |>
    combine(_, :gqd => median => :gqd)

##

format_number(x::Float64) = @sprintf("%.3f", x)
format_number(x::Missing) = missing  # Handle missing values


@pipe df |>
    groupby(_, [:dataset, :method]) |>
    combine(_, :gqd => median => :gqd) |>
    unstack(_, :method, :gqd) |>
    transform(_, 
        :correspondences => (x -> format_number.(x)) => :correspondences,
        :cognates => (x -> format_number.(x)) => :cognates,
        :combined => (x -> format_number.(x)) => :combined
    ) |>
    select(_, :dataset, :cognates, :correspondences, :combined) |>
    pretty_table(_, backend=Val(:latex))
    

##

@pipe df |>
    groupby(_, :method) |>
    combine(_, :gqd => median => :gqd)