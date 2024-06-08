cd(@__DIR__)
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

##

using CSV
using DataFrames
using Pipe

##

lxb = CSV.File("../data/lexibank_wordlist.csv") |> DataFrame

leipzig_collection = CSV.File(download("https://github.com/rbouckaert/global-language-tree-pipeline/raw/master/TreeSetAnalysisScripts/all_tips_by_year6ALL.csv")) |> DataFrame

##

lxb_pruned = @pipe lxb |>
    filter(x -> x.Glottocode ∈ leipzig_collection.Glottocode, _)

CSV.write("../data/lexibank_wordlist_pruned.csv", lxb_pruned)