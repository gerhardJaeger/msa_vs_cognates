cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.instantiate()

##

using CSV
using DataFrames
using Pipe
using ProgressMeter

##

lxb = @pipe CSV.File("../data/lexibank_wordlist_pruned.csv") |>
    DataFrame |>
    dropmissing(_, [:ASJP, :Cognateset_ID])

insertcols!(lxb, 1, :id => 1:nrow(lxb))
    

##
lexstat_file = readdir("../data/lexstat/")

lexstat = vcat([CSV.File("../data/lexstat/$(f)") |> DataFrame for f in lexstat_file]...)

##

d = innerjoin(
    select(lxb, Not(:id)),
    select(lexstat, Not(:id)),
    on = [
        :db,
        :Segments => :forms,
        :Concepticon_Gloss => :concept
    ],
    makeunique = true
) |> unique


CSV.write("../data/wordlist_complete.csv", d)

##
dbs = unique(d.db)

##

mkpath("phylip")

@showprogress for db in dbs
    df = filter(x -> x.db == db, d)
    concepts = unique(df.Concepticon_Gloss)
    taxa = unique(df.Glottocode)
    if length(taxa) < 4
        continue
    end
    char_mtc = []
    for c in concepts
        d_c = @pipe df |>
            filter(x -> x.Concepticon_Gloss == c, _) |>
            select(_, :Glottocode, :Cognateset_ID) |>
            unique |>
            insertcols(_, :presence => "1") |>
            unstack(_, :Glottocode, :Cognateset_ID, :presence, fill="0")
        push!(char_mtc, d_c)
    end
    char_mtx = outerjoin(char_mtc..., on = :Glottocode, makeunique=true)
    for col in names(char_mtx)
        char_mtx[!, col] = coalesce.(char_mtx[!, col], "-")
    end
    phy = """
    $(size(char_mtx, 1)) $(size(char_mtx, 2)-1)
    """
    for (i,tx) in enumerate(char_mtx.Glottocode)
        phy *= "$tx $(join(char_mtx[i, 2:end], ""))\n"
    end
    open("phylip/$(db)_cc.phy", "w") do f
        write(f, phy)
    end
end

##

@showprogress for db in dbs
    df = filter(x -> x.db == db, d)
    concepts = unique(df.Concepticon_Gloss)
    taxa = unique(df.Glottocode)
    if length(taxa) < 4
        continue
    end
    char_mtc = []
    for c in concepts
        d_c = @pipe df |>
            filter(x -> x.Concepticon_Gloss == c, _) |>
            select(_, :Glottocode, :lexstatid) |>
            unique |>
            insertcols(_, :presence => "1") |>
            unstack(_, :Glottocode, :lexstatid, :presence, fill="0")
        push!(char_mtc, d_c)
    end
    char_mtx = outerjoin(char_mtc..., on = :Glottocode, makeunique=true)
    for col in names(char_mtx)
        char_mtx[!, col] = coalesce.(char_mtx[!, col], "-")
    end
    phy = """
    $(size(char_mtx, 1)) $(size(char_mtx, 2)-1)
    """
    for (i,tx) in enumerate(char_mtx.Glottocode)
        phy *= "$tx $(join(char_mtx[i, 2:end], ""))\n"
    end
    open("phylip/$(db)_lexstat.phy", "w") do f
        write(f, phy)
    end
end