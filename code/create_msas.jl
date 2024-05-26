cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.instantiate()

##
# Julia Packages
using CSV
using DataFrames
using MCPhyloTree
using ProgressMeter
using Pipe
using ArgCheck
using Base.Threads
using StatsBase




## Include external scripts
include("alignment_functions.jl")

##

wl = @pipe CSV.File("../data/lexibank_wordlist.csv") |> 
    DataFrame |>
    dropmissing(_, [:ASJP, :Cognateset_ID]) |>
    filter(x -> x.ASJP != "", _)

## Load datasets
dbs = unique(wl.db)

function pmiStar(w1::Union{Missing,String}, w2::Union{Missing,String}, p::NW)
    if ismissing(w1) || ismissing(w2)
        return missing
    end
    v1 = split(w1,"-")
    v2 = split(w2,"-")
    scores = Vector{Float64}(undef, length(v1)*length(v2))
    counter = 1
    for x in v1
        for y in v2
            @inbounds scores[counter] = nw(x, y, p)
            counter += 1
        end
    end
    maximum(scores)
end


function dercPMI(
    i::Int,
    j::Int,
    p::NW,
    dMtx::Matrix{Union{Missing, String}} = dMtx,
    maxSim::Float64 = maxSim,
    minSim::Float64 = minSim,
)
    defined1 = findall(.!ismissing.(dMtx[i, :]))
    defined2 = findall(.!ismissing.(dMtx[j, :]))
    definedBoth = intersect(defined1, defined2)
    nBoth = length(definedBoth)
    if nBoth == 0
        return missing
    end
    dg = Vector{Float64}(undef, nBoth)
    for (k, c) in enumerate(definedBoth)
        dg[k] = pmiStar(dMtx[i, c], dMtx[j, c], p)
    end
    nOffD = length(defined1) * length(defined2) - nBoth
    offDg = Vector{Float64}(undef, nOffD)
    counter = 1
    for k1 in defined1
        w1 = dMtx[i, k1]
        for k2 in defined2
            if k1 != k2
                w2 = dMtx[j, k2]
                @inbounds offDg[counter] = pmiStar(w1, w2, p)
                counter += 1
            end
        end
    end
    ranks = Vector{Float64}(undef, nBoth)
    for k = 1:nBoth
        @inbounds x = dg[k]
        @inbounds ranks[k] = geomean(1 .+ (sum(offDg .> x):sum(offDg .>= x)))
    end
    stc = mean(-log.(ranks ./ (1 + nOffD)))
    sim = (stc - 1) * sqrt(nBoth)
    (maxSim - sim) / (maxSim - minSim)
end




function tCoffee(guide_tree; pmiPar::NW=pmiPar)
    words = [string(split(x.name, ":")[2]) for x in get_leaves(guide_tree)]
    taxa = [string(split(x.name, ":")[1]) for x in get_leaves(guide_tree)]
    extLibrary = createExtendedLibrary(words; pmiPar=pmiPar)
    alHistory = Dict()
    nums = Dict()
    for nd in post_order(guide_tree)
        if length(nd.children) == 0
            w = string(split(nd.name, ":")[2])
            alHistory[nd.num] = reshape(collect(w), :, 1)
            nums[nd.num] = [nd.num]
        elseif length(nd.children) == 1
            alHistory[nd.num] = alHistory[nd.children[1].num]
            nums[nd.num] = [nums[nd.children[1].num]]
        else
            ch1, ch2 = nd.children
            al1 = alHistory[ch1.num]
            al2 = alHistory[ch2.num]
            nums1 = nums[ch1.num]
            nums2 = nums[ch2.num]
            alHistory[nd.num] = nwBlock(al1, al2, extLibrary)
            nums[nd.num] = vcat(nums1, nums2)
        end
    end
    df = DataFrame(permutedims(alHistory[guide_tree.num]), :auto)
    insertcols!(df, 1, :language => taxa)
    df
end


function create_guide_tree(data::DataFrame; tree::GeneralNode=tree)
    words2lang = @pipe data |>
        zip(_.ASJP, _.Glottocode) 
    words = first.(words2lang)
    taxa = last.(values(words2lang))
    unique_taxa = unique(taxa)
    guide_tree = deepcopy(tree)
    for nd in post_order(guide_tree)
        if nd.nchild == 0 && nd.name ∉ unique_taxa && !isroot(nd)
            mother = get_mother(nd)
            remove_child!(mother, nd)
        end
    end
    while guide_tree.nchild == 1
        guide_tree = guide_tree.children[1]
    end
    for nd in post_order(guide_tree)
        if (nd.nchild == 1)
            delete_node!(nd)
        end
    end
    for nd in get_leaves(guide_tree)
        language = nd.name
        nd_words = words[findall(taxa .== language)]
        while length(nd_words) > 1
            nd1 = Node()
            nd2 = Node()
            nd1.name = pop!(nd_words)
            nd1.name = language * ":" * nd1.name
            add_child!(nd, nd1)
            add_child!(nd, nd2)
            nd = nd2
        end
        nd.name = nd_words[1]
        nd.name = language * ":" * nd.name
    end

    for (i, nd) in enumerate(post_order(guide_tree))
        nd.num = i
    end
    guide_tree
end



function get_alignment(data::DataFrame; tree::GeneralNode=tree, pmiPar::NW=pmiPar)
    guide_tree = create_guide_tree(data; tree=tree)
    al = tCoffee(guide_tree)
end

par = CSV.read("../data/pmiParameters.csv", DataFrame)

pmi = CSV.read("../data/pmi.csv", DataFrame)[:,2:end] |> Array

sounds = first.(CSV.read("../data/pmi.csv", DataFrame)[:,1])

pmiDict = Dict{Tuple{Char, Char}, Float64}()

for (i, s1) in enumerate(sounds), (j, s2) in enumerate(sounds)
    pmiDict[s1, s2] = pmi[i,j]
end

pmiPar = NW(
    sounds,
    pmiDict,
    par[1,1],
    par[1,2],
)


function compute_pmidists(languages, dMtx, pmiPar, maxSim, minSim)
    index_pairs = [(i,j) for i in 1:length(languages), j in 1:length(languages) if i < j]
    pmidists = zeros(Union{Float64, Missing}, (length(languages), length(languages)))
    @showprogress @threads for (i,j) in index_pairs
        pmidists[i, j] = pmidists[j, i] = dercPMI(i, j, pmiPar, dMtx, maxSim, minSim)
    end
    pmidists[ismissing.(pmidists)] .= mean(skipmissing(pmidists))
    return pmidists
end

function build_tree(pmidists, languages)
    tree = upgma(convert(Matrix{Float64}, pmidists), convert(Vector{String}, languages))
end


function get_alignments(concepts, d, tree)
    alignments = Dict()
    for c in concepts
        data = filter(x -> x.Concepticon_Gloss == c, d)
        alignments[c] = get_alignment(data; tree=tree)
    end
    return alignments
end

function create_character_matrix(concepts, alignments)
    concept_char_mtc = []
    for c in concepts
        al = alignments[c]
        @pipe al |>
              1 .- ismissing.(Array(_[:, 2:end])) |>
              DataFrame(_, :auto) |>
              insertcols!(_, 1, :language => al.language) |>
              groupby(_, :language) |>
              combine(_, names(_, Not(:language)) .=> maximum) |>
              push!(concept_char_mtc, _)
    end
    char_mtx = outerjoin(concept_char_mtc..., on=:language, makeunique=true)
end

function write_nexus_file(char_mtx, db_name)
    nex = """
    #NEXUS

    Begin data;
    Dimensions ntax=$(size(char_mtx, 1)) nchar = $(size(char_mtx, 2) - 1);
    Format datatype=restriction gap=-;
    MATRIX
    """
    pad = maximum(length.(char_mtx.language)) + 5
    for i in 1:size(char_mtx, 1)
        l = char_mtx.language[i]
        ln = "   " * rpad(l, pad)
        row = join(replace(char_mtx[i, 2:end] |> Vector, missing => "-"))
        ln *= row * "\n"
        nex *= ln
    end
    nex *= """
    ;
    End;
    """
    open(joinpath("mrbayes", "$(db_name)_msa.nex"), "w") do f
        write(f, nex)
    end
end

function write_phylip_file(char_mtx, db_name)
    mkpath("phylip")
    phy = """
    $(size(char_mtx, 1)) $(size(char_mtx, 2) - 1)
    """
    pad = maximum(length.(char_mtx.language)) + 5
    for i in 1:size(char_mtx, 1)
        l = char_mtx.language[i]
        ln = "   " * rpad(l, pad)
        row = join(replace(char_mtx[i, 2:end] |> Vector, missing => "-"))
        ln *= row * "\n"
        phy *= ln
    end
    open(joinpath("phylip", "$(db_name)_msa.phy"), "w") do f
        write(f, phy)
    end
end

function write_mrbayes_file(db_name)
    mb = """#Nexus
    Begin MrBayes;
        execute $(db_name)_msa.nex;
        prset brlenspr = clock:uniform;
        prset clockvarpr = igr;
        lset rates=gamma;
        lset covarion=yes;
        prset clockratepr=exp(1.0);
        lset coding=noabsencesites;
        mcmcp stoprule=no stopval=0.01 filename=output/$(db_name)_msa samplefreq=1000;
        mcmc ngen=10000000 nchains=2 nruns=2 append=no;
        sumt;
        sump;
        q;
    end;
    """
    open(joinpath("mrbayes", "$(db_name)_msa.mb.nex"), "w") do f
        write(f, mb)
    end
end

##
mkpath("upgma_trees")
mkpath("neighbour_joining_trees")
mkpath("msa")

for db ∈ dbs
    @info "Processing $db"
    d = filter(x -> x.db == db, wl)
    languages = unique(d.Glottocode)
    if length(languages) < 10
        @warn "Skipping $db: Not enough languages"
        continue
    end
    concepts = unique(d.Concepticon_Gloss)
    d_wide = unstack(d, :Glottocode, :Concepticon_Gloss, :ASJP, allowmissing=true, combine=x -> join(unique(x), "-"))
    languages = d_wide.Glottocode
    ln2index = Dict(zip(d_wide.Glottocode, 1:size(d_wide, 1)))
    dMtx = Matrix(d_wide[:, 2:end])
    nconcepts = length(concepts)
    minSim = -sqrt(nconcepts)
    maxSim = (log(nconcepts * (nconcepts - 1) + 1) - 1) * sqrt(nconcepts)
    pmidists = compute_pmidists(languages, dMtx, pmiPar, maxSim, minSim)
    tree = build_tree(pmidists, languages)
    njtree = neighbor_joining(convert(Matrix{Float64}, pmidists), convert(Vector{String}, languages))
    open(joinpath("upgma_trees", "$(db)_upgma.tre"), "w") do f
        write(f, newick(tree))
    end
    open(joinpath("neighbour_joining_trees", "$(db)_nj.tre"), "w") do f
        write(f, newick(njtree))
    end
    alignments = get_alignments(concepts, d, tree)
    mkpath(joinpath("msa", db))
    for c in concepts
        CSV.write(joinpath("msa", db, "$c.csv"), coalesce.(alignments[c], "-"))
    end
    char_mtx = create_character_matrix(concepts, alignments)
    write_nexus_file(char_mtx, db)
    write_phylip_file(char_mtx, db)
    write_mrbayes_file(db)
end


