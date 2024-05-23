cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.instantiate()


# Julia Packages
using CSV
using DataFrames
using MCPhyloTree
using ProgressMeter
using Pipe
using ArgCheck
using Base.Threads
using StatsBase
using RCall



# Include external scripts
include("needlemanWunsch.jl")

# Conda and Python setup
using Conda
Conda.add("r-phangorn")
ENV["R_HOME"] = "*"
Pkg.build("RCall")


wl = @pipe CSV.File("../data/lexibank_wordlist.csv") |> 
    DataFrame |>
    dropmissing(_, [:ASJP, :Cognateset_ID]) |>
    filter(x -> x.ASJP != "", _)

# Load datasets
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



"""
Takes two gapped strings and returns the hamming distance between
them. Positions containing a gap in at least one string are ignored.
"""
function sHamming(al)
    ds = []
    for i in 1:size(al, 1)
        s1, s2 = al[i, :]
        if !ismissing(s1) && !ismissing(s2)
            push!(
                ds,
                Int(s1 != s2)
            )
        end
    end
    mean(ds)
end


"""
Takes a pairwise alignment (i.e. a pair of gapped strings with identical length)
as input and returns a matrix representation M as output.
The matrix M is defined as M[i,j] = 1 if x[i] is matched with y[j]
in the alignment, 0 else (where x,y are the two ungapped strings to be aligned).
"""

function algnMatrix(al)
    w1 = filter(x -> !ismissing(x), al[:, 1])
    w2 = filter(x -> !ismissing(x), al[:, 2])
    dm = zeros(Int, length(w1), length(w2))
    i, j = 1, 1
    for k in 1:size(al, 1)
        s1, s2 = al[k, :]
        if !ismissing(s1)
            if !ismissing(s2)
                dm[i, j] += 1
                i += 1
                j += 1
            else
                i += 1
            end
        else
            j += 1
        end
    end
    dm
end


"""
Takes a list of sequences and returns a library in the sense of the
T-Coffee algorithm. A library is a dictionary with sequence pairs
as keys and pairwise alignments in matrix format as columns.
"""


function createLibrary(words; pmiPar::NW=pmiPar)
    library = Dict{Tuple{String, String}, Tuple{Matrix{Int}, Float64}}()
    library_lock = ReentrantLock()
    uwords = unique(words)
    @threads for w1 in uwords
        for w2 in uwords
            if (w2, w1) in keys(library)
                lock(library_lock)
                try
                    x = library[w2, w1]
                    library[w1, w2] = (Matrix{Int}(x[1]'), x[2])
                finally
                    unlock(library_lock)
                end
            else
                al = nwAlign(w1, w2, pmiPar)[1]
                lock(library_lock)
                try
                    library[w1, w2] = (algnMatrix(al), 1 - sHamming(al))
                finally
                    unlock(library_lock)
                end
            end
        end
    end
    return library
end


"""
Takes a list of sequences and returns an extended library in the
sense of the T-Coffee algorithm. An extended library is a dictionary with
sequence pairs as keys and a score matrix as values.
For a pair of sequences x,y and a corresponding score matrix M,
M[i,j] is the score for aligning x[i] with y[j].
"""


function createExtendedLibrary(words; pmiPar::NW=pmiPar)
    uwords = unique(words)
    library = createLibrary(uwords; pmiPar=pmiPar)
    extLibrary = Dict{Tuple{String, String}, Matrix{Float32}}()
    extLibrary_lock = ReentrantLock()

    # Precompute some values to avoid repeated computation in the loop
    word_pairs = [(i, j, uwords[i], uwords[j]) for i in 1:length(uwords), j in 1:length(uwords) if i <= j]

    @threads for (i, j, w1, w2) in word_pairs
        n, m = length.(collect.([w1, w2]))
        dm = zeros(Float32, n, m)
        
        for w3 in words
            a1, s1 = library[w1, w3]
            a2, s2 = library[w3, w2]
            a1, a2 = Matrix{Float32}(a1), Matrix{Float32}(a2)
            dm += (s1 + s2) * (a1 * a2)
        end
    
        lock(extLibrary_lock)
        try
            extLibrary[w1, w2] = dm
            extLibrary[w2, w1] = Matrix{Float32}(dm')
        finally
            unlock(extLibrary_lock)
        end
    end

    return extLibrary
end


"""
Returns the index of gappedString[i] in the
ungapped version thereof.
If gappedString[i] is a gap, returns -1
"""

function pos(alVector, i)
    if ismissing(alVector[i])
        return -1
    end
    return i - sum(ismissing.(alVector[1:i]))
end


"""
Needleman-Wunsch alignment of two aligned blocks b1 and b2,
using the scores in the extended library lib.
"""
function nwBlock(al1, al2, extLibrary)
    # Prepare words1 and words2 using comprehensions
    words1 = [join(filter(x -> !ismissing(x), al1[:, i]), "") for i in 1:size(al1, 2)]
    words2 = [join(filter(x -> !ismissing(x), al2[:, i]), "") for i in 1:size(al2, 2)]

    n, m = size(al1, 1), size(al2, 1)
    dp = zeros(n + 1, m + 1)
    pointers = zeros(Int, n + 1, m + 1)

    # Initialize first row and column of pointers
    pointers[1, 2:end] .= 3  # All deletions
    pointers[2:end, 1] .= 2  # All insertions

    match_cache = Dict{Tuple{Int, Int}, Float64}()

    for i in 2:(n + 1)
        for j in 2:(m + 1)
            insert = dp[i-1, j]
            delet = dp[i, j-1]
            match = dp[i-1, j-1]

            match_val = 0.0
            for k in 1:length(words1)
                for l in 1:length(words2)
                    if !ismissing(al1[i-1, k]) && !ismissing(al2[j-1, l])
                        pos1 = i - 1
                        pos2 = j - 1
                        w1, w2 = words1[k], words2[l]

                        if haskey(match_cache, (pos1, pos2))
                            match_val = match_cache[(pos1, pos2)]
                        else
                            if pos1 <= size(extLibrary[w1, w2], 1) && pos2 <= size(extLibrary[w1, w2], 2)
                                match_val += extLibrary[w1, w2][pos1, pos2]
                                match_cache[(pos1, pos2)] = match_val
                            end
                        end
                    end
                end
            end

            match += match_val

            # Update dp and pointers
            dp[i, j], pointers[i, j] = maximum([(match, 1), (insert, 2), (delet, 3)])
        end
    end

    # Traceback to find the alignment path
    i, j = n + 1, m + 1
    indices = Vector{Tuple{Int, Int}}()
    while i > 1 || j > 1
        p = pointers[i, j]
        if p == 1
            i -= 1
            j -= 1
            pushfirst!(indices, (i, j))
        elseif p == 2
            i -= 1
            pushfirst!(indices, (i, -1))
        else
            j -= 1
            pushfirst!(indices, (-1, j))
        end
    end

    # Initialize alNew array
    alNew = Array{Union{Char, Missing}}(missing, length(indices), size(al1, 2) + size(al2, 2))

    # Fill alNew with aligned sequences
    for (k, (i, j)) in enumerate(indices)
        if i == -1
            x1 = fill(missing, size(al1, 2))
        else
            x1 = al1[i, :]
        end
        if j == -1
            x2 = fill(missing, size(al2, 2))
        else
            x2 = al2[j, :]
        end
        alNew[k, :] = vcat(x1, x2)
    end

    alNew
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
    R"""
    library(phangorn)
    dst = $pmidists
    rownames(dst) <- colnames(dst) <- $languages
    tree <- upgma(dst)
    treeS <- write.tree(tree, file="")
    """
    return ParseNewick(@rget treeS)
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


for db ∈ dbs
    @info "Processing $db"
    d = filter(x -> x.db == db, wl)
    languages = unique(d.Glottocode)
    if length(languages) < 4
        @warn "Skipping $db: Not enough languages"
        continue
    end
    concepts = unique(d.Concepticon_Gloss)
    d_wide = unstack(d, :Glottocode, :Concepticon_Gloss, :ASJP, allowmissing=true, combine=x -> join(unique(x), "-"))
    ln2index = Dict(zip(d_wide.Glottocode, 1:size(d_wide, 1)))
    dMtx = Matrix(d_wide[:, 2:end])
    nconcepts = length(concepts)
    minSim = -sqrt(nconcepts)
    maxSim = (log(nconcepts * (nconcepts - 1) + 1) - 1) * sqrt(nconcepts)
    pmidists = compute_pmidists(languages, dMtx, pmiPar, maxSim, minSim)
    tree = build_tree(pmidists, languages)
    alignments = get_alignments(concepts, d, tree)
    char_mtx = create_character_matrix(concepts, alignments)
    db_name = split(split(db, "/")[end], ".")[1]
    write_nexus_file(char_mtx, db_name)
    write_phylip_file(char_mtx, db_name)
    write_mrbayes_file(db_name)
end


