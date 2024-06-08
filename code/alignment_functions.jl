using Base.Threads
using ArgCheck

mutable struct NW{T}
    alphabet::Vector{T}
    s::Dict{Tuple{T,T},Float64}
    gp1::Float64 # must be non-negative!
    gp2::Float64 # must be non-negative!
end


function nwAlign!(
    dp::Array{Float64,3},
    pt::Array{Int,3},
    w1::Union{AbstractString, Vector{T}},
    w2::Union{AbstractString, Vector{T}},
    p::NW{T},
) where {T}
    @argcheck all([s ∈ p.alphabet for s in w1])
    @argcheck all([s ∈ p.alphabet for s in w2])
    n, m = length(w1), length(w2)
    @argcheck size(dp) <= (n + 1, m + 1, 3)
    @argcheck size(pt) <= (n + 1, m + 1, 3)
    fill!(dp, -Inf)
    fill!(pt, 0)

    dp[2, 2, 1] = p.s[w1[1], w2[1]]
    dp[2, 1, 2] = -p.gp1
    dp[1, 2, 3] = -p.gp1
    pt[2, 2, 1] = -1
    pt[2, 1, 2] = -1
    pt[1, 2, 3] = -1

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1], pt[i, j, 1] =
                findmax(dp[i-1, j-1, :])
            dp[i, j, 1] += p.s[w1[i-1], w2[j-1]]
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2], pt[i, j, 2] =
                findmax(dp[i-1, j, :] + [-p.gp1, -p.gp2, -Inf])
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3], pt[i, j, 3] =
                findmax(dp[i, j-1, :] + [-p.gp1, -Inf, -p.gp2])
        end
    end
    llMax, finalpt = findmax(dp[n+1, m+1, :])
    increments = [
        1 1
        1 0
        0 1
    ]
    path = Int[]
    let (i, j) = size(pt)
        current = finalpt
        while current != -1
            pushfirst!(path, current)
            (i, j, current) = (
                i - increments[current, 1],
                j - increments[current, 2],
                pt[i, j, current],
            )
        end
    end

    a = Union{T, Missing}[]
    b = Union{T, Missing}[]
    let (i, j) = (1, 1)
        for x in path
            if x == 1
                push!(a, w1[i])
                i += 1
                push!(b, w2[j])
                j += 1
            elseif x == 2
                push!(a, w1[i])
                i += 1
                push!(b, missing)
            else
                push!(a, missing)
                push!(b, w2[j])
                j += 1
            end
        end
    end
    return (alignment = convert(Matrix{Union{Missing, T}},[a b]), score = llMax::Float64)
end

#---

function nwAlign(
    w1::Union{AbstractString, Vector{T}},
    w2::Union{AbstractString, Vector{T}},
    p::NW{T},
) where {T}
    dp = Array{Float64,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    pt = Array{Int,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    nwAlign!(dp, pt, w1, w2, p)
end

#---

function nw!(
    dp::Array{Float64,3},
    w1::Union{AbstractString, Vector},
    w2::Union{AbstractString, Vector},
    p::NW,
)
    @argcheck all([s ∈ p.alphabet for s in w1])
    @argcheck all([s ∈ p.alphabet for s in w2])
    n, m = length(w1), length(w2)
    @argcheck size(dp) <= (n + 1, m + 1, 3)
    fill!(dp, -Inf)

    dp[2, 2, 1] = p.s[w1[1], w2[1]]
    dp[2, 1, 2] = -p.gp1
    dp[1, 2, 3] = -p.gp1

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1] =
                maximum(dp[i-1, j-1, :])
            dp[i, j, 1] += p.s[w1[i-1], w2[j-1]]
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2] =
                maximum(dp[i-1, j, :] + [-p.gp1, -p.gp2, -Inf])
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3] =
                maximum(dp[i, j-1, :] + [-p.gp1, -Inf, -p.gp2])
        end
    end
    maximum(dp[n+1, m+1, :])
end

function nw(
    w1::Union{AbstractString, Vector{T}},
    w2::Union{AbstractString, Vector{T}},
    p::NW{T}
) where {T}
    dp = Array{Float64,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    nw!(dp, w1, w2, p)
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
    words1 = []
    for i in 1:size(al1, 2)
        push!(
            words1,
            join(filter(x -> !ismissing(x), al1[:, i]))
        )
    end
    words2 = []
    for i in 1:size(al2, 2)
        push!(
            words2,
            join(filter(x -> !ismissing(x), al2[:, i]))
        )
    end
    n, m = size(al1, 1), size(al2, 1)
    dp = zeros(n + 1, m + 1)
    pointers = zeros(Int, n + 1, m + 1)
    pointers[1, 2:end] .= 3
    pointers[2:end, 1] .= 2
    for i in 2:(n+1), j in 2:(m+1)
        insert = dp[i-1, j]
        delet = dp[i, j-1]
        match = dp[i-1, j-1]
        for k in 1:length(words1), l in 1:length(words2)
            if !ismissing(al1[i-1, k]) && !ismissing(al2[j-1, l])
                pos1 = pos(al1[:, k], i - 1)
                pos2 = pos(al2[:, l], j - 1)
                w1, w2 = words1[k], words2[l]
                match += extLibrary[w1, w2][pos1, pos2]
            end
        end
        dp[i, j] = maximum([insert, delet, match])
        pointers[i, j] = argmax(([match, insert, delet]))
    end
    i, j = size(dp)
    indices = []
    while maximum([i, j]) > 1
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
    alNew = Array{Union{Char,Missing}}(
        missing,
        length(indices),
        size(al1, 2) + size(al2, 2))

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
