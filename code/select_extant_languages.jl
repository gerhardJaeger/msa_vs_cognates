cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.instantiate()

##

using CSV
using DataFrames
using HTTP
using ZipFile
using Pipe


##


url = "https://zenodo.org/record/10804582/files/glottolog/glottolog-cldf-v5.0.zip?download=1"
zip_path = "../data/glottolog-cldf-v5.0.zip"
HTTP.download(url, zip_path)

reader = ZipFile.Reader(zip_path)

for f in reader.files
    global languages
    if occursin("languages.csv", f.name)
        languages = CSV.File(IOBuffer(read(f, String))) |> DataFrame
    end
end
rm(zip_path)

##
wl = @pipe CSV.File("../data/lexibank_wordlist.csv") |> 
    DataFrame |>
    dropmissing(_, :ASJP) |>
    filter(x -> x.ASJP != "", _)

##

@pipe wl |>
    unique(_, :Language_ID) |>
    filter(x -> occursin("OLD", uppercase(x.Language_ID)) || occursin("MIDDLE", uppercase(x.Language_ID)) || occursin("ANCIENT", uppercase(x.Language_ID)) || occursin("LATIN", uppercase(x.Language_ID)) || occursin("SANSKRIT", uppercase(x.Language_ID)) || occursin("PALI", uppercase(x.Language_ID)) || occursin("AVESTAN", uppercase(x.Language_ID)), _) |> _.Language_ID |> unique



filter(x -> occursin("Latin", x.Name), dropmissing(languages, :Language_ID))