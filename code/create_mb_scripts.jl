using Pkg
Pkg.activate(".")
Pkg.instantiate()

##

datasets = readdir(
    "../data/correspondences_nexus",
    join=true
)

mkpath("mrbayes/output")

for nexus_file in datasets
    nm = split(split(nexus_file, "/")[end], ".")[1]*"_correspondences"
    mb = """
    #Nexus
    Begin MrBayes;
        execute ../$nexus_file;
        prset brlenspr = clock:uniform;
        lset rates=gamma;
        prset brlenspr = clock:uniform;
        prset clockvarpr=igr;
        lset coding=all;
        mcmcp stoprule=yes burninfrac=0.25 stopval=0.01 filename=output/$(nm) samplefreq=1000 printfreq=1000;
        mcmc ngen=100000000 nchains=4 nruns=4;
        sumt;
        sump;
        q;
    end;
    """
    write("mrbayes/$nm.mb.nex", mb)
end

##


datasets = readdir(
    "../data/cognate_classes_nexus",
    join=true
)



for nexus_file in datasets
    nm = split(split(nexus_file, "/")[end], ".")[1]*"_cognate_classes"
    mb = """
    #Nexus
    Begin MrBayes;
        execute ../$nexus_file;
        prset brlenspr = clock:uniform;
        lset rates=gamma;
        prset brlenspr = clock:uniform;
        prset clockvarpr=igr;
        lset coding=all;
        mcmcp stoprule=yes burninfrac=0.25 stopval=0.01 filename=output/$(nm)_cc samplefreq=1000 printfreq=1000;
        mcmc ngen=100000000 nchains=4 nruns=4;
        sumt;
        sump;
        q;
    end;
    """
    write("mrbayes/$nm.mb.nex", mb)
end

##

mkpath("../data/combined/")

for ds in datasets
    ds_name = split(split(ds, "/")[end], ".")[1]
    phylip_correspondences = open("../data/correspondences_phylip/$ds_name.phy") do f
        readlines(f)
    end
    n_correspondences = parse(Int, split(phylip_correspondences[1])[2])
    phylip_cognates = open("../data/cognate_classes_phylip/$ds_name.phy") do f
        readlines(f)
    end
    n_cognates = parse(Int, split(phylip_cognates[1])[2])
    correspondences_dict = Dict(split.(phylip_correspondences)[2:end])
    cognates_dict = Dict(split.(phylip_cognates)[2:end])
    taxa = string.(intersect(keys(correspondences_dict), keys(cognates_dict)))
    n_taxa = length(taxa)
    n_characters = n_correspondences + n_cognates
    pad = maximum(length.(taxa))+5
    nex = """
    #Nexus
    BEGIN DATA;
    DIMENSIONS ntax=$(n_taxa) nchar = $(n_characters);
    FORMAT DATATYPE=Restriction GAP=? MISSING=- interleave=no;
    MATRIX

    """

    for l in taxa
        nex *= rpad(l, pad) * " " * correspondences_dict[l] * cognates_dict[l] * "\n"
    end
    nex *= """;
    END
    """
    nexus_file = "../data/combined/$ds_name.nex"
    write(nexus_file, nex)
    nm = ds_name*"_combined"
    mb = """
        #Nexus
        Begin MrBayes;
            execute ../$nexus_file;
            prset brlenspr = clock:uniform;
            prset clockvarpr = igr;
            lset rates=gamma;
            mcmcp stoprule=yes burninfrac=0.25 stopval=0.01 filename=output/$(nm) samplefreq=1000 printfreq=1000 append=no;
            mcmc ngen=1000000000 nchains=4 nruns=2;
            sumt;
            sump;
            q;
        end;
        """
        write("mrbayes/$nm.mb.nex", mb)
end