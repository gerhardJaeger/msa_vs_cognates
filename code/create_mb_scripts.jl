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
    nm = split(split(nexus_file, "/")[end], ".")[1]
    mb = """
    #Nexus
    Begin MrBayes;
        execute ../$nexus_file;
        prset brlenspr = clock:uniform;
        lset rates=gamma;
        prset brlenspr = clock:uniform;
        prset clockvarpr=igr;
        lset coding=all;
        mcmcp stoprule=no burninfrac=0.25 stopval=0.01 filename=output/$(nm) samplefreq=1000 printfreq=1000;
        mcmc ngen=20000000 nchains=2 nruns=2;
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

mkpath("mrbayes_cc/output")

for nexus_file in datasets
    nm = split(split(nexus_file, "/")[end], ".")[1]
    mb = """
    #Nexus
    Begin MrBayes;
        execute ../$nexus_file;
        prset brlenspr = clock:uniform;
        lset rates=gamma;
        prset brlenspr = clock:uniform;
        prset clockvarpr=igr;
        lset coding=all;
        mcmcp stoprule=no burninfrac=0.25 stopval=0.01 filename=output/$(nm)_cc samplefreq=1000 printfreq=1000;
        mcmc ngen=20000000 nchains=2 nruns=2;
        sumt;
        sump;
        q;
    end;
    """
    write("mrbayes_cc/$nm.mb.nex", mb)
end
