import os

exe_path = "./bin/raxml-ng"


def best_tree_path(prefix):
    return prefix + ".raxml.bestTree"


def consensus_tree_path(prefix):
    return prefix + ".raxml.consensusTreeMR"

def alpha(prefix):
    with open(prefix + ".raxml.log", "r") as logfile:
        lines = logfile.readlines()
    for line in lines:
        if line.startswith("   Rate heterogeneity:"):
            return float(line.split(",  ")[1].split(" ")[1])
    return float('nan')

def aic(prefix):
    with open(prefix + ".raxml.log", "r") as logfile:
        lines = logfile.readlines()
    for line in lines:
        if line.startswith("AIC"):
            return float(line.split(" / ")[0].split(" ")[2])
    return float('nan')

def consense_tree(prefixes, prefix):
    with open("trees.nw", "w+") as outfile:
        for p in prefixes:
            tree_path = best_tree_path(p)
            if os.path.isfile(tree_path):
                with open(tree_path, "r") as infile:
                    outfile.write(infile.read())
    command = exe_path
    command += " --consense"
    command += " --tree trees.nw "
    command += " --prefix " + prefix
    os.system(command)
    os.remove("trees.nw")


def run_inference(msa_path, model, prefix, args = ""):
    if exe_path == "":
        print("Please specify raxmlng.exe_path")
        return
    if msa_path is None:
        print("MSA " + msa_path + " does not exist")
        return
    if os.path.isfile(prefix + ".log"):
        print("Run files for " + prefix + " present")
        return
    prefix_dir = "/".join(prefix.split("/")[:-1])
    if not os.path.isdir(prefix_dir):
        os.makedirs(prefix_dir)
    command = exe_path
    command += " --msa " + msa_path
    command += " --model " + model
    command += " --prefix " + prefix
    command += " --threads 2 --seed 2"
    command += " " + args
    os.system(command)




