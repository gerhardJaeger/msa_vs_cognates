import os

exe_path = "./bin/raxml-ng"


def best_tree_path(prefix):
    return prefix + ".raxml.bestTree"

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




