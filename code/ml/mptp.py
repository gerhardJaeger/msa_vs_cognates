import os
exe_path = "./bin/mptp"

def get_num_species(output_path):
    outfile_path = output_path + ".txt"
    if not os.path.isfile(outfile_path):
        print("Outfile does not exist!")
        return float("nan")
    with open(outfile_path, "r") as outfile:
        lines = outfile.readlines()
    for line in lines:
        if line.startswith("Number of delimited species:"):
            return int(line.split(" ")[-1])
    return float("nan")

def run(tree_path, output_path, args = ""):
    outdir = "/".join(output_path.split("/")[:-1])
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    command = exe_path
    command += " --ml --multi"
    command += " --tree_file " + tree_path
    command += " --output_file " + output_path
    command += " " + args
    os.system(command)
