import os

raxmlng_path = "bin/raxml-ng"
predictor_path = "predictors/latest.pckl"


def get_difficulty(prefix):
    if not os.path.isfile(prefix):
        return float("nan")
    with open(prefix, "r") as outfile:
        lines = outfile.readlines()
        if len(lines) == 0:
            return float("nan")
        return float(lines[0])

def run(msa_path, prefix):
    if os.path.isfile(prefix):
        print("Files with prefix " + prefix + " already exist")
        return
    d = "/".join(prefix.split("/")[:-1])
    if not os.path.isdir(d):
        os.makedirs(d)
    command = "pythia -m " + msa_path + " -o " + prefix + " -r " + raxmlng_path + " -p " + predictor_path + " --removeDuplicates -v"
    os.system(command)

