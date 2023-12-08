import os
import wrapper as raxmlng

import iqtree_statstest_parser
iq_tree_path = "./bin/iqtree2"

def sigtests(msa_path, model, unconstrained_tree_path, constrained_tree_path, prefix):
    prefix_dir = "/".join(prefix.split("/")[:-1])
    if not os.path.isdir(prefix_dir):
        os.makedirs(prefix_dir)
    with open(unconstrained_tree_path, "r") as u_tree_file:
        u_tree_string = u_tree_file.read()
    with open(constrained_tree_path, "r") as c_tree_file:
        c_tree_string = c_tree_file.read()
    z_tree_path = prefix + "_z.trees"
    with open(z_tree_path, "w+") as z_tree_file:
        z_tree_file.write(u_tree_string)
        z_tree_file.write(c_tree_string)
    command = iq_tree_path
    command += " -s " + msa_path
    command += " -pre " + prefix
    command += " -m " + model
    command += " -z " + z_tree_path
    command += " -te " + unconstrained_tree_path
    command += "  -n 0 -zb 10000 -zw -au -seed 0"
    os.system(command)


def get_results(prefix):
    res = iqtree_statstest_parser.get_iqtree_results(prefix + ".iqtree")
    return res
