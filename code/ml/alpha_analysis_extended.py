import pythia
import wrapper as raxmlng
import mptp
import os
import pandas as pd
from tabulate import tabulate
from ete3 import Tree
from Bio import AlignIO

results_dir = "../../data/"
ling_types = ["cognate_classes", "correspondences", "combined"]
def prefix(experiment, ds_id, ling_type):
    return os.path.join(results_dir, experiment, ds_id, ling_type)

def get_num_taxa(msa_path):
    alignment = AlignIO.read(msa_path, "phylip-relaxed")
    return len(alignment)



def average_branch_length(tree_name):
    t  =  Tree(tree_name)
    brlens = []
    for node in t.traverse("postorder"):
        brlens.append(node.dist)
    return sum(brlens)/len(brlens)

def print_alphas(ds_ids):
    table = [[ds_id] + [raxmlng.alpha(prefix("raxmlng_gamma", ds_id, ling_type)) for ling_type in ling_types] for ds_id in ds_ids]
    print("Alpha")
    print(tabulate(table, tablefmt="latex_raw", floatfmt=".3f", headers = ["ds_id"] + [ling_type for ling_type in ling_types]))


def print_num_species(ds_ids):
    num_species = [[mptp.get_num_species(prefix("mptp_gamma", ds_id, ling_type)) for ling_type in ling_types] for ds_id in ds_ids]
    num_taxa = [get_num_taxa(os.path.join(cognate_dir, ds_id + ".phy")) for ds_id in ds_ids]
    table = [[ds_id] + num_species[i] + [num_taxa[i]]  for i,ds_id in enumerate(ds_ids)]
    print("Number of species determined with mptp in the  best tree inferred with BIN+G")
    print(tabulate(table, tablefmt="latex_raw", headers = ["ds_id"] + [ling_type for ling_type in ling_types] + ["number of taxa"]))

def print_avg_brlen(ds_ids):
    table = [[ds_id] + [average_branch_length(raxmlng.best_tree_path(prefix("raxmlng_gamma", ds_id, ling_type))) for ling_type in ling_types] for ds_id in ds_ids]
    print("Average branch length in the best tree inferred with BIN+G")
    print(tabulate(table, tablefmt="latex_raw", floatfmt=".3f", headers = ["ds_id"] + [ling_type for ling_type in ling_types]))


results_dir = "../../data/"
correspondence_dir = "../../data/correspondences_phylip/"
cognate_dir = "../../data/cognate_classes_phylip/"
combined_nexus_dir = "../../data/combined/"
combined_dir = "../../data/combined_phylip/"

ds_ids = ["constenlachibchan", "crossandean", "dravlex", "felekesemitic", "hattorijaponic", "houchinese", "leekoreanic", "robinsonap", "walworthpolynesian", "zhivlovobugrian"]

print_alphas(ds_ids)
print_num_species(ds_ids)
print_avg_brlen(ds_ids)
