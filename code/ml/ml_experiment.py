import os
import pandas as pd
from tabulate import tabulate
from Bio import AlignIO
from Bio.AlignIO.PhylipIO import RelaxedPhylipWriter
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.AlignIO.NexusIO import NexusIterator
from ete3 import Tree



ling_types = ["cognate_classes", "correspondences", "combined"]


def prefix(experiment, ds_id, ling_type):
    return os.path.join(results_dir, experiment, ds_id, ling_type)

def best_tree_path(prefix):
    return prefix + ".raxml.bestTree"

def ml_trees_path(prefix):
    return prefix + ".raxml.mlTrees"


def final_llh(prefix):
    with open(prefix + ".raxml.log", "r") as logfile:
        lines = logfile.readlines()
    for line in lines:
        if line.startswith("Final LogLikelihood:"):
            return float(line.split(" ")[-1])
    return float('nan')


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
            parts = line.split(" / ")
            scores = []
            for part in parts:
                scores.append(float(part.split(" ")[2]))
            return scores
    return [float('nan'), float('nan'), float('nan')]

def run_inference(msa_path, model, prefix, args = ""):
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


def average_distance(glottolog_tree_path, ml_trees_path,  metric):
    with open(ml_trees_path, "r") as trees_file:
        tree_lines = trees_file.readlines()
    distances = []
    for line in tree_lines:
        with open("temp.tree", "w+") as temp_tree_file:
            temp_tree_file.write(line)
        if metric == "gq":
            distances.append(gq_distance(glottolog_tree_path, "temp.tree"))
        if metric == "rf":
            distances.append(rf_distance(glottolog_tree_path, "temp.tree"))
        os.remove("temp.tree")
    return sum(distances) /len(distances)

def convert_nex_to_phy(nex_path, phy_path):
        end = "\n ;\nend;\n"
        with open(nex_path, "r") as nexus_file:
            content = nexus_file.read()
        content = "\n".join(content.split("\n")[:-2]) + end
        parts = content.split("MATRIX")
        line = parts[0].split("\n")[2].split(" ")
        n_char = line[4][:-1]
        n_taxa = line[1].split('=')[1]
        align = parts[1][2:]
        head = "#NEXUS\n\nBEGIN DATA;\n\tDIMENSIONS NTAX=" + str(n_taxa) + " NCHAR=" + str(n_char) + ";\n\tFORMAT DATATYPE=Standard SYMBOLS= \"01\" MISSING=? GAP= -;\nMATRIX\n"
        new_content = head + align
        with open("temp.nex", "w+") as temp_nexus_file:
            temp_nexus_file.write(new_content)
        align = AlignIO.read("temp.nex", "nexus")
        with open(phy_path,"w+") as f:
            writer = RelaxedPhylipWriter(f)
            writer.write_alignment(align)
        os.remove("temp.nex")

def create_combined_msas(ds_id):
    for ds_id in ds_ids:
        print(ds_id)
        combined_nexus_path = os.path.join(combined_nexus_dir, ds_id + ".nex")
        combined_msa_path = os.path.join(combined_dir, ds_id + ".phy")
        convert_nex_to_phy(combined_nexus_path, combined_msa_path)



def run_raxmlng(ds_ids):
    for ds_id in ds_ids:
        print(ds_id)
        correspondence_msa_path = os.path.join(correspondence_dir, ds_id + ".phy")
        cognate_msa_path = os.path.join(cognate_dir, ds_id + ".phy")
        combined_msa_path = os.path.join(combined_dir, ds_id + ".phy")

        run_inference(correspondence_msa_path, "BIN+G", prefix("raxmlng_gamma", ds_id, "correspondences"))
        run_inference(cognate_msa_path, "BIN+G", prefix("raxmlng_gamma", ds_id, "cognate_classes"))
        run_inference(correspondence_msa_path, "BIN", prefix("raxmlng_nogamma", ds_id, "correspondences"))
        run_inference(cognate_msa_path, "BIN", prefix("raxmlng_nogamma", ds_id, "cognate_classes"))
        if os.path.isfile(combined_msa_path):
            run_inference(combined_msa_path, "BIN+G", prefix("raxmlng_gamma", ds_id, "combined"))
            run_inference(combined_msa_path, "BIN", prefix("raxmlng_nogamma", ds_id, "combined"))


def gq_distance(tree_name1, tree_name2):
    if tree_name1 is None or tree_name2 is None:
        return float('nan')
    if tree_name1 != tree_name1 or tree_name2 != tree_name2:
        return float("nan")
    os.system("qdist " + tree_name1 + " " + tree_name2 + " >out.txt")
    lines = open("out.txt").readlines()
    if len(lines) < 2: #error occurred
        return float('nan')
    res_q = float(lines[1].split("\t")[-3])
    qdist = 1 - res_q
    os.remove("out.txt")
    return qdist


def rf_distance(tree_name1, tree_name2):
    if tree_name1 is None or tree_name2 is None:
        return float('nan')
    if tree_name1 != tree_name1 or tree_name2 != tree_name2:
            return float("nan")
    try:
        t1 = Tree(tree_name1)
        t2 = Tree(tree_name2)
    except:
        return float("nan")
    rf, max_rf, common_leaves, parts_t1, parts_t2,discard_t1, discart_t2 = t1.robinson_foulds(t2, unrooted_trees = True)
    if max_rf == 0:
        return float('nan')
    return rf/max_rf


def print_distances(ds_ids, experiment, metric):
    columns = [ "dataset"] + ling_types
    best_matrix = []
    ml_avg_matrix = []
    for ds_id in ds_ids:
        glottolog_tree_path = "../../data/glottolog_trees/" + ds_id + "_glottolog.tre"
        if metric == "gq":
            best_matrix.append([ds_id] + [gq_distance(glottolog_tree_path, best_tree_path(prefix(experiment, ds_id, ling_type))) for ling_type in ling_types])
        if metric == "rf":
            best_matrix.append([ds_id] + [rf_distance(glottolog_tree_path, best_tree_path(prefix(experiment, ds_id, ling_type))) for ling_type in ling_types])
        ml_avg_matrix.append([ds_id] + [average_distance(glottolog_tree_path, ml_trees_path(prefix(experiment, ds_id, ling_type)), metric) for ling_type in ling_types])

    best_df = pd.DataFrame(best_matrix, columns = columns)
    ml_avg_df = pd.DataFrame(ml_avg_matrix, columns = columns)

    best_matrix.append(["median"] + [best_df[ling_type].median() for ling_type in ling_types])
    ml_avg_matrix.append(["median"] + [ml_avg_df[ling_type].median() for ling_type in ling_types])
    print("Experiment:", experiment)
    print( metric, "distance of best tree to gold standard tree")
    print(tabulate(best_matrix, tablefmt="pipe", floatfmt=".3f", headers = columns))
    print("Average", metric, "distances of all 20 ML trees to gold standard tree")
    print(tabulate(ml_avg_matrix, tablefmt="pipe", floatfmt=".3f", headers = columns))


def print_alphas(ds_ids, experiment):
    table = [[ds_id] + [alpha(prefix(experiment, ds_id, ling_type)) for ling_type in ling_types] for ds_id in ds_ids]
    print("Alpha")
    print(tabulate(table, tablefmt="pipe", floatfmt=".3f", headers = ["ds_id"] + [ling_type for ling_type in ling_types]))



exe_path = "raxml-ng"
plots_dir = "plots/"
results_dir = "../../data/"
correspondence_dir = "../../data/correspondences_phylip/"
cognate_dir = "../../data/cognate_classes_phylip/"
combined_nexus_dir = "../../data/combined/"
combined_dir = "../../data/combined_phylip/"
if not os.path.isdir(combined_dir):
    os.makedirs(combined_dir)

ds_ids = ["constenlachibchan", "crossandean", "dravlex", "felekesemitic", "hattorijaponic", "houchinese", "leekoreanic", "robinsonap", "walworthpolynesian", "zhivlovobugrian"]
create_combined_msas(ds_ids)
run_raxmlng(ds_ids)
print_distances(ds_ids, "raxmlng_gamma", "gq")
print_distances(ds_ids, "raxmlng_nogamma", "gq")
print_alphas(ds_ids, "raxmlng_gamma")
