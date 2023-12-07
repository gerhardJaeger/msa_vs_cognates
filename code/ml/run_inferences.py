import pythia
import wrapper as raxmlng
import mptp
import os
import pandas as pd
import math
from Bio import AlignIO
from Bio.AlignIO.PhylipIO import RelaxedPhylipWriter
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.AlignIO.NexusIO import NexusIterator
from tabulate import tabulate
from ete3 import Tree


import matplotlib.pyplot as plt

ling_types = ["cognate_classes", "correspondences", "combined"]


def prefix(experiment, ds_id, ling_type):
    return os.path.join(results_dir, experiment, ds_id, ling_type)


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

        raxmlng.run_inference(correspondence_msa_path, "BIN+G", prefix("raxmlng_gamma", ds_id, "correspondences"))
        raxmlng.run_inference(cognate_msa_path, "BIN+G", prefix("raxmlng_gamma", ds_id, "cognate_classes"))
        raxmlng.run_inference(correspondence_msa_path, "BIN", prefix("raxmlng_nogamma", ds_id, "correspondences"))
        raxmlng.run_inference(cognate_msa_path, "BIN", prefix("raxmlng_nogamma", ds_id, "cognate_classes"))
        if os.path.isfile(combined_msa_path):
            raxmlng.run_inference(combined_msa_path, "BIN+G", prefix("raxmlng_gamma", ds_id, "combined"))
            raxmlng.run_inference(combined_msa_path, "BIN", prefix("raxmlng_nogamma", ds_id, "combined"))

def run_raxmlng_with_constraint(ds_ids):
    for ds_id in ds_ids:
        print(ds_id)
        correspondence_msa_path = os.path.join(correspondence_dir, ds_id + ".phy")
        cognate_msa_path = os.path.join(cognate_dir, ds_id + ".phy")
        combined_msa_path = os.path.join(combined_dir, ds_id + ".phy")
        glottolog_tree_path = "../../data/glottolog_trees_modified/" + ds_id + "_glottolog.tre"
        args = " --tree-constraint " + glottolog_tree_path

        raxmlng.run_inference(correspondence_msa_path, "BIN+G", prefix("raxmlng_gamma_constraint", ds_id, "correspondences"), args)
        raxmlng.run_inference(cognate_msa_path, "BIN+G", prefix("raxmlng_gamma_constraint", ds_id, "cognate_classes"), args)
        raxmlng.run_inference(correspondence_msa_path, "BIN", prefix("raxmlng_nogamma_constraint", ds_id, "correspondences"), args)
        raxmlng.run_inference(cognate_msa_path, "BIN", prefix("raxmlng_nogamma_constraint", ds_id, "cognate_classes"), args)
        if os.path.isfile(combined_msa_path):
            raxmlng.run_inference(combined_msa_path, "BIN+G", prefix("raxmlng_gamma_constraint", ds_id, "combined"), args)
            raxmlng.run_inference(combined_msa_path, "BIN", prefix("raxmlng_nogamma_constraint", ds_id, "combined"), args)


def run_pythia(ds_ids):
    for ds_id in ds_ids:
        print(ds_id)
        correspondence_msa_path = os.path.join(correspondence_dir, ds_id + ".phy")
        cognate_msa_path = os.path.join(cognate_dir, ds_id + ".phy")
        combined_msa_path = os.path.join(combined_dir, ds_id + ".phy")

        pythia.run(correspondence_msa_path, prefix("pythia", ds_id, "correspondences"))
        pythia.run(cognate_msa_path, prefix("pythia", ds_id, "cognate_classes"))
        if os.path.isfile(combined_msa_path):
            pythia.run(combined_msa_path,  prefix("pythia", ds_id, "combined"))


def run_mptp(ds_ids):
    for ds_id in ds_ids:
        for ling_type in ling_types:
            best_tree_path = raxmlng.best_tree_path(prefix("raxmlng_gamma", ds_id, ling_type))
            mptp.run(best_tree_path, prefix("mptp_gamma", ds_id, ling_type))


def gq_distance(tree_name1, tree_name2):
    if tree_name1 is None or tree_name2 is None:
        return float('nan')
    if tree_name1 != tree_name1 or tree_name2 != tree_name2:
        return float("nan")
    os.system("./bin/qdist " + tree_name1 + " " + tree_name2 + " >out.txt")
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


def modeltesting(ds_ids):
    alphas = {}
    for ling_type in ling_types:
        print(ling_type)
        scores_gamma = []
        scores_nogamma = []
        alphas[ling_type] = []
        for ds_id in ds_ids:
            gamma_prefix = prefix("raxmlng_gamma", ds_id, ling_type)
            nogamma_prefix = prefix("raxmlng_nogamma", ds_id, ling_type)
            alphas[ling_type].append(raxmlng.alpha(gamma_prefix))
            scores_gamma.append(raxmlng.aic(gamma_prefix))
            scores_nogamma.append(raxmlng.aic(nogamma_prefix))
        gamma = 0
        no_gamma = 0
        for i, ds_id in enumerate(ds_ids):
            if scores_gamma[i][0] <= scores_nogamma[i][0]:
                rel = math.exp((scores_gamma[i][0] - scores_nogamma[i][0])/2)
                print("BIN+G")
                print(rel)
                gamma += 1
            else:
                rel = math.exp((scores_nogamma[i][0] - scores_gamma[i][0])/2)
                print("BIN")
                print(rel)
                no_gamma += 1
        print("BIN: " + str(no_gamma) + " BIN+G: " + str(gamma))
        relative_likelihoods = [math.exp((scores_gamma[i][0] - scores_nogamma[i][0])/2) for i in range(len(ds_ids))]
        fig, ax = plt.subplots()
        ax.scatter(alphas[ling_type], relative_likelihoods, s=2)
        ax.set_yscale('log')
        plt.xlabel('alpha')
        plt.ylabel('relative likelihood of BIN')
        plt.savefig(os.path.join(plots_dir, "relative_lhs_alpha_" + ling_type + '.png'))
        plt.clf()


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

def results(ds_ids, experiment, metric):
    columns = [ "dataset"] + ling_types
    best_matrix = []
    ml_avg_matrix = []
    for ds_id in ds_ids:
        glottolog_tree_path = "../../data/glottolog_trees/" + ds_id + "_glottolog.tre"
        if metric == "gq":
            best_matrix.append([ds_id] + [gq_distance(glottolog_tree_path, raxmlng.best_tree_path(prefix(experiment, ds_id, ling_type))) for ling_type in ling_types])
        if metric == "rf":
            best_matrix.append([ds_id] + [rf_distance(glottolog_tree_path, raxmlng.best_tree_path(prefix(experiment, ds_id, ling_type))) for ling_type in ling_types])
        ml_avg_matrix.append([ds_id] + [average_distance(glottolog_tree_path, raxmlng.ml_trees_path(prefix(experiment, ds_id, ling_type)), metric) for ling_type in ling_types])

    best_df = pd.DataFrame(best_matrix, columns = columns)
    ml_avg_df = pd.DataFrame(ml_avg_matrix, columns = columns)

    best_matrix.append(["median"] + [best_df[ling_type].median() for ling_type in ling_types])
    ml_avg_matrix.append(["median"] + [ml_avg_df[ling_type].median() for ling_type in ling_types])
    print(experiment)
    print(metric)
    print("best trees")
    print(tabulate(best_matrix, tablefmt="pipe", floatfmt=".3f", headers = columns))
    print("average for all 20 ml trees")
    print(tabulate(ml_avg_matrix, tablefmt="pipe", floatfmt=".3f", headers = columns))



def llh_comparison(ds_ids, experiment1, experiment2):
    matrix = []
    for ds_id in ds_ids:
        row = [ds_id]
        for ling_type in ling_types:
            row.append(raxmlng.final_llh(prefix(experiment1, ds_id, ling_type)))
            row.append(raxmlng.final_llh(prefix(experiment2, ds_id, ling_type)))
        matrix.append(row)
    headers = ["ds_id"]
    ex1_string = "_".join(experiment1.split("_")[1:])
    ex2_string = "_".join(experiment2.split("_")[1:])
    for ling_type in ling_types:
        headers.append(ling_type + " " + ex1_string)
        headers.append(ling_type + " " + ex2_string)
    print("Final llhs")
    print(tabulate(matrix, tablefmt="pipe", floatfmt=".3f", headers = headers))


def difficulties(ds_ids, experiment):
    matrix = []
    columns = [ "dataset",
               "cognate",
               "correspondences",
               "combined"
               ]
    for ds_id in ds_ids:
        row_new = []

        row_new.append(ds_id)
        row_new.append(pythia.get_difficulty(prefix(experiment, ds_id, "cognate_classes")))
        row_new.append(pythia.get_difficulty(prefix(experiment, ds_id, "correspondences")))
        row_new.append(pythia.get_difficulty(prefix(experiment, ds_id, "combined")))

        matrix.append(row_new)
    print(tabulate(matrix, tablefmt="pipe", floatfmt=".3f", headers = columns))
    df = pd.DataFrame(matrix, columns = columns)
    df = df.sort_values('dataset')
    print("method\t\tdifficulty(median)")
    print("correspondences\t" + str(df['correspondences'].median()))
    print("combined\t" + str(df['combined'].median()))
    print("cognate\t\t" + str(df['cognate'].median()))
    return df






plots_dir = "plots/"
results_dir = "../../data/"
correspondence_dir = "../../data/correspondences_phylip/"
cognate_dir = "../../data/cognate_classes_phylip/"
combined_nexus_dir = "../../data/combined/"
combined_dir = "../../data/combined_phylip/"
if not os.path.isdir(combined_dir):
    os.makedirs(combined_dir)

ds_ids = ["constenlachibchan", "crossandean", "dravlex", "felekesemitic", "hattorijaponic", "houchinese", "leekoreanic", "robinsonap", "walworthpolynesian", "zhivlovobugrian"]
#create_combined_msas(ds_ids)
#run_raxmlng(ds_ids)
#run_raxmlng_with_constraint(ds_ids)
#run_pythia(ds_ids)
#run_mptp(ds_ids)
results(ds_ids, "raxmlng_gamma_constraint", "gq")
#results(ds_ids, "raxmlng_nogamma_constraint", "gq")
#llh_comparison(ds_ids, "raxmlng_gamma", "raxmlng_gamma_constraint")
#llh_comparison(ds_ids, "raxmlng_nogamma", "raxmlng_nogamma_constraint")
