import pythia
import raxmlng
import os
import pandas as pd
from Bio import AlignIO
from Bio.AlignIO.PhylipIO import RelaxedPhylipWriter
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.AlignIO.NexusIO import NexusIterator




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

        raxmlng.run_inference(correspondence_msa_path, "BIN+G", prefix("raxmlng", ds_id, "correspondences"))
        raxmlng.run_inference(cognate_msa_path, "BIN+G", prefix("raxmlng", ds_id, "cognate_classes"))
        if os.path.isfile(combined_msa_path):
            raxmlng.run_inference(combined_msa_path, "BIN+G", prefix("raxmlng", ds_id, "combined"))


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




def results(ds_ids):
    matrix = []
    columns = [ "dataset",
               "cognate",
               "correspondences",
               "combined"
               ]
    for ds_id in ds_ids:
        row_new = []
        glottolog_tree_path = "../../data/glottolog_trees/" + ds_id + "_glottolog.tre"
        tree_paths = []
        tree_paths.append(raxmlng.best_tree_path(prefix("raxmlng", ds_id, "cognate_classes")))
        tree_paths.append(raxmlng.best_tree_path(prefix("raxmlng", ds_id, "correspondences")))
        tree_paths.append(raxmlng.best_tree_path(prefix("raxmlng", ds_id, "combined")))

        row_new.append(ds_id)
        for tree_path in tree_paths:
            row_new.append(gq_distance(glottolog_tree_path, tree_path))

        matrix.append(row_new)
    df = pd.DataFrame(matrix, columns = columns)
    df = df.sort_values('dataset')
    print(df)
    return df






results_dir = "../../data/"
correspondence_dir = "../../data/correspondences_phylip/"
cognate_dir = "../../data/cognate_classes_phylip/"
combined_nexus_dir = "../../data/combined/"
combined_dir = "../../data/combined_phylip/"
if not os.path.isdir(combined_dir):
    os.makedirs(combined_dir)

ds_ids = ["walworthpolynesian", "constenlachibchan", "crossandean", "robinsonap", "zhivlovobugrian", "hattorijaponic", "felekesemitic", "houchinese", "dravlex", "leekoreanic"]
#create_combined_msas(ds_ids)
#run_raxmlng(ds_ids)
#run_pythia(ds_ids)
results(ds_ids)







