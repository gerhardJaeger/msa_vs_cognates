import pythia
import wrapper as raxmlng
import os
import pandas as pd
from scipy import stats
import math
from tabulate import tabulate

results_dir = "../../data/"
def prefix(experiment, ds_id, ling_type):
    return os.path.join(results_dir, experiment, ds_id, ling_type)

df = pd.read_csv("../../data/info.csv")
properties = df.columns[3:]
result_table = []
for ling_type in ["cognate_classes", "correspondences", "combined"]:
    alphas = []
    heterogenity = []
    for i,row in df.iterrows():
        alpha = raxmlng.alpha(prefix("raxmlng_gamma", row["ds_id"], ling_type))
        alphas.append(alpha)
        if alpha < 20:
            heterogenity.append(True)
        else:
            heterogenity.append(False)
    alpha_col = "alpha_" + ling_type
    df[alpha_col] = alphas
    h_col = "heterogenity_" + ling_type
    df[h_col] = heterogenity
    for prop in properties:
        pearson = stats.pearsonr(df[alpha_col], df[prop])
        result_table.append([alpha_col, prop, pearson[0], pearson[1]])
        #pearson_h = stats.pearsonr(df[h_col], df[prop])
print(tabulate(result_table, tablefmt="pipe", floatfmt=".2f", headers = ["alpha", "property", "pearson correlation", "p-value"]))

