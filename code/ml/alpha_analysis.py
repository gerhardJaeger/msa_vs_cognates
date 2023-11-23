import pythia
import wrapper as raxmlng
import os
import pandas as pd
from scipy import stats
import math

results_dir = "../../data/"
def prefix(experiment, ds_id, ling_type):
    return os.path.join(results_dir, experiment, ds_id, ling_type)

df = pd.read_csv("linguistic_properties.csv")
properties = df.columns[2:]
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
        #pearson_h = stats.pearsonr(df[h_col], df[prop])
        print(alpha_col + " " + prop + " " + str(round(pearson[0], 3)) + " " + str(round(pearson[1], 3)))


