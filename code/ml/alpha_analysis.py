import pythia
import wrapper as raxmlng
import os
import pandas as pd
import math

results_dir = "../../data/"
def prefix(experiment, ds_id, ling_type):
    return os.path.join(results_dir, experiment, ds_id, ling_type)

df = pd.read_csv("linguistic_properties.csv")
properties = df.columns[2:]
for ling_type in ["cognate_classes", "correspondences", "combined"]:
    alphas = []
    for i,row in df.iterrows():
        alphas.append(raxmlng.alpha(prefix("raxmlng_gamma", row["ds_id"], ling_type)))
    alpha_col = "alpha_" + ling_type
    df[alpha_col] = alphas
    for prop in properties:
        print(alpha_col + " " + prop + " " + str(df[alpha_col]. corr(df[prop])))


