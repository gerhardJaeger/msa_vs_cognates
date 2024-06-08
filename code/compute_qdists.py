#%%
import os
import pandas as pd
from subprocess import run


# %%

dbs = [fn.split(".")[0] for fn in os.listdir("../languages")]
# %%

d1 = dict()
for db in dbs:
    cmd = f'qdist ../data/correspondences_mltree/{db}_ml.tre ../data/glottolog_trees/{db}_glottolog.tre | tail -n 1'
    out = run(cmd, capture_output=True, shell=True).stdout
    try:
        qd = 1-float(out.decode('utf-8').split("\t")[5])
        d1[db] = qd
    except:
        pass

# %%

d2 = dict()
for db in dbs:
    cmd = f'qdist ../data/cognate_classes_mltree/{db}_ml.tre ../data/glottolog_trees/{db}_glottolog.tre | tail -n 1'
    out = run(cmd, capture_output=True, shell=True).stdout
    try:
        qd = 1-float(out.decode('utf-8').split("\t")[5])
        d2[db] = qd
    except:
        pass
# %%

results = pd.concat([pd.Series(d1), pd.Series(d2)], axis=1)
results.columns = ['correspondence', 'cognates']
# %%

results.round(3).to_csv("../data/qdist_comparison.tsv", sep="\t")
# %%
