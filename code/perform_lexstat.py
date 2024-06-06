#%%
import lingpy
import pandas as pd
import numpy as np
# %%

d = pd.read_csv('../data/lexibank_wordlist_pruned.csv', dtype=str).dropna(subset=["ASJP", "Cognateset_ID"])

d["forms"] = [str(x) for x in d.Segments]
d["id"] = range(len(d))

d = d[["id", "db", "Glottocode", "Concepticon_Gloss", "forms"]]
d = d.dropna()

d.insert(
    loc=4,
    column='tokens', 
    value=d['forms'].apply(lambda x: str(x).split()).to_list())

# %%

dbs = d.db.unique()


#%%

outputs = []

for db in dbs:
    df = d[d.db == db][["id", "Glottocode", "Concepticon_Gloss", "forms", "tokens"]]
    df.columns = ["id", "doculect", "concept", "forms", "tokens"]
    data_dict = {
        0: df.columns.to_list()
    }
    for i in range(len(df)):
        data_dict[i+1] = df.iloc[i].to_list() 
    wl = lingpy.Wordlist(data_dict)
    lex = lingpy.LexStat(wl)
    lex.get_scorer(runs=10000)
    cognate_detect = [
        ('lexstat', 'lexstatid', 0.55, 'infomap')
    ]
    for method, ref, threshold, clustering in cognate_detect:
        lex.cluster(method=method, threshold=threshold, ref=ref, cluster_method=clustering)
    output = pd.DataFrame(columns=lex.columns)
    for i in range(len(lex)):
        output.loc[i] = lex[i+1]
    output = output[["id", "doculect", "concept", "forms", "lexstatid"]]
    output["db"] = dbs[1]
    outputs.append(output)
# %%

final_output = pd.concat(outputs, ignore_index=True)

final_output.to_csv("../data/lexstat_output.csv", index=False)