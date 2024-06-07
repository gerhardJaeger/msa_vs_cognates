#%%
import lingpy
import pandas as pd
import numpy as np
import os
import sys

#%%
directory_path = "../data/lexstat"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)


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
db = dbs[int(sys.argv[1])]

#%%
def process_db(db, d=d):
    df = d[d["db"] == db][["id", "Glottocode", "Concepticon_Gloss", "forms", "tokens"]]
    df.columns = ["id", "doculect", "concept", "forms", "tokens"]

    # Create the data dictionary for lingpy.Wordlist
    data_dict = {0: df.columns.to_list()}
    for i in range(len(df)):
        data_dict[i + 1] = df.iloc[i].to_list()

    # Initialize the Wordlist and LexStat objects
    wl = lingpy.Wordlist(data_dict)
    lex = lingpy.LexStat(wl)

    # Generate the scorer
    lex.get_scorer(runs=10000)

    # Define the cognate detection parameters
    cognate_detect = [
        ('lexstat', 'lexstatid', 0.55, 'infomap')
    ]

    # Perform clustering
    for method, ref, threshold, clustering in cognate_detect:
        lex.cluster(method=method, threshold=threshold, ref=ref, cluster_method=clustering)

    # Construct the output DataFrame
    output = pd.DataFrame(columns=lex.columns)
    for i in range(len(lex)):
        output.loc[i] = lex[i + 1]
    output = output[["id", "doculect", "concept", "forms", "lexstatid"]]
    output["db"] = db
    output.to_csv(f"../data/lexstat/lexstat_{db}.csv", index=False)

# %%

process_db(db)