#%%
import lingpy
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

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
def process_db(db):
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

    return output
# %%


outputs = []

# Run the processing in parallel
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_db, db): db for db in dbs}
    for future in as_completed(futures):
        try:
            result = future.result()
            outputs.append(result)
        except Exception as exc:
            print(f'Generated an exception: {exc}')

# Combine all outputs into a single DataFrame
final_output = pd.concat(outputs, ignore_index=True)

#%%

final_output.to_csv("../data/lexstat_output.csv", index=False)