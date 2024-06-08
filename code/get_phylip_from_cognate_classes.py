#%%
import pandas as pd
import numpy as np
import os

#%%

dbs = os.listdir("../datasets")

#%%

def get_char_matrix(db):
    d = pd.read_table(os.path.join("../datasets", db))
    c_matrices = []
    concepts = d.CONCEPT.unique()
    c_matrices = []
    for c in concepts:
        d_c = d[d.CONCEPT == c]
        c_matrices.append(pd.crosstab(d_c.DOCULECT, d_c.COGID))
    charMtx = pd.concat(c_matrices, axis=1).fillna(-1).astype(int)
    charMtx[charMtx > 1] = 1
    return charMtx

#%%
def cm_to_phy(charMtx):
    characters = np.array(['0', '1', '-'])
    pad = max([len(x) for x in charMtx.index])+5
    phy = f'{charMtx.shape[0]} {charMtx.shape[1]}\n'
    for l in charMtx.index:
        phy += l.ljust(pad)
        phy += ''.join(characters[charMtx.loc[l]])+'\n'
    return phy

# %%
phy_pth = "../data/cognate_classes_phylip"
isExist = os.path.exists(phy_pth)

if not isExist:
    os.mkdir(phy_pth)

# %%

for db in dbs:
    charMtx = get_char_matrix(db)
    phy = cm_to_phy(charMtx)
    fn = os.path.join(phy_pth, db.split('.')[0] + '.phy')
    with open(fn, 'w') as f:
        f.write(phy)
# %%
