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
def cm_to_nex(charMtx):
    characters = np.array(['0', '1', '-'])
    pad = max([len(x) for x in charMtx.index])+5
    nex = f'''#Nexus
BEGIN DATA;
DIMENSIONS ntax={charMtx.shape[0]} nchar = {charMtx.shape[1]};
FORMAT DATATYPE=Restriction GAP=? MISSING=- interleave=no;
MATRIX

'''
    for l in charMtx.index:
        nex += l.ljust(pad)
        nex += ''.join(characters[charMtx.loc[l]])+'\n'
    nex += ';\nEND\n'
    return nex

# %%
nex_pth = "../data/cognate_classes_nexus"
isExist = os.path.exists(nex_pth)

if not isExist:
    os.mkdir(nex_pth)

# %%

for db in dbs:
    charMtx = get_char_matrix(db)
    nex = cm_to_nex(charMtx)
    fn = os.path.join(nex_pth, db.split('.')[0] + '.nex')
    with open(fn, 'w') as f:
        f.write(nex)
# %%
