#%%
import pandas as pd
import numpy as np
import os

#%%

def get_char_mtx(db):
    d = pd.read_table("../data/correspondences/"+db)
    languages = d.columns[3:-2]

    matrices = []

    for c in d.index:
        char = d.iloc[c,:]
        if char.FREQUENCY > 2:
            valueTokens = char.values[3:-2].astype('unicode')
            valueTypes = pd.unique(valueTokens)
            valueTypes = pd.Series([s for s in valueTypes if s != 'Ø'])
            cMtx = np.transpose(
                np.vstack([valueTokens == s for s in valueTypes], dtype=int)
                )
            cMtx[valueTokens == 'Ø',:] = -1
            matrices.append(cMtx)
    return pd.DataFrame(np.hstack(matrices), index=languages)
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
#%%

dbs = os.listdir("../data/correspondences")


# %%
nexus_pth = "../data/correspondences_nexus"
isExist = os.path.exists(nexus_pth)

if not isExist:
    os.mkdir(nexus_pth)

# %%

for db in dbs:
    charMtx = get_char_mtx(db)
    nex = cm_to_nex(charMtx)
    fn = os.path.join(nexus_pth, db.split('.')[0] + '.nex')
    with open(fn, 'w') as f:
        f.write(nex)
# %%
