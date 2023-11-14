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
def cm_to_phy(charMtx):
    characters = np.array(['0', '1', '-'])
    pad = max([len(x) for x in charMtx.index])+5
    phy = f'{charMtx.shape[0]} {charMtx.shape[1]}\n'
    for l in charMtx.index:
        phy += l.ljust(pad)
        phy += ''.join(characters[charMtx.loc[l]])+'\n'
    return phy
#%%

dbs = os.listdir("../data/correspondences")


# %%
phy_pth = "../data/correspondences_phylip"
isExist = os.path.exists(phy_pth)

if not isExist:
    os.mkdir(phy_pth)

# %%

for db in dbs:
    charMtx = get_char_mtx(db)
    phy = cm_to_phy(charMtx)
    fn = os.path.join(phy_pth, db.split('.')[0] + '.phy')
    with open(fn, 'w') as f:
        f.write(phy)
# %%
