from lingpy import *
from glob import glob
from tabulate import tabulate
from collections import defaultdict
import pandas as pd


table = []
for ds in glob("trimmed/*.tsv"):
    print(ds)
    sounds = defaultdict(lambda: {"sounds": [], "words": [], "wpl": 0})
    wl = Wordlist(ds)
    try:
        wl.calculate("diversity")
    except:
        wl.diversity = 0
        print('problem with diversity in {0}'.format(ds))
    row = [ds.split("/")[1][:-4], len(wl), wl.height, wl.width, wl.diversity]
    try:
        wl.calculate('distances', ref="cogid")
        dists = []
        for line in wl.distances:
            dists += [sum(line) / len(line)]
        row.append(sum(dists)/len(dists))
    except:
        row.append(0.0)
    
    for idx, language, tokens in wl.iter_rows('doculect', 'tokens'):
        sounds[language]["wpl"] += 1
        for token in tokens:
            sounds[language]["sounds"].append(token.split("/")[1] if "/" in token else
                                 token)
            sounds["all"]["sounds"].append(
                    token.split("/")[1] if "/" in token else token)
        sounds[language]["words"] += [len(tokens)]
        sounds["all"]["words"] += [len(tokens)]

    row.extend([
            len(set(sounds["all"]["sounds"])),
            sum([len(set(sounds[x]["sounds"])) for x in wl.cols]) / wl.width,
            sum(sounds["all"]["words"]) / len(wl),
            sum([sum(sounds[x]["words"]) / len(sounds[x]["words"]) for x in
                 wl.cols]) / wl.width,
            sum([sounds[x]["wpl"] for x in wl.cols]) / wl.width
            ])
    table += [row]
headers=[ "Dataset", "Concepts", "Languages", "Diversity", "Distances", "SoundsTotal",
    "SoundsAverage", "WordLength", "WordLengthAverage", "WordsPerLanguage"]
print(tabulate(table, tablefmt="pipe", floatfmt=".2f", headers = headers))

df = pd.DataFrame(table, columns = ["ds_id"] + headers)
df.sort_values("ds_id")
df.to_csv("data/info.csv")
