import yaml
import pandas as pd
import os

with open("config/paths.yml", "r") as f:
    pths = yaml.safe_load(f)

df = pd.read_csv(pths["data"]["concepts"])
index = 0
idx = []
concepts = []
for e in df:
    if index >= 2:
        idx.append(index-2)
        concepts.append(e)
    index+=1

out = pd.DataFrame({"name":concepts})
out.to_csv(os.path.join(pths["data"]["root"],"concept_map.csv"),index=False)
out = pd.DataFrame({"name":list(df["class_name"])})
out.to_csv(os.path.join(pths["data"]["root"],"class_map.csv"), index=False)
