import pandas as pd
df = pd.read_csv("test_concept_report.csv")
print(df.index.name)
for i in range(len(df)):
    print(df.iloc[i].index.name)
