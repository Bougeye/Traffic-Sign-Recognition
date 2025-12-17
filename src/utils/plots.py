import seaborn as sns
import matplotlib.pyplot as plt
import pandas

def plot_loss(df,out_folder):
    df_train = df.copy()
    df_val = df.copy()
    df_train.loc[:,"mode"] = "Training"
    df_val.loc[:,"mode"] = "Validation"
    df_train.rename(columns={"Training Loss":"Loss"})
    df_val.rename(columns={"Validation Loss":"Loss"})
    df_val.drop(columns=["Training Loss"])
    df_fin = pd.concat([df_train, df_val]
    g = sns.lineplot(x="epoch",y="loss",
                     hue=
