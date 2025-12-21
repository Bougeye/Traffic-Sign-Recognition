import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def epoch_loss(df,out_folder):
    df_train = df.copy()
    df_val = df.copy()
    df_train.loc[:,"Mode"] = "Training"
    df_val.loc[:,"Mode"] = "Validation"
    df_train = df_train.rename(columns={"Training Loss":"Loss"}).drop(columns=["Validation Loss"])
    df_val = df_val.rename(columns={"Validation Loss":"Loss"}).drop(columns=["Training Loss"])
    df_fin = pd.concat([df_train, df_val])
    g = sns.lineplot(x="Epoch",y="Loss",
                     hue="Mode", data=df_fin)
    os.makedirs(out_folder, exist_ok=True)
    fig = g.figure
    fig.savefig(os.path.join(out_folder,"epoch_loss.png"))
    plt.close(g.figure)

def batch_loss(df_b,bpdc,total_samples,out_folder):
    df_b["Batch_total"] = df_b["Epoch"]*(total_samples//bpdc)+df_b["Batch"]
    g = sns.lineplot(x="Batch_total", y="Training Loss", data = df_b)
    os.makedirs(out_folder, exist_ok=True)
    fig = g.figure
    fig.savefig(os.path.join(out_folder,"batch_loss.png"))
    plt.close(g.figure)

def epoch_accuracy(df,out_folder):
    g = sns.lineplot(x="epoch",y="accuracy",data=df)
    os.makedirs(out_folder, exist_ok=True)
    fig = g.figure
    fig.savefig(os.path.join(out_folder,"epoch_accuracy.png"))
    plt.close(g.figure)

def report(df,out_folder):
    df["class"] = df.groupby("epoch").cumcount()%43
    print(df)
    os.makedirs(out_folder, exist_ok=True)
    for e in ["precision","recall","f1-score"]:
        g = sns.lineplot(x="epoch",y=e,
                         hue="class",data=df)
        fig = g.figure
        fig.savefig(os.path.join(out_folder,f"epoch_{e}.png"))
        plt.close(g.figure)

