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
    fig = g.figure
    fig.savefig(os.path.join(out_folder,"epoch_plot.png"))
    plt.close(g.figure)

def batch_loss(df_b,bpdc,total_samples,out_folder):
    df_b["Batch_total"] = df_b["Epoch"]*(total_samples//bpdc)+df_b["Batch"]
    g = sns.lineplot(x="Batch_total", y="Training Loss", data = df_b)
    fig = g.figure
    fig.savefig(os.path.join(out_folder,"batch_plot.png"))
    plt.close(g.figure)

def total_accuracy(preds,targets):
    total = len(preds)
    correct = 0
    for i in range(total):
        correct += preds[i] == targets[i]
        if preds[i] == targets[i]:
            correct+=1
    return round(correct/total, 4)

def working_accuracy_per_class(preds,targets):
    accs = {}
    total = len(preds)
    for i in range(total):
        if str(targets[i]) not in accs:
            accs[str(targets[i])] = preds[i] == targets[i]
        else:
            accs[str(targets[i])] += preds[i] == targets[i]
    for e in accs:
        accs[e] = round(accs[e]/total,4)
    return accs

def right_accuracy_per_class(preds,targets,labelmap):
    accs = {}
    total = len(preds)
    for i in range(total):
        label = labelmap[str(targets[i])]
        if label not in accs:
            accs[label] = preds[i] == targets[i]
        else:
            accs[label] += preds[i] == targets[i]
    for e in accs:
        accs[e] = round(accs[e]/total,4)
    return accs

epoch_data = pd.read_csv("Output_epochs.csv")
batch_data = pd.read_csv("Output_batches.csv")
epoch_loss(epoch_data,".")
batch_loss(batch_data,20,39209,".")
