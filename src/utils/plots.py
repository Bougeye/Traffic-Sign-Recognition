import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def epoch_loss(df,out_folder):
    """
    Req.: A pandas dataframe is provided containing the following columns:
          #Epoch: The epoch number
          #Training Loss: The training loss of the respective epoch
          #Validation Loss: The validation loss of the respective epoch
          Additionally an output folder must be specified that the plot is written to.
    Eff.: Validation loss and training loss per epoch are plotted and saved to the output folder.
    Res.: -
    """
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
    """
    Req.: A pandas dataframe is provided containing the following columns:
              #Epoch: The epoch number
              #Batch: The amount of processed batches in the respective epoch
              #Training Loss: The training loss of the last -bpdc- 
          Additionally the following parameters must be specified:
          #bpdc: the amount of batches after which data collection is performed mid-epoch
          #total_samples: The total amount of samples in the underlying dataset
          #out_folder: The output folder the plot is saved to
    Eff.: Training loss per bpdc and epoch are plotted and saved to the output folder.
    Res.: -
    """
    df_b["Batch_total"] = df_b["Epoch"]*(total_samples//bpdc)+df_b["Batch"]
    g = sns.lineplot(x="Batch_total", y="Training Loss", data = df_b)
    os.makedirs(out_folder, exist_ok=True)
    fig = g.figure
    fig.savefig(os.path.join(out_folder,"batch_loss.png"))
    plt.close(g.figure)

def epoch_accuracy(df,out_folder):
    """
    Req.: A pandas dataframe is provided containing the following columns:
              #epoch: The epoch number
              #accuracy: The validation accuracy of the respective epoch
          Additionally an output folder must be specified that the plot is written to.
    Eff.: Accuracy per epoch is plotted and saved to the output folder.
    Res.: -
    """
    g = sns.lineplot(x="epoch",y="accuracy",data=df)
    os.makedirs(out_folder, exist_ok=True)
    fig = g.figure
    fig.savefig(os.path.join(out_folder,"epoch_accuracy.png"))
    plt.close(g.figure)

def report(df,out_folder):
    """
    Req.: A pandas dataframe by the scikit classification report format is provided.
          Additionally an output folder must be specified that the plot is written to.
    Eff.: Precision, recall and f1-score per class are plotted seperately and saved to the output folder
    Res.: -
    """
    df["class"] = df.groupby("epoch").cumcount()%43
    print(df)
    os.makedirs(out_folder, exist_ok=True)
    for e in ["precision","recall","f1-score"]:
        g = sns.lineplot(x="epoch",y=e,
                         hue="class",data=df)
        fig = g.figure
        fig.savefig(os.path.join(out_folder,f"epoch_{e}.png"))
        plt.close(g.figure)

def per_label_accuracy(df, out_folder):
    real_out = os.path.join(out_folder,"per_label_metrics")
    os.makedirs(real_out, exist_ok=True)
    for e in df:
        if e != "epoch":
            g = sns.lineplot(x="epoch",y=e,data=df)
            g.figure.savefig(os.path.join(real_out,f"label_{e}_accuracy.png"))
            plt.close(g.figure)

def class_distribution(preds,train_pth,out_folder):
    train = []
    for e in os.listdir(train_pth):
        fpath = os.path.join(train_pth,e)
        if os.path.isdir:
            train+=[int(e)]*len(os.listdir(fpath))
    df = pd.DataFrame({"class":train+list(preds),"group":["train"]*len(train)+["test"]*len(preds)})
    g = sns.catplot(x="class",kind="count",col="group",hue="group",col_wrap=1,height=3,aspect=4.0,sharey=False,data=df)
    os.makedirs(out_folder, exist_ok=True)
    fig = g.figure
    ###compute distance:
    train = df[df.group == "train"].groupby("class").size()
    test = df[df.group == "test"].groupby("class").size()
    s_dist = (train/train.sum())-(test/test.sum())
    avg_dist = s_dist.abs().mean()
    trtote_dist = (s_dist.abs()*(train/train.sum())).sum()
    tetotr_dist = (s_dist.abs()*(test/test.sum())).sum()
    print("avg_dist: ",avg_dist)
    print("trtote_dist: ",trtote_dist)
    print("tetotr_dist: ",tetotr_dist)
    #fig.text(0.05,0.0,f"Average error assuming stratified train/test split: {avg_dist:.8f}")
    #fig.text(0.05,-0.05,f"Total error assuming stratified train/test split: {total_dist:.8f}")
    fig.savefig(os.path.join(out_folder,"class_distribution.png"))
    plt.close(g.figure)
    
