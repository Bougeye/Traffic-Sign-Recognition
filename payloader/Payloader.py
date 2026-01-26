import os
import sys
import pandas as pd
from fabric import Connection
import shutil
from datetime import datetime
import yaml
import shutil
import argparse

def payloader(key_file="key", make_zip=True, remove_zip=True):
    print("### STARTING TASK - PAYLOADER ###")    
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.append(ROOT)

    time = datetime.now()
    ts = str(time)[:10]+"_"+time.strftime("%H:%M:%S").replace(":","-")
    
    with open("config/paths.yml","r") as f:
        pths = yaml.safe_load(f)

    loader_path = os.path.join(ROOT,pths["payloader"])
    load_path = os.path.join(loader_path,"Payload")

    server = input("Enter server: ")
    user = input("Enter username: ")
    experiment = input("Enter experiment (optional): ")
    if experiment == "":
        experiment = "Unlabeled"
    out="Results.csv"
    shutil.copy(os.path.join(loader_path,"work.slurm"),os.path.join(load_path,"work.slurm"))
    if make_zip:
        try:
            os.remove(os.path.join(loader_path,"Payload.zip"))
        except:
            pass
        shutil.make_archive(load_path, "zip", load_path)
        print("--- Built Payload zip-file")
    print("--- Connecting to remote site")
    c = Connection(host=server,
                   user=user,
                   connect_kwargs={"key_filename":os.path.join(loader_path,key_file)})
    tpath = "~/Payload"
    c.run(f"mkdir {tpath}", in_stream=False)
    c.put(os.path.join(loader_path,"Payload.zip"),remote=f"{tpath[2:]}/Payload.zip")
    print("--- Pushed Payload to remote site")
    with c.cd(tpath):
        c.run("unzip -o -q Payload.zip", in_stream=False)
        c.run("rm Payload.zip", in_stream=False)
        print("--- Executing Payload")
        c.run("sbatch -Q --parsable --wait work.slurm", in_stream=False)
        print("--- Payload executed successfully")
        c.run(f"mv Results {ts}", in_stream=False)
        c.run(f"zip -q -r {ts}.zip {ts}", in_stream=False)
    c.get(f"{tpath[2:]}/{ts}.zip", local=os.path.join(loader_path,f"Results/{ts}.zip"))
    print("--- Fetched results from remote site")
    c.run(f"rm -r {tpath}", in_stream=False)
    c.close()
    shutil.unpack_archive(os.path.join(loader_path,f"Results/{ts}.zip"), os.path.join(loader_path,"Results"), "zip")
    os.remove(os.path.join(loader_path,f"Results/{ts}.zip"))
    if remove_zip:
        os.remove(os.path.join(loader_path,"Payload.zip"))
    exp_path = os.path.join(ROOT,pths["experiments"],experiment)
    os.makedirs(exp_path, exist_ok=True)
    shutil.copytree(os.path.join(loader_path,f"Results/{ts}"),os.path.join(exp_path,ts))
    print("--- Copied results to experiments")
    print("### TASK FINISHED ###")

def get_results(key_file="key", make_zip=True, remove_zip=True):
    print("### STARTING TASK - FETCHING RESULTS ###")
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    
    with open("config/paths.yml","r") as f:
        pths = yaml.safe_load(f)
    
    loader_path = os.path.join(ROOT,pths["payloader"])
    load_path = os.path.join(loader_path,"Payload")

    server = input("Enter server: ")
    user = input("Enter username: ")
    experiment = input("Enter experiment (optional): ")
    if experiment == "":
        experiment = "Unlabeled"
    
    time = datetime.now()
    ts = str(time)[:10]+"_"+time.strftime("%H:%M:%S").replace(":","-")
    print("--- Connecting to remote site")
    c = Connection(host=server,
                   user=user,
                   connect_kwargs={"key_filename":os.path.join(loader_path,key_file)})
    tpath = "~/Payload"
    print("--- Fetching results from remote site")
    with c.cd(tpath):
        c.run(f"mv Results {ts}", in_stream=False)
        c.run(f"zip -q -r {ts}.zip {ts}", in_stream=False)
        c.get(f"{tpath[2:]}/{ts}.zip", local=os.path.join(loader_path,f"Results/{ts}.zip"))
        print("--- Fetched results from remote site")
    c.close()
    shutil.unpack_archive(os.path.join(loader_path,f"Results/{ts}.zip"), os.path.join(loader_path,"Results"), "zip")
    os.remove(os.path.join(loader_path,f"Results/{ts}.zip"))
    exp_path = os.path.join(ROOT,pths["experiments"],experiment)
    os.makedirs(exp_path, exist_ok=True)
    shutil.copytree(os.path.join(loader_path,f"Results/{ts}"),os.path.join(exp_path,ts))
    print("--- Copied results to experiments")
    print("### TASK FINISHED ###")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=str, help="Must be 'results' to fetch results or 'payloader' for payloader", default="payloader")
    parser.add_argument("--mz", type=bool, help="(Re-)build zip file", default=False)
    parser.add_argument("--rz", type=bool, help="Remove zip after completion", default=False)
    
    args = parser.parse_args()
    if args.t == "payloader":
        payloader(make_zip=args.mz, remove_zip=args.rz)
    elif args.t == "results":
        get_results()
    else:
        print("Value error on task")
