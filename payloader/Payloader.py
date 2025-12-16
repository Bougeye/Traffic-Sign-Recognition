import os
import pandas as pd
from fabric import Connection
import shutil
from datetime import datetime

def payloader(key_file="key", make_zip=True, remove_zip=True):
    server = input("Enter server: ")
    user = input("Enter username: ")
    time = datetime.now()
    ts = str(time)[:10]+"_"+time.strftime("%H:%M:%S").replace(":","-")
    out="Results.csv"
    shutil.copy("work.slurm","Payload/work.slurm")
    if make_zip:
        shutil.make_archive("Payload", "zip", "Payload")
    c = Connection(host=server,
                   user=user,
                   connect_kwargs={"key_filename":key_file})
    tpath = "~/Payload"
    c.run(f"mkdir {tpath}", in_stream=False)
    c.put("Payload.zip",remote=f"{tpath[2:]}/Payload.zip")
    print("--- Pushed Payload to remote site")
    with c.cd(tpath):
        c.run("unzip -o -q Payload.zip", in_stream=False)
        c.run("rm Payload.zip", in_stream=False)
        print("--- Executing Payload")
        c.run("sbatch -Q --parsable --wait work.slurm", in_stream=False)
        print("--- Payload executed successfully")
        c.run(f"mv Results {ts}", in_stream=False)
        c.run(f"zip -q -r {ts}.zip {ts}", in_stream=False)
    c.get(f"{tpath[2:]}/{ts}.zip", local=f"Results/{ts}.zip")
    print("--- Fetched results from remote site")
    c.run(f"rm -r {tpath}", in_stream=False)
    c.close()
    shutil.unpack_archive(f"Results/{ts}.zip", "Results", "zip")
    os.remove(f"Results/{ts}.zip")
    if remove_zip:
        os.remove("Payload.zip")
    print("### TASK FINISHED ###")


payloader(make_zip=True, remove_zip=False)
