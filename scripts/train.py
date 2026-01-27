import os
import argparse
import yaml
import sys
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.models.CBMModel import CBMModel

class train:
    def __init__(self, tr_cfg, ds_cfg, pth_cfg, model_variant="S", lr_stage1=0.0001, lr_stage2=0.001):
        self.cbmmodel = CBMModel(tr_cfg, ds_cfg, pth_cfg, model_variant=model_variant, lr_stage1=lr_stage1, lr_stage2=lr_stage2)

    def train_model(self, random_seed=42, epochs_stage1=20, epochs_stage2=20, early_stopping=True):
        self.cbmmodel.train(random_seed=random_seed, epochs_stage1=epochs_stage1, epochs_stage2=epochs_stage2, early_stopping=early_stopping)

    def experiment(self, model_variant="S", random_seed=42, epochs_stage1=20, epochs_stage2=20, early_stopping=True):
        self.cbmmodel.experiment_training(model_variant=model_variant, random_seed=random_seed, epochs_stage1=epochs_stage1, epochs_stage2=epochs_stage2, early_stopping=early_stopping)
    
if __name__ == "__main__":
    
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--trcfg", type=str, default="config/training.yml")
    pre_parser.add_argument("--dscfg", type=str, default="config/dataset.yml")
    pre_parser.add_argument("--pthcfg", type=str, default="config/paths.yml")

    pre_args, remaining_args = pre_parser.parse_known_args()

    with open(pre_args.trcfg, "r") as f:
        tr_cfg = yaml.safe_load(f)
    with open(pre_args.dscfg, "r") as f:
        ds_cfg = yaml.safe_load(f)
    with open(pre_args.pthcfg, "r") as f:
        pth_cfg = yaml.safe_load(f)
        
    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--mv", type=str, help="EfficientNetV2 model variant", default=tr_cfg["stage_1"]["model_variant"])
    parser.add_argument("--lrs1", type=float, help="Learning rate for concept classification model", default=tr_cfg["stage_1"]["lr"])
    parser.add_argument("--lrs2", type=float, help="Learning rate for traffic sign recognition model", default=tr_cfg["stage_2"]["lr"])
    parser.add_argument("--rs", type=int, help="Seed for random number generators", default=tr_cfg["random_seed"])
    parser.add_argument("--eps1", type=int, help="Number of epochs for training of concept classification model", default=tr_cfg["stage_1"]["epochs"])
    parser.add_argument("--eps2", type=int, help="Number of epochs for training of traffic sign recognition model", default=tr_cfg["stage_2"]["epochs"])
    parser.add_argument("--es", type=bool, help="Set if early stopping should be used in training", default=bool(tr_cfg["early_stopping"]))
    args = parser.parse_args(remaining_args)
    x = train(tr_cfg, ds_cfg, pth_cfg, model_variant=args.mv, lr_stage1=args.lrs1, lr_stage2=args.lrs2)
    #x.train_model(random_seed=args.rs, epochs_stage1=args.eps1, epochs_stage2=args.eps2, early_stopping=args.es)
    x.experiment(model_variant=args.mv, random_seed=args.rs, epochs_stage1=args.eps1, epochs_stage2=args.eps2, early_stopping=args.es)
    #ds = x.forward_stage_1()
    #x.read_dataset(ds)
    os.makedirs("Results/configs/", exist_ok=True)
    for e in ["training.yml","dataset.yml","paths.yml"]:
        shutil.copy(os.path.join("config",e),os.path.join("Results",e))
    
