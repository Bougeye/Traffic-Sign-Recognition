import argparse
print("Test")
parser = argparse.ArgumentParser(description="Test.")
parser.add_argument("--ds_cfg", type=str, help="dataset config")
parser.add_argument("--lrs1", type=int, help="Learning Rate stage 1")
args = parser.parse_args()
print(args.ds_cfg)
