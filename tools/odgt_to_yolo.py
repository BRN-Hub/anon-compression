import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--odgt_paths", nargs="+", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--include_occluded", action="store_true")
parser.add_argument("--include_unsure", action="store_true")
args = parser.parse_args()

out_dir_fbox = os.path.join(args.out_dir, "fbox")
out_dir_vbox = os.path.join(args.out_dir, "vbox")
out_dir_hbox = os.path.join(args.out_dir, "hbox")
odgt_paths = args.odgt_paths

os.makedirs(out_dir_fbox, exist_ok=True)
os.makedirs(out_dir_vbox, exist_ok=True)
os.makedirs(out_dir_hbox, exist_ok=True)

include_occluded = args.include_occluded
include_unsure = args.include_unsure

for path in odgt_paths:
    with open(path, "r") as f:
        for line in f:
            ann = json.loads(line)
            filename = ann["ID"] + ".txt"
            with (
                open(os.path.join(out_dir_fbox, filename), "w") as fboxf, 
                open(os.path.join(out_dir_vbox, filename), "w") as vboxf, 
                open(os.path.join(out_dir_hbox, filename), "w") as hboxf
            ):
                for gtbox in ann["gtboxes"]:
                    if gtbox["tag"] != "person": continue
                    if (
                        "occ" in gtbox["extra"] and 
                        gtbox["extra"]["occ"] == "1" and 
                        not include_occluded
                    ): 
                        continue
                        
                    if (
                        "occ" in gtbox["head_attr"] and 
                        gtbox["head_attr"]["occ"] == "1" and 
                        not include_occluded 
                        or
                        "unsure" in gtbox["head_attr"] and 
                        gtbox["head_attr"]["unsure"] and 
                        not include_unsure
                    ): 
                        continue
                    
                    if "fbox" in gtbox and len(gtbox["fbox"]) == 4:
                        fboxf.write("0 "+" ".join([str(i) for i in gtbox["fbox"]]) + "\n")
                    if "vbox" in gtbox and len(gtbox["vbox"]) == 4:
                        vboxf.write("0 "+" ".join([str(i) for i in gtbox["vbox"]]) + "\n")
                    if "hbox" in gtbox and len(gtbox["hbox"]) == 4:
                        hboxf.write("0 "+" ".join([str(i) for i in gtbox["hbox"]]) + "\n")
