from torch.utils.data import Subset
import sklearn.metrics as metrics
import os
import sys
import yaml
import torch
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import shutil
import textwrap

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.data.gtsrb_dataset import GTSRBDataset
from src.data.concepts_dataset import ConceptsDataset
from src.models.ENV2 import ENV2 as Stage_1
from src.models.LabelModel import LabelModel as Stage_2
import src.utils.plots as plots
import src.utils.reports as reports


class demonstrate:
    def __init__(self, pth_model_1=None, pth_model_2=None, last=False,
                 pth_data=None, model_variant="M", layers=1):
        with open("config/training.yml", "r") as f:
            self.tr_cfg = yaml.safe_load(f)
        with open("config/dataset.yml", "r") as f:
            self.ds_cfg = yaml.safe_load(f)
        with open("config/paths.yml", "r") as f:
            self.pth_cfg = yaml.safe_load(f)

        if last is False:
            mode = "best_model"
        else:
            mode = "last_model"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stage_1 = Stage_1(model_variant=model_variant)
        self.stage_2 = Stage_2(
            layers=self.tr_cfg["stage_2"]["layers"],
            hidden_dim=self.tr_cfg["stage_2"]["hidden_dim"],
            hidden_dim2=int(self.tr_cfg["stage_2"]["hidden_dim"] / 2)
        )
        self.instance_1 = self.stage_1.get_instance()
        self.instance_2 = self.stage_2.get_instance()
        self.ts = str(datetime.datetime.now())[:10] + "_" + time.strftime("%H:%M:%S").replace(":", "-")

        if pth_model_1 is not None:
            print("Fetching stage 1 from specified file path...")
            try:
                tmp = torch.load(pth_model_1, map_location=self.device)
                self.instance_1.load_state_dict(tmp[mode])
                self.instance_1.to(self.device)
                self.instance_1.eval()
            except Exception as e:
                raise ValueError(
                    "Could not find model path for stage 1. "
                    "Please make sure model path relative to project root is correct."
                ) from e
        else:
            print("Fetching stage 1 from standard file path...")
            tmp = torch.load(os.path.join(self.pth_cfg["registry"], "stage_1.pth"), map_location=self.device)
            self.instance_1.load_state_dict(tmp[mode])
            self.instance_1.to(self.device)
            self.instance_1.eval()

        if pth_model_2 is not None:
            print("Fetching stage 2 from specified file path...")
            try:
                tmp = torch.load(pth_model_2, map_location=self.device)
                self.instance_2.load_state_dict(tmp[mode])
                self.instance_2.to(self.device)
                self.instance_2.eval()
            except Exception as e:
                raise ValueError(
                    "Could not find model path for stage 2. "
                    "Please make sure model path relative to project root is correct."
                ) from e
        else:
            print("Fetching stage 2 from standard file path...")
            tmp = torch.load(os.path.join(self.pth_cfg["registry"], "stage_2.pth"), map_location=self.device)
            self.instance_2.load_state_dict(tmp[mode])
            self.instance_2.to(self.device)
            self.instance_2.eval()

        self.max_workers = 4 + 4 * (self.device.type != "cuda")
        self.bsize = self.tr_cfg["stage_1"]["bsize"]

    def _get_single_image_transform(self):
        ds_cfg = self.ds_cfg["dataset"]
        size = ds_cfg.get("image_size", 64) * ds_cfg["zoom_factor"]

        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform

    def _load_single_image(self, image_path):
        if not os.path.isfile(image_path):
            raise ValueError(f"Image file not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        transform = self._get_single_image_transform()
        x = transform(img).unsqueeze(0)  # [1, C, H, W]
        return x, img

    def _read_name_list(self, csv_path):
        df = pd.read_csv(csv_path)

        return df["name"].astype(str).tolist()
    def _save_demo_pdf(self, original_image, image_path, out_pdf_path, true_label_name, true_concepts,
                       predicted_label_name, detected_concepts):

        detected_concepts_text = "Keine Konzepte erkannt."
        if len(detected_concepts) > 0:
            detected_concepts_text = "\n".join([f"- {c}" for c in detected_concepts])
            
        true_concepts_text = "Keine Ground-Truth-Konzepte gefunden."
        if len(true_concepts) > 0:
            true_concepts_text = "\n".join([f"- {c}" for c in true_concepts])

        wrapped_image_path = "\n".join(textwrap.wrap(image_path, width=90))
        predicted_wrapped_label = "\n".join(textwrap.wrap(predicted_label_name, width=70))
        true_wrapped_label = "\n".join(textwrap.wrap(true_label_name, width=70))

        fig = plt.figure(figsize=(12, 16))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[7, 9])

        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(original_image)
        ax_img.set_title("Input Image", fontsize=16)
        ax_img.axis("off")

        ax_txt = fig.add_subplot(gs[1])
        ax_txt.axis("off")

        report_text = (
            f"Demo Report\n"
            f"{'=' * 80}\n\n"
            f"Timestamp:\n{self.ts}\n\n"
            f"Predicted label name:\n{predicted_wrapped_label}\n\n"
            f"True label name:\n{true_wrapped_label}\n\n"
            f"Detected concepts:\n{detected_concepts_text}\n\n"
            f"True concepts:\n{true_concepts_text}\n\n"
        )

        ax_txt.text(
            0.01, 0.99, report_text,
            va="top", ha="left",
            fontsize=11,
            family="monospace"
        )

        fig.tight_layout()
        fig.savefig(out_pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    def _get_class_id_from_path(self, image_path):
        class_folder = os.path.basename(os.path.dirname(image_path))

        class_id = int(class_folder)
        
        return class_id
    def _get_ground_truth_from_csv(self, image_path):
        class_id = self._get_class_id_from_path(image_path)
        
        concept_df = pd.read_csv(os.path.join(self.pth_cfg["data"]["root"], "concepts_per_class.csv"))
        row = concept_df[concept_df["class_id"] == class_id]
        if row.empty:
            raise ValueError(f"Keine Zeile mit class_id={class_id} in concepts_per_class.csv gefunden.")

        row = row.iloc[0]

        true_label = row["class_name"]

        concept_columns = [
            col for col in concept_df.columns
            if col not in ["class_id", "class_name"]
        ]

        true_concepts = [col for col in concept_columns if int(row[col]) == 1]

        return class_id, true_label, true_concepts
    def demonstrate_on(self, image_path):
        start = time.time()
        print("Running demo on single image...")

        xb, original_img = self._load_single_image(image_path)
        xb = xb.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                concept_logits = self.instance_1(xb)

                concept_pred = (concept_logits.detach().cpu() >= 0).numpy().astype(np.float32)[0]

                concept_pred_tensor = torch.tensor(
                    concept_pred,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)

                label_logits = self.instance_2(concept_pred_tensor)
                label_pred = int(label_logits.argmax(dim=1).detach().cpu().item())


        label_csv = os.path.join(self.pth_cfg["data"]["root"], "class_map.csv")
        concept_csv = os.path.join(self.pth_cfg["data"]["root"], "concept_map.csv")

        label_names = self._read_name_list(label_csv)
        concept_names = self._read_name_list(concept_csv)
        
        true_class_id, true_label, true_concepts = self._get_ground_truth_from_csv(image_path)

        predicted_label_name = (
            label_names[label_pred] if label_pred < len(label_names) else f"label_{label_pred}"
        )

        detected_concepts = []
        for i, val in enumerate(concept_pred):
            if int(val) == 1:
                cname = concept_names[i] if i < len(concept_names) else f"concept_{i}"
                detected_concepts.append(cname)


        out_dir = f"reports/Demo-{self.ts}"
        os.makedirs(out_dir, exist_ok=True)


        image_filename = os.path.basename(image_path)
        copied_image_path = os.path.join(out_dir, image_filename)
        shutil.copy2(image_path, copied_image_path)

        pdf_path = os.path.join(out_dir, "demo_report.pdf")
        self._save_demo_pdf(
            original_image=original_img,
            image_path=os.path.abspath(image_path),
            out_pdf_path=pdf_path,
            true_label_name=true_label,
            true_concepts=true_concepts,
            predicted_label_name=predicted_label_name,
            detected_concepts=detected_concepts
        )

        print("Predicted label:", predicted_label_name)
        print("Detected concepts:", detected_concepts)
        print("PDF report saved to:", pdf_path)

        end = time.time()
        print("Time spent:", end - start)


if __name__ == "__main__":
    with open("config/training.yml", "r") as f:
        tr_cfg = yaml.safe_load(f)
    with open("config/paths.yml", "r") as f:
        pth_cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mpth1", type=str, help="Path to Checkpoint of Stage1 Model.", default=None)
    parser.add_argument("--mpth2", type=str, help="Path to Checkpoint of Stage2 Model.", default=None)
    parser.add_argument("--img", type=str, help="Path to input image.", default="00000/00000_00000.ppm")
    parser.add_argument("--last", type=bool, help="Grab last model (True) or best model (False)", default=False)
    args = parser.parse_args()

    d = demonstrate(
        pth_model_1=args.mpth1,
        pth_model_2=args.mpth2,
        last=args.last,
        pth_data=None,
        model_variant=tr_cfg["stage_1"]["model_variant"],
        layers=tr_cfg["stage_2"]["layers"]
    )
    image_path = os.path.join(pth_cfg["data"]["training"], args.img)
    d.demonstrate_on(image_path=image_path)