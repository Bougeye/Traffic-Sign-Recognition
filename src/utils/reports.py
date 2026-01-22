from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

from typing import Sequence
import torch
from PIL import Image as PILImage
import io

def tensor_to_png(t):
    t = t.detach().cpu()

    if t.ndim == 3 and t.shape[0] not in (1, 3):
        t = t.permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    if t.shape[0] == 3:
        t = t * std + mean

    t = t.clamp(0, 1)
    t = (t * 255).byte()

    if t.shape[0] == 1:
        img = PILImage.fromarray(t[0].numpy(), mode="L").convert("RGB")
    else:
        img = PILImage.fromarray(t.permute(1, 2, 0).numpy(), mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def misclassification_report(cases, out_path, title="Misclassification report"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
    )
    
    def fmt(xs) -> str:
        if isinstance(xs, torch.Tensor):
            xs = xs.detach().cpu().tolist()
        try:
            import numpy as np
            if isinstance(xs, np.ndarray):
                xs = xs.tolist()
        except Exception:
            pass

        return ", ".join(map(str, xs)) if xs else "—"

    contents = []
    contents.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    contents.append(Spacer(1, 0.4*cm))
    contents.append(Paragraph(f"Total incorrect cases: <b>{len(cases)}</b>", styles["Normal"]))
    contents.append(Spacer(1, 0.8*cm))

    page_w, page_h = A4
    img_max_w = page_w - doc.leftMargin - doc.rightMargin
    img_max_h = img_max_w

    for i,e in enumerate(cases):
        pred_label = e["label_pred"]
        true_label = e["label_target"]

        pred_concepts = e["concept_pred"]
        true_concepts = e["concept_target"]

        pred_set = set(pred_concepts)
        true_set = set(true_concepts)
        overlap = sorted(pred_set & true_set)
        false_pos = sorted(pred_set - true_set)
        false_neg = sorted(true_set - pred_set)

        contents.append(Paragraph(f"Predicted label: <b>{pred_label}</b>", styles["Normal"]))
        contents.append(Paragraph(f"Ground truth: <b>{true_label}</b>", styles["Normal"]))

        contents.append(Spacer(1, 0.3*cm))

        png_buf = tensor_to_png(e["input"])
        
        pil = PILImage.open(png_buf)
        img_w, img_h = pil.size
        png_buf.seek(0)
        contents.append(Image(png_buf, width=img_w, height=img_h))
        contents.append(Spacer(1, 0.4*cm))

        table = [
            ["Predicted Concepts", fmt(pred_concepts)],
            ["True Concepts", fmt(true_concepts)],
            ["Overlap", fmt(overlap)],
            ["False Positives", fmt(false_pos)],
            ["False Negatives", fmt(false_neg)],
        ]

        tbl = Table(table, colWidths=[5*cm, img_max_w-5*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        contents.append(tbl)
        contents.append(Spacer(1, 0.8*cm))
        if i != len(cases) -1 :
            contents.append(PageBreak())
    doc.build(contents)
    
