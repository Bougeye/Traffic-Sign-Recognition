from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

from typing import Sequence
import torch
from PIL import Image as PILImage
import io
import os
import pandas as pd

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

def to_binary(x, n):
    out = None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().flatten()
        if x.numel() == n:
            if x.dtype.is_floating_point:
                return (x >= 0.5).to(torch.int).tolist()
            return x.to(torch.int).tolist()
        idx = x.to(torch.int).tolist()
        out = [0] * n
        for i in idx:
            if 0 <= int(i) < n:
                out[int(i)] = 1
    return out

def concept_card(gt_label, pred_label, concept_names, gt_vec, pred_vec, card_width):
    
    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        "CardHeader",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        spaceAfter=6,
    )

    hd = sum(int(a != b) for a,b in zip(gt_vec, pred_vec))
    lines = [
        f"<b>Ground Truth (GT): </b> {gt_label}",
        f"<b>Prediction (P):</b> {pred_label}",
    ]
    lines.append(f"<b>Hamming  Dist. (HD):</b> {hd}")

    header_par = Paragraph("<br/>".join(lines), header_style)

    active = [i for i,(g,p) in enumerate(zip(gt_vec, pred_vec)) if (g == 1 or p == 1)]
    if not active:
        active = []
    data = [["", "GT", "P"]]
    for i in active:
        data.append([concept_names.iloc[i]["name"], str(gt_vec[i]), str(pred_vec[i])])

    name_col = card_width * 0.5
    gt_col = card_width * 0.15
    p_col = card_width * 0.15
    col_widths = [name_col, gt_col, p_col]
    tbl = Table(data, colWidths=col_widths)

    blue = colors.HexColor("#1f4e79")
    light_bg = colors.HexColor("#f2f2f2")

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), light_bg),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (1, 0), (-1, -1), 0.25, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (0, 1), (1, 0), "LEFT"),
    ]

    for r in range(1, len(data)):
        for i in [1,2]:
            if data[r][i] == "1":
                style_cmds += [
                    ("BACKGROUND", (i,r), (i,r), blue),
                    ("TEXTCOLOR", (i,r), (i,r), colors.white),
                    ("FONTNAME", (i,r), (i,r), "Helvetica-Bold"),
                ]
    tbl.setStyle(TableStyle(style_cmds))
    card = Table([[header_par], [tbl]], colWidths=[card_width])
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eeeeee")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 1), (1, 0), "LEFT"),
    ]))
    return card

def fit_dims(img_w_px: int, img_h_px: int, max_w_pt: float, max_h_pt: float):
    scale = min(max_w_pt / img_w_px, max_h_pt / img_h_px)
    return img_w_px * scale, img_h_px * scale



def misclassification_report(cases, concept_names, label_names, out_path, title="Misclassification Report"):
        
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
    contents.append(Spacer(1, 0.2*cm))
    contents.append(Paragraph(f"Total incorrect cases: <b>{len(cases)}</b>", styles["Normal"]))
    contents.append(Spacer(1, 0.3*cm))

    page_w, page_h = A4
    usable_w = page_w - doc.leftMargin - doc.rightMargin
    img_col_w = 7.5 * cm
    card_col_w = usable_w - img_col_w

    img_max_w = img_col_w
    img_max_h = img_max_w

    n_concepts = len(concept_names)

    for i,c in enumerate(cases):
        png_buf = tensor_to_png(c["input"])
        
        pil = PILImage.open(png_buf)
        iw, ih = pil.size
        png_buf.seek(0)
        w_pt, h_pt = fit_dims(iw, ih, img_max_w, img_max_h)
        img = Image(png_buf, width=w_pt, height=h_pt)
        gt_vec = to_binary(c["concept_target"], 43)
        pred_vec = to_binary(c["concept_pred"], 43)

        card = concept_card(
            gt_label=str(label_names.iloc[int(c["label_target"])]["name"]),
            pred_label=str(label_names.iloc[int(c["label_pred"])]["name"]),
            concept_names=concept_names,
            gt_vec=gt_vec,
            pred_vec=pred_vec,
            card_width=card_col_w
        )
        row = Table([[img, card]], colWidths=[img_col_w, card_col_w])
        row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))

        contents.append(row)
        contents.append(Spacer(1, 0.6 * cm))
        if i%3 == 2 and i != 2:
            contents.append(PageBreak())

    doc.build(contents)

def metrics_report(report_dict, names_map, out_path, title="Concept Report"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
    )
    contents = []
    contents.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    contents.append(Spacer(1, 0.2*cm))
    data = [["Name","Precision","Recall","f1-Score","Support"]]
    
    df = pd.DataFrame({})
    df["Name"] = names_map["name"]
    df["Precision"] = list(report_dict.iloc[:43]["precision"])
    df["Recall"] = list(report_dict.iloc[:43]["recall"])
    df["f1-Score"] = list(report_dict.iloc[:43]["f1-score"])
    for i in range(43):
        row = report_dict.iloc[i]
        data.append([names_map.iloc[i]["name"],round(row["precision"],6),round(row["recall"],6),round(row["f1-score"],6),int(row["support"])])
    if len(report_dict) == 46:
        names = ["accuracy","macro avg","weightes avg"]
    else:
        names = ["micro avg","macro avg","weighted avg","samples avg"]
    for i in range(43,43+len(names)):
        row = report_dict.iloc[i]
        data.append([names[i-43],round(row["precision"],6),round(row["recall"],6),round(row["f1-score"],6),int(row["support"])])
    page_w, page_h = A4
    usable_w = page_w - doc.leftMargin - doc.rightMargin
    name_col = usable_w * 0.48
    pr_col = usable_w * 0.13
    re_col = usable_w * 0.13
    f1_col = usable_w * 0.13
    sup_col = usable_w * 0.13
    col_widths = [name_col, pr_col, re_col, f1_col, sup_col]
    tbl = Table(data, colWidths=col_widths)
    
    light_bg = colors.HexColor("#f2f2f2")
    blue1 = colors.HexColor("#97a6c4")
    blue2 = colors.HexColor("#384860")
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), light_bg),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (1, 0), (-1, -1), 0.25, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (0, 1), (1, 0), "LEFT"),
    ]
    for r in range(1, len(data)):
        if r%2 == 0:
            style_cmds += [
                ("BACKGROUND", (0,r), (-1,r), blue1),
                ("TEXTCOLOR", (0,r), (-1,r), colors.black),
                ("FONTNAME", (0,r), (-1,r), "Helvetica-Bold"),
            ]
        else:
            style_cmds+= [
                ("BACKGROUND", (0,r), (-1,r), blue2),
                ("TEXTCOLOR", (0,r), (-1,r), colors.white),
                ("FONTNAME", (0,r), (-1,r), "Helvetica-Bold"),
            ]
    tbl.setStyle(TableStyle(style_cmds))
    contents.append(tbl)
    doc.build(contents)
    return df

def cm_card(label,TP,FP,FN,TN,card_width):
    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        "CardHeader",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        spaceAfter=6,
    )

    data = [["", "PR Pos", "PR Neg"],["GT Pos",TP,FN],["GT Neg",FP,TN]]

    pred_col = card_width * 0.3
    gtpos_col = card_width * 0.3
    gtneg_col = card_width * 0.3
    col_widths = [pred_col, gtpos_col, gtneg_col]
    tbl = Table(data, colWidths=col_widths)

    blue = colors.HexColor("#1f4e79")
    light_bg = colors.HexColor("#f2f2f2")

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), light_bg),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (1, 0), (-1, -1), 0.25, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (0, 1), (1, 0), "LEFT"),
    ]
    colorings = [[colors.HexColor("#fff699"),(0,1)],[colors.HexColor("#a1c4fc"),(0,2)],[colors.HexColor("#b6eea7"),(1,0)],[colors.HexColor("#faada5"),(2,0)],
                 [colors.HexColor("#e9ed98"),(1,1)],[colors.HexColor("#a6f5d8"),(1,2)],[colors.HexColor("#ffcfa1"),(2,1)],[colors.HexColor("#d5bcfe"),(2,2)]]
    for e in colorings:
        style_cmds += [
            ("BACKGROUND", e[1], e[1], e[0]),
            ("TEXTCOLOR", e[1], e[1], colors.black),
            ("FONTNAME", e[1], e[1], "Helvetica-Bold"),
        ]

    lines = [
        f"<b>Label Name: </b> {label}",
        f"<b>Precision: </b> {TP/(TP+FP):.6f}",
        f"<b>Recall: </b> {TP/(TP+FN):.6f}",
        f"<b>Accuracy (P):</b> {(TP+TN)/(TP+FP+FN+TN):.6f}",
    ]
    header_par = Paragraph("<br/>".join(lines), header_style)
    
    tbl.setStyle(TableStyle(style_cmds))
    card = Table([[header_par], [tbl]], colWidths=[card_width])
    
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eeeeee")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 1), (1, 0), "LEFT"),
    ]))
    return card

def label_cm_report(train_folder, label_cm, label_map, out_path, title="Label Confusion Matrix Report"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
    )
    contents = []
    contents.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    contents.append(Spacer(1, 0.2*cm))

    page_w, page_h = A4
    usable_w = page_w - doc.leftMargin - doc.rightMargin
    img_col_w = 7.5 * cm
    card_col_w = usable_w - img_col_w

    img_max_w = img_col_w
    img_max_h = img_max_w

    for i in range(len(label_cm)):
        TP = label_cm[i,i]
        FN = sum(label_cm[i,:])-label_cm[i,i]
        FP = sum(label_cm[:,i])-label_cm[i,i]
        TN = sum(sum(label_cm[:,:]))-TP-FN-FP

        label_folder = (5-len(str(i)))*"0"+str(i)
        img_pth = os.path.join(train_folder,label_folder,f"00000_00000.ppm")
        pil = PILImage.open(img_pth).convert("RGB")
        iw, ih = pil.size
        w_pt, h_pt = fit_dims(iw, ih, img_max_w, img_max_h)
        img = Image(img_pth, width=w_pt, height=h_pt)

        card = cm_card(label_map.iloc[i]["name"], TP, FP, FN, TN, card_width=card_col_w)
        
        row = Table([[img, card]], colWidths=[img_col_w, card_col_w])
        row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))

        contents.append(row)
        contents.append(Spacer(1, 0.6 * cm))
        if i%3 == 2 and i != 2:
            contents.append(PageBreak())

    doc.build(contents)
    
