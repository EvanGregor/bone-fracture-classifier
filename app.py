"""
Bone Fracture Classification - Hugging Face Space
Ensemble of EfficientNet (classifier) + YOLOv8 (detector)
Redesigned UI: clinical dark-mode dashboard aesthetic
"""

import os, cv2, warnings, tempfile
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms
import timm
from ultralytics import YOLO
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# ── Model paths ───────────────────────────────────────────────────────────────
EFFNET_CKPT = Path(hf_hub_download(repo_id="Evangregor/bone-fracture-models", filename="best_effnet_v4.pth"))
YOLO_CKPT   = Path(hf_hub_download(repo_id="Evangregor/bone-fracture-models", filename="best.pt"))

ENSEMBLE_CONFIG = {
    "effnet_threshold" : 0.6623,
    "yolo_conf"        : 0.15,
    "yolo_iou"         : 0.45,
    "yolo_imgsz"       : 640,
    "effnet_weight"    : 0.45,
    "yolo_weight"      : 0.55,
    "final_threshold"  : 0.4355,
    "use_clahe"        : True,
    "border_crop_pct"  : 0.04,
    "img_size"         : 260,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"  : "#0D1117",
    "axes.facecolor"    : "#0D1117",
    "axes.edgecolor"    : "#30363D",
    "axes.labelcolor"   : "#8B949E",
    "xtick.color"       : "#8B949E",
    "ytick.color"       : "#8B949E",
    "text.color"        : "#E6EDF3",
    "grid.color"        : "#21262D",
    "grid.linewidth"    : 0.5,
    "font.family"       : "monospace",
})

# ── Preprocessing ─────────────────────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((ENSEMBLE_CONFIG["img_size"], ENSEMBLE_CONFIG["img_size"])),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def remove_xray_border(pil_img, pct=None):
    pct   = pct or ENSEMBLE_CONFIG["border_crop_pct"]
    img   = np.array(pil_img)
    h, w  = img.shape[:2]
    mh, mw = int(h * pct), int(w * pct)
    return Image.fromarray(img[mh:h-mh, mw:w-mw])

def apply_clahe(pil_img):
    img_np   = np.array(pil_img.convert("L"))
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    return remove_xray_border(Image.fromarray(enhanced))

# ── Model loading ─────────────────────────────────────────────────────────────
def load_effnet(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    m    = timm.create_model(
        cfg["model_name"], pretrained=False,
        num_classes=cfg["num_classes"], drop_rate=cfg["drop_rate"]
    )
    m.load_state_dict(ckpt["model_state"])
    m = m.to(device).eval()
    print(f"EfficientNet loaded | AUC {ckpt['val_auc']:.4f}")
    return m

print("Loading models…")
effnet_model = load_effnet(EFFNET_CKPT)
yolo_model   = YOLO(str(YOLO_CKPT))
print("YOLOv8 loaded")

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o)
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )

    def generate(self, input_tensor):
        self.model.eval()
        out  = self.model(input_tensor)
        cls  = out.argmax(dim=1).item()
        self.model.zero_grad()
        out[0, cls].backward(retain_graph=True)
        weights    = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam        = (weights * self.activations).sum(dim=1, keepdim=True)
        cam        = torch.relu(cam).squeeze().detach().cpu().numpy()
        cam        = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs      = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
        return cam, probs

def get_target_layer(model):
    try:    return model.blocks[-1][-1].conv_pwl
    except: return model.blocks[-1]

# ── Inference helpers ─────────────────────────────────────────────────────────
def run_effnet(image_clahe):
    tensor = infer_transform(image_clahe).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(effnet_model(tensor), dim=1)[0].cpu().numpy()
    return float(probs[1])

def run_yolo(img_path):
    results    = yolo_model.predict(
        source=img_path,
        conf=ENSEMBLE_CONFIG["yolo_conf"],
        iou=ENSEMBLE_CONFIG["yolo_iou"],
        imgsz=ENSEMBLE_CONFIG["yolo_imgsz"],
        verbose=False,
    )
    result     = results[0]
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cf              = float(box.conf[0].cpu())
            detections.append({
                "confidence": round(cf, 3),
                "bbox"      : [int(x1), int(y1), int(x2), int(y2)],
            })
    yolo_score = max((d["confidence"] for d in detections), default=0.0)
    return yolo_score, detections

def ensemble_decision(effnet_score, yolo_score):
    w_e   = ENSEMBLE_CONFIG["effnet_weight"]
    w_y   = ENSEMBLE_CONFIG["yolo_weight"]
    score = w_e * effnet_score + w_y * yolo_score

    effnet_pred  = effnet_score >= ENSEMBLE_CONFIG["effnet_threshold"]
    yolo_pred    = yolo_score   >= ENSEMBLE_CONFIG["yolo_conf"]
    models_agree = effnet_pred  == yolo_pred
    uncertain    = not models_agree or 0.40 <= score <= 0.65
    fracture     = score >= ENSEMBLE_CONFIG["final_threshold"]
    return score, fracture, uncertain, models_agree

def fig_to_array(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return np.array(img)

def styled_title(ax, title, subtitle=None, color="#58A6FF"):
    ax.set_title(title, fontsize=11, color=color, fontweight="bold",
                 loc="left", pad=10, fontfamily="monospace")
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes,
                fontsize=8, color="#8B949E", va="bottom", ha="left",
                fontfamily="monospace")

# ── Main prediction function ──────────────────────────────────────────────────
def predict_xray(image):
    if image is None:
        return None, None, None, None, "⚠  Upload an X-ray image to begin analysis."

    pil_orig  = Image.fromarray(image).convert("L")
    pil_clahe = apply_clahe(pil_orig)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pil_orig.save(tmp_path)

    effnet_score              = run_effnet(pil_clahe)
    yolo_score, dets          = run_yolo(tmp_path)
    score, fracture, uncertain, agree = ensemble_decision(effnet_score, yolo_score)
    n_det = len(dets)

    # ── Verdict colours ───────────────────────────────────────────────────────
    if uncertain:
        verdict_label = "UNCERTAIN"
        verdict_sub   = "Radiologist review required"
        accent        = "#E3B341"   # amber
        icon          = "⚠"
    elif fracture:
        verdict_label = "FRACTURE DETECTED"
        verdict_sub   = "Abnormal — seek clinical evaluation"
        accent        = "#F85149"   # red
        icon          = "✕"
    else:
        verdict_label = "NO FRACTURE"
        verdict_sub   = "Normal — no abnormality detected"
        accent        = "#3FB950"   # green
        icon          = "✓"

    BG   = "#0D1117"
    SURF = "#161B22"
    GRID = "#21262D"
    DIM  = "#8B949E"
    HI   = "#E6EDF3"

    # ── Panel 1: CLAHE ────────────────────────────────────────────────────────
    clahe_np  = np.array(pil_clahe.resize((480, 480)))
    fig1, ax1 = plt.subplots(figsize=(5, 5.4))
    fig1.patch.set_facecolor(BG)
    ax1.set_facecolor(BG)
    ax1.imshow(clahe_np, cmap="bone")
    styled_title(ax1, "CLAHE Enhancement", "Contrast-limited adaptive histogram equalization")
    ax1.axis("off")
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID)
    plt.tight_layout(pad=0.8)
    panel1 = fig_to_array(fig1);  plt.close(fig1)

    # ── Panel 2: Grad-CAM ─────────────────────────────────────────────────────
    target_layer = get_target_layer(effnet_model)
    gc           = GradCAM(effnet_model, target_layer)
    input_tensor = infer_transform(pil_clahe).unsqueeze(0).to(device)
    cam, _       = gc.generate(input_tensor)
    sz           = (480, 480)
    cam_resized  = cv2.resize(cam, sz)
    heatmap      = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_INFERNO)
    orig_np      = np.array(pil_clahe.resize(sz))
    orig_bgr     = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2BGR)
    overlay      = cv2.addWeighted(orig_bgr, 0.5, heatmap, 0.5, 0)

    fig2, axes2  = plt.subplots(1, 2, figsize=(10, 5.4))
    fig2.patch.set_facecolor(BG)
    for ax in axes2:
        ax.set_facecolor(BG)
        ax.axis("off")

    axes2[0].imshow(cam_resized, cmap="inferno")
    styled_title(axes2[0], "Grad-CAM Heatmap", f"EfficientNet attention  ·  score {effnet_score:.3f}")

    axes2[1].imshow(overlay[..., ::-1])
    styled_title(axes2[1], "Overlay", "Heatmap blended onto CLAHE image")

    # Thin colourbar
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig2.colorbar(sm, ax=axes2[0], fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=DIM)
    cb.outline.set_edgecolor(GRID)

    plt.tight_layout(pad=0.8)
    panel2 = fig_to_array(fig2);  plt.close(fig2)

    # ── Panel 3: YOLO detections ──────────────────────────────────────────────
    orig_for_yolo = np.array(Image.open(tmp_path).convert("RGB"))
    fig3, ax3     = plt.subplots(figsize=(5, 5.4))
    fig3.patch.set_facecolor(BG)
    ax3.set_facecolor(BG)
    ax3.imshow(orig_for_yolo)
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        rect = patches.FancyBboxPatch(
            (x1, y1), x2-x1, y2-y1,
            boxstyle="square,pad=0",
            linewidth=2, edgecolor="#F85149", facecolor=(0.97, 0.32, 0.29, 0.08)
        )
        ax3.add_patch(rect)
        ax3.text(x1 + 4, y1 + 18, f"{det['confidence']:.2f}",
                 color="#F85149", fontsize=9, fontweight="bold",
                 fontfamily="monospace",
                 bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=2))

    det_color = "#F85149" if n_det > 0 else "#3FB950"
    styled_title(ax3, "YOLOv8 Detections",
                 f"{n_det} region(s) localised  ·  max conf {yolo_score:.3f}", color=det_color)
    ax3.axis("off")
    plt.tight_layout(pad=0.8)
    panel3 = fig_to_array(fig3);  plt.close(fig3)

    # ── Panel 4: Ensemble summary ──────────────────────────────────────────────
    fig4 = plt.figure(figsize=(6, 5.4))
    fig4.patch.set_facecolor(BG)

    # Big verdict block at top
    ax_top = fig4.add_axes([0.0, 0.72, 1.0, 0.28])
    ax_top.set_facecolor(SURF)
    ax_top.axis("off")
    ax_top.add_patch(patches.FancyBboxPatch(
        (0.01, 0.04), 0.98, 0.92,
        boxstyle="square,pad=0",
        facecolor=SURF, edgecolor=accent, linewidth=1.5,
        transform=ax_top.transAxes, clip_on=False
    ))
    ax_top.text(0.5, 0.72, f"{icon}  {verdict_label}", ha="center", va="center",
                fontsize=18, fontweight="bold", color=accent,
                transform=ax_top.transAxes, fontfamily="monospace")
    ax_top.text(0.5, 0.28, verdict_sub, ha="center", va="center",
                fontsize=9, color=DIM, transform=ax_top.transAxes,
                fontfamily="monospace")

    # Score bars
    ax_bar = fig4.add_axes([0.08, 0.10, 0.84, 0.58])
    ax_bar.set_facecolor(BG)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.2, 3.2)
    ax_bar.axis("off")

    models = [
        ("ENSEMBLE",     score,        accent,   ENSEMBLE_CONFIG["final_threshold"]),
        ("YOLOv8",       yolo_score,   "#79C0FF", ENSEMBLE_CONFIG["yolo_conf"]),
        ("EfficientNet", effnet_score, "#56D364", ENSEMBLE_CONFIG["effnet_threshold"]),
    ]

    for i, (label, val, clr, thresh) in enumerate(models):
        y = i * 0.95 + 0.2
        bar_h = 0.34
        # Track
        ax_bar.add_patch(patches.FancyBboxPatch(
            (0, y), 1.0, bar_h,
            boxstyle="square,pad=0",
            facecolor=GRID, edgecolor="none"
        ))
        # Fill
        ax_bar.add_patch(patches.FancyBboxPatch(
            (0, y), val, bar_h,
            boxstyle="square,pad=0",
            facecolor=clr, edgecolor="none", alpha=0.85
        ))
        # Threshold tick
        ax_bar.axvline(x=thresh, ymin=(y) / 3.2, ymax=(y + bar_h) / 3.2,
                       color="#30363D", linewidth=1.5, linestyle="--")
        # Labels
        ax_bar.text(-0.02, y + bar_h / 2, label,
                    ha="right", va="center", fontsize=9, color=HI,
                    fontfamily="monospace", fontweight="bold")
        ax_bar.text(val + 0.02 if val < 0.88 else val - 0.04,
                    y + bar_h / 2, f"{val:.3f}",
                    ha="left" if val < 0.88 else "right",
                    va="center", fontsize=9, color=clr,
                    fontfamily="monospace", fontweight="bold")
        ax_bar.text(thresh, y - 0.08, f"τ={thresh}", ha="center",
                    fontsize=6.5, color=DIM, fontfamily="monospace")

    # Agreement badge
    agree_txt = "models agree" if agree else "models disagree"
    agree_clr = "#3FB950" if agree else "#E3B341"
    ax_bar.text(0.5, -0.15, f"{'✓' if agree else '⚠'}  {agree_txt}",
                ha="center", va="center", fontsize=9,
                color=agree_clr, fontfamily="monospace",
                transform=ax_bar.transAxes)

    plt.tight_layout(pad=0)
    panel4 = fig_to_array(fig4);  plt.close(fig4)

    # ── Text report ───────────────────────────────────────────────────────────
    agree_str = "agree" if agree else "disagree"
    uncertain_str = "yes — radiologist review advised" if uncertain else "no"
    report = (
        f"┌{'─'*43}┐\n"
        f"│{'  BONE FRACTURE ANALYSIS REPORT':^43}│\n"
        f"├{'─'*43}┤\n"
        f"│  verdict        {verdict_label:<25}│\n"
        f"│  ensemble score {score:.4f}  (τ = {ENSEMBLE_CONFIG['final_threshold']})     │\n"
        f"├{'─'*43}┤\n"
        f"│  EfficientNet   {effnet_score:.4f}  (τ = {ENSEMBLE_CONFIG['effnet_threshold']})    │\n"
        f"│  YOLOv8         {yolo_score:.4f}  ({n_det} region(s))          │\n"
        f"├{'─'*43}┤\n"
        f"│  agreement      {agree_str:<25}│\n"
        f"│  uncertain      {uncertain_str:<25}│\n"
        f"├{'─'*43}┤\n"
        f"│  ⚠  Research / educational use only.    │\n"
        f"│     Not a substitute for diagnosis.     │\n"
        f"└{'─'*43}┘"
    )

    os.unlink(tmp_path)
    return panel1, panel2, panel3, panel4, report


# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
/* ── Global ─────────────────────────────────────── */
:root {
    --bg-primary   : #0D1117;
    --bg-surface   : #161B22;
    --bg-elevated  : #21262D;
    --border-dim   : #30363D;
    --text-primary : #E6EDF3;
    --text-muted   : #8B949E;
    --accent-blue  : #58A6FF;
    --accent-green : #3FB950;
    --accent-red   : #F85149;
    --accent-amber : #E3B341;
    --mono         : 'JetBrains Mono', 'Fira Code', 'Cascadia Code', ui-monospace, monospace;
}
body, .gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
}

/* ── Header ──────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 20px;
    border-bottom: 1px solid var(--border-dim);
    margin-bottom: 24px;
}
.app-header .logo {
    font-size: 36px;
    line-height: 1;
}
.app-header .titles h1 {
    margin: 0;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--text-primary);
}
.app-header .titles p {
    margin: 4px 0 0;
    font-size: 12px;
    color: var(--text-muted);
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    border: 1px solid;
    margin-left: 6px;
    vertical-align: middle;
}
.badge-blue  { color: var(--accent-blue);  border-color: var(--accent-blue);  }
.badge-green { color: var(--accent-green); border-color: var(--accent-green); }

/* ── Section labels ──────────────────────────────── */
.section-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-dim);
}

/* ── Upload area ─────────────────────────────────── */
.upload-area .wrap {
    background  : var(--bg-surface) !important;
    border      : 1px dashed var(--border-dim) !important;
    border-radius: 8px !important;
    min-height  : 300px;
    transition  : border-color 0.2s;
}
.upload-area .wrap:hover {
    border-color: var(--accent-blue) !important;
}

/* ── Analyse button ──────────────────────────────── */
#analyse-btn {
    width       : 100%;
    margin-top  : 12px;
    background  : var(--accent-blue) !important;
    color       : #0D1117 !important;
    border      : none !important;
    border-radius: 6px !important;
    font-size   : 13px !important;
    font-weight : 700 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    height      : 44px !important;
    cursor      : pointer;
    transition  : opacity 0.15s;
}
#analyse-btn:hover { opacity: 0.85; }

/* ── Report textbox ──────────────────────────────── */
#report-box textarea {
    background  : var(--bg-surface) !important;
    color       : var(--accent-green) !important;
    font-family : var(--mono) !important;
    font-size   : 12px !important;
    line-height : 1.6 !important;
    border      : 1px solid var(--border-dim) !important;
    border-radius: 6px !important;
    padding     : 14px !important;
}
#report-box label { color: var(--text-muted) !important; font-size: 10px !important; letter-spacing: 1px; text-transform: uppercase; }

/* ── Image panels ────────────────────────────────── */
.panel-img .wrap {
    background  : var(--bg-surface) !important;
    border      : 1px solid var(--border-dim) !important;
    border-radius: 8px !important;
    overflow    : hidden;
}
.panel-img label { color: var(--text-muted) !important; font-size: 10px !important; letter-spacing: 1px; text-transform: uppercase; }

/* ── Examples ────────────────────────────────────── */
.examples-holder table { background: var(--bg-surface) !important; border: 1px solid var(--border-dim) !important; border-radius: 6px !important; }
.examples-holder td    { color: var(--text-muted) !important; font-size: 12px !important; }
.examples-holder td:hover { color: var(--accent-blue) !important; cursor: pointer; }

/* ── Disclaimer bar ──────────────────────────────── */
.disclaimer {
    margin-top  : 28px;
    padding     : 10px 16px;
    background  : rgba(232,179,65,0.08);
    border-left : 3px solid var(--accent-amber);
    border-radius: 0 6px 6px 0;
    font-size   : 11px;
    color       : var(--accent-amber);
    letter-spacing: 0.3px;
}

/* ── Gradio chrome overrides ─────────────────────── */
footer { display: none !important; }
.gr-prose h1, .gr-prose h2, .gr-prose h3 { color: var(--text-primary) !important; }
"""

HEADER_HTML = """
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<div class="app-header">
    <div class="logo">🦴</div>
    <div class="titles">
        <h1>Bone Fracture Classifier
            <span class="badge badge-blue">EfficientNet</span>
            <span class="badge badge-green">YOLOv8</span>
        </h1>
        <p>Ensemble model with Grad-CAM explainability &nbsp;·&nbsp; Research use only</p>
    </div>
</div>
"""

DISCLAIMER_HTML = """
<div class="disclaimer">
    ⚠&nbsp; This tool is intended for <strong>research and educational purposes only</strong>.
    Results must not be used as a substitute for professional clinical diagnosis.
    Always consult a qualified radiologist or physician.
</div>
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Bone Fracture Classifier") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        # ── Left column: input + report ───────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<p class="section-label">Input</p>')
            img_input = gr.Image(
                label="Upload X-ray",
                type="numpy",
                elem_classes=["upload-area"],
                height=300,
            )
            run_btn = gr.Button(
                "Analyse →",
                variant="primary",
                elem_id="analyse-btn",
            )
            gr.HTML('<p class="section-label" style="margin-top:20px">Report</p>')
            report_box = gr.Textbox(
                label="Analysis Output",
                lines=15,
                interactive=False,
                elem_id="report-box",
                placeholder="Run analysis to see results here…",
            )

        # ── Right column: 2×2 panel grid ─────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML('<p class="section-label">Visualisations</p>')
            with gr.Row():
                panel1_out = gr.Image(label="CLAHE Enhanced",     elem_classes=["panel-img"], show_download_button=False)
                panel4_out = gr.Image(label="Ensemble Scores",    elem_classes=["panel-img"], show_download_button=False)
            with gr.Row():
                panel2_out = gr.Image(label="Grad-CAM Heatmap",  elem_classes=["panel-img"], show_download_button=False)
                panel3_out = gr.Image(label="YOLOv8 Detections", elem_classes=["panel-img"], show_download_button=False)

    gr.HTML('<p class="section-label" style="margin-top:24px">Examples</p>')
    gr.Examples(
        examples=[["examples/sample_fracture.jpg"], ["examples/sample_normal.jpg"]],
        inputs=[img_input],
        label="",
        elem_classes=["examples-holder"],
    )

    gr.HTML(DISCLAIMER_HTML)

    run_btn.click(
        fn=predict_xray,
        inputs=[img_input],
        outputs=[panel1_out, panel2_out, panel3_out, panel4_out, report_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")