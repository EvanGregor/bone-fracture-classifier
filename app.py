"""
Bone Fracture Classification - Hugging Face Space
Ensemble of EfficientNet (classifier) + YOLOv8 (detector)
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

# ── Model paths (downloaded from HF Hub) ─────────────────────────────────────
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
    print(f"✅ EfficientNet loaded | AUC {ckpt['val_auc']:.4f}")
    return m

print("Loading models…")
effnet_model = load_effnet(EFFNET_CKPT)
yolo_model   = YOLO(str(YOLO_CKPT))
print("✅ YOLOv8 loaded")

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
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf  = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)

# ── Main prediction function ──────────────────────────────────────────────────
def predict_xray(image):
    if image is None:
        return None, None, None, None, "⚠️ Please upload an X-ray image."

    pil_orig  = Image.fromarray(image).convert("L")
    pil_clahe = apply_clahe(pil_orig)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pil_orig.save(tmp_path)

    effnet_score              = run_effnet(pil_clahe)
    yolo_score, dets          = run_yolo(tmp_path)
    score, fracture, uncertain, agree = ensemble_decision(effnet_score, yolo_score)

    # Panel 1 – CLAHE
    clahe_np  = np.array(pil_clahe.resize((512, 512)))
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.imshow(clahe_np, cmap="gray")
    ax1.set_title("CLAHE Enhanced X-ray", fontsize=12, fontweight="bold")
    ax1.axis("off")
    plt.tight_layout()
    panel1 = fig_to_array(fig1);  plt.close(fig1)

    # Panel 2 – Grad-CAM
    target_layer = get_target_layer(effnet_model)
    gc           = GradCAM(effnet_model, target_layer)
    input_tensor = infer_transform(pil_clahe).unsqueeze(0).to(device)
    cam, _       = gc.generate(input_tensor)
    sz           = (512, 512)
    cam_resized  = cv2.resize(cam, sz)
    heatmap      = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    orig_np      = np.array(pil_clahe.resize(sz))
    orig_bgr     = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2BGR)
    overlay      = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)
    fig2, axes2  = plt.subplots(1, 2, figsize=(10, 5))
    axes2[0].imshow(cam_resized, cmap="jet");  axes2[0].set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold"); axes2[0].axis("off")
    axes2[1].imshow(overlay[..., ::-1]);       axes2[1].set_title(f"Overlay | EfficientNet: {effnet_score:.3f}", fontsize=11); axes2[1].axis("off")
    plt.tight_layout()
    panel2 = fig_to_array(fig2);  plt.close(fig2)

    # Panel 3 – YOLO detections
    orig_for_yolo = np.array(Image.open(tmp_path).convert("RGB"))
    fig3, ax3     = plt.subplots(figsize=(5, 5))
    ax3.imshow(orig_for_yolo)
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=3, edgecolor="red", facecolor="none")
        ax3.add_patch(rect)
        ax3.text(x1, max(y1-10, 0), f"fractured {det['confidence']:.2f}",
                 color="white", fontsize=10, fontweight="bold",
                 bbox=dict(facecolor="red", alpha=0.8, edgecolor="none", pad=2))
    n_det = len(dets)
    ax3.set_title(f"YOLO: {n_det} region(s) | Score: {yolo_score:.3f}",
                  fontsize=11, fontweight="bold",
                  color="red" if n_det > 0 else "green")
    ax3.axis("off");  plt.tight_layout()
    panel3 = fig_to_array(fig3);  plt.close(fig3)

    # Panel 4 – Ensemble result
    if uncertain:
        verdict, verdict_sub, bar_color = "⚠️  UNCERTAIN", "Radiologist review required", "#FFA500"
    elif fracture:
        verdict, verdict_sub, bar_color = "🔴  FRACTURE DETECTED", "Abnormal X-ray", "#FF4444"
    else:
        verdict, verdict_sub, bar_color = "🟢  NORMAL", "No fracture detected", "#44BB44"

    fig4, ax4 = plt.subplots(figsize=(6, 5))
    ax4.axis("off")
    metrics = [
        ("EfficientNet", effnet_score, "#4488FF"),
        ("YOLO",         yolo_score,   "#FF8844"),
        ("Ensemble",     score,        bar_color),
    ]
    for (label, val, color), y in zip(metrics, [0.75, 0.55, 0.30]):
        ax4.barh(y, val, height=0.12, color=color, alpha=0.85)
        ax4.text(-0.02, y, label, ha="right", va="center", fontsize=11)
        ax4.text(val + 0.02, y, f"{val:.3f}", ha="left", va="center", fontsize=11, fontweight="bold")
    ax4.set_xlim(-0.3, 1.2);  ax4.set_ylim(0.1, 1.0)
    ax4.axvline(x=ENSEMBLE_CONFIG["final_threshold"], color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax4.text(ENSEMBLE_CONFIG["final_threshold"], 0.15,
             f"threshold={ENSEMBLE_CONFIG['final_threshold']}", ha="center", fontsize=8, color="gray")
    ax4.text(0.5, 0.95, verdict, ha="center", va="top",
             fontsize=16, fontweight="bold", color=bar_color, transform=ax4.transAxes)
    ax4.text(0.5, 0.84, verdict_sub, ha="center", va="top",
             fontsize=11, color="gray", transform=ax4.transAxes)
    ax4.text(0.5, 0.07, f"Models {'agree ✓' if agree else 'disagree ✗'}",
             ha="center", fontsize=10, color="green" if agree else "orange", transform=ax4.transAxes)
    plt.tight_layout()
    panel4 = fig_to_array(fig4);  plt.close(fig4)

    # Text report
    report = (
        f"{'='*45}\n"
        f"  BONE FRACTURE ANALYSIS REPORT\n"
        f"{'='*45}\n"
        f"  Verdict      : {verdict}\n"
        f"  Ensemble Score : {score:.4f}  (threshold {ENSEMBLE_CONFIG['final_threshold']})\n"
        f"\n  Model Scores:\n"
        f"    EfficientNet : {effnet_score:.4f}  (threshold {ENSEMBLE_CONFIG['effnet_threshold']})\n"
        f"    YOLOv8       : {yolo_score:.4f}  ({n_det} region(s) detected)\n"
        f"\n  Agreement    : {'✅ Models agree' if agree else '⚠️  Models disagree'}\n"
        f"  Uncertainty  : {'Yes — radiologist review advised' if uncertain else 'No'}\n"
        f"{'='*45}\n"
        f"  ⚠️  For research / educational use only.\n"
        f"  Not a substitute for professional diagnosis.\n"
        f"{'='*45}"
    )

    os.unlink(tmp_path)
    return panel1, panel2, panel3, panel4, report

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Bone Fracture Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🦴 Bone Fracture Classifier
        ### EfficientNet + YOLOv8 Ensemble with Grad-CAM Explainability

        Upload a bone X-ray and the model will classify it as **Fractured** or **Normal**,
        localise fracture regions, and show a Grad-CAM heatmap for transparency.

        > ⚠️ **Disclaimer:** For research and educational use only. Not a substitute for clinical diagnosis.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(label="Upload X-ray Image", type="numpy")
            run_btn   = gr.Button("🔍 Analyse", variant="primary", size="lg")

        with gr.Column(scale=2):
            report_box = gr.Textbox(label="Analysis Report", lines=14, interactive=False)

    with gr.Row():
        panel1_out = gr.Image(label="CLAHE Enhanced")
        panel2_out = gr.Image(label="Grad-CAM Heatmap")

    with gr.Row():
        panel3_out = gr.Image(label="YOLO Detections")
        panel4_out = gr.Image(label="Ensemble Scores")

    run_btn.click(
        fn=predict_xray,
        inputs=[img_input],
        outputs=[panel1_out, panel2_out, panel3_out, panel4_out, report_box],
    )

    gr.Examples(
        examples=[["examples/sample_fracture.jpg"], ["examples/sample_normal.jpg"]],
        inputs=[img_input],
        label="Sample Images",
    )

if __name__ == "__main__":
    demo.launch()
