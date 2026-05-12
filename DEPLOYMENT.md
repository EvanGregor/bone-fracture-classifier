# 🚀 Deployment Guide

Complete steps to push this project to GitHub and deploy on Hugging Face Spaces.

---

## Step 1 — Push to GitHub

```bash
# 1a. Initialise git (if not already done)
git init
git add .
git commit -m "Initial commit: bone fracture classifier"

# 1b. Create a new GitHub repo (do this on github.com first, then:)
git remote add origin https://github.com/<your-username>/bone-fracture-classifier.git
git branch -M main
git push -u origin main
```

> Model weights (`*.pth`, `*.pt`) are excluded from git via `.gitignore`.
> They'll live on Hugging Face Hub instead (see Step 2).

---

## Step 2 — Upload Model Weights to Hugging Face Hub

The weights are too large for git. Store them on HF Hub and load them in `app.py`.

```bash
pip install huggingface_hub

# Edit upload_models.py: set REPO_ID = "<your-hf-username>/bone-fracture-models"
python upload_models.py --token hf_YOUR_WRITE_TOKEN
```

Then update `app.py` to download weights at startup — add this block near the top:

```python
from huggingface_hub import hf_hub_download

MODEL_DIR.mkdir(exist_ok=True)
EFFNET_CKPT = Path(hf_hub_download(
    repo_id="<your-hf-username>/bone-fracture-models",
    filename="best_effnet_v4.pth"
))
YOLO_CKPT = Path(hf_hub_download(
    repo_id="<your-hf-username>/bone-fracture-models",
    filename="best.pt"
))
```

And add `huggingface_hub` to `requirements.txt`.

---

## Step 3 — Create a Hugging Face Space

1. Go to https://huggingface.co/new-space
2. Choose:
   - **Owner**: your username
   - **Space name**: `bone-fracture-classifier`
   - **SDK**: `Gradio`
   - **Hardware**: `CPU Basic` (free) or `T4 GPU` for faster inference
3. Click **Create Space**

---

## Step 4 — Push Code to the Space

```bash
# Clone your new Space repo
git clone https://huggingface.co/spaces/<your-username>/bone-fracture-classifier hf-space
cd hf-space

# Copy project files into it
cp -r ../bone-fracture-classifier/{app.py,requirements.txt,README.md,.gitignore} .
cp -r ../bone-fracture-classifier/notebooks .   # optional
mkdir -p examples  # add sample images here

# Push to HF
git add .
git commit -m "Deploy bone fracture classifier"
git push
```

The Space will build automatically. Check the **Logs** tab on HF for any errors.

---

## Step 5 — Add Sample Images (optional but recommended)

Put a couple of example X-rays into `examples/`:
```
examples/
  sample_fracture.jpg
  sample_normal.jpg
```

The Gradio `gr.Examples` block in `app.py` will show these as one-click demos.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Space times out loading models | Switch to GPU hardware or add `@spaces.GPU` decorator |
| `ModuleNotFoundError` | Ensure all deps are in `requirements.txt` |
| Model file not found | Check HF Hub repo ID and filename in `hf_hub_download` calls |
| CUDA OOM on CPU Space | Already handled — code falls back to CPU automatically |
| Kaggle API key in notebooks | Remove `os.environ['KAGGLE_KEY']` before pushing notebooks |
