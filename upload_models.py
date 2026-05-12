"""
upload_models.py
----------------
Upload your trained model weights to the Hugging Face Hub so the
Gradio Space can load them at runtime.

Usage:
    pip install huggingface_hub
    python upload_models.py --token hf_YOUR_TOKEN
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "<your-hf-username>/bone-fracture-models"   # ← change this
MODEL_FILES = [
    "models/best_effnet_v4.pth",
    "models/best.pt",
]

def main(token: str):
    api = HfApi()

    # Create model repo (model card repo, not a Space)
    create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        exist_ok=True,
        token=token,
    )
    print(f"✅ Repo ready: https://huggingface.co/{REPO_ID}")

    for fpath in MODEL_FILES:
        p = Path(fpath)
        if not p.exists():
            print(f"⚠️  {fpath} not found — skipping")
            continue
        print(f"📤 Uploading {p.name} …")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=REPO_ID,
            repo_type="model",
            token=token,
        )
        print(f"✅ {p.name} uploaded")

    print("\n🎉 Done! Update REPO_ID in app.py if you changed it above.")
    print(f"   Files live at: https://huggingface.co/{REPO_ID}/tree/main")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace API token (write access)")
    args = parser.parse_args()
    main(args.token)
