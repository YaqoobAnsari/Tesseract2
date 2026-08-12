"""Download model weights from HF Space repo during Docker build."""
import os
from huggingface_hub import hf_hub_download

WEIGHTS = [
    ("craft_mlt_25k.pth", 50),
    ("None-VGG-BiLSTM-CTC.pth", 20),
    ("door_mdl_32.pth", 200),
]

os.makedirs("/app/Model_weights", exist_ok=True)

for name, min_mb in WEIGHTS:
    path = f"/app/Model_weights/{name}"
    hf_hub_download(
        repo_id="yansari/Tesseract",
        filename=f"Model_weights/{name}",
        repo_type="space",
        local_dir="/app",
    )
    size = os.path.getsize(path)
    print(f"{name}: {size / 1e6:.1f} MB")
    if size < min_mb * 1e6:
        raise RuntimeError(f"{name} too small: {size} bytes")

print("All weights verified!")
