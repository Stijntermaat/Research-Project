import os
from pathlib import Path
import numpy as np
from PIL import Image
SRC_ROOT = Path("/scratch/s3777006/CP-CHILD/CP-CHILD-A")
DST_ROOT = Path("/scratch/s3777006/CP-CHILD/M2")

# M2 PARAMETERS
M = np.array([
    [ 1.10, -0.25,  0.00],
    [-0.35,  1.35,  0.00],
    [ 0.00, -0.15,  1.15],
], dtype=np.float32)

# Fix C: saturation boost (AFI-like color separation)
SAT = 1.30

# =========================
# MAIN LOOP
# =========================
for split in ["Train", "Test"]:
    for cls in ["Polyp", "Non-Polyp"]:
        src_dir = SRC_ROOT / split / cls
        dst_dir = DST_ROOT / split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(src_dir.glob("*.jpg"))

        for idx, src_path in enumerate(files, start=1):
            img = Image.open(src_path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)

            # --- M2.1: linear color remapping ---
            h, w, _ = arr.shape
            flat = arr.reshape(-1, 3)
            flat = flat @ M.T
            arr = flat.reshape(h, w, 3)

            # --- M2.2: saturation boost ---
            # luminance from the remapped image
            lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])[..., None]
            arr = lum + SAT * (arr - lum)

            # back to uint8
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)

            out_img = Image.fromarray(arr, mode="RGB")
            out_img.save(dst_dir / src_path.name, format="JPEG")

            if idx % 500 == 0:
                print(f"[M2] {split}/{cls}: {idx}/{len(files)} images processed")

print("M2 dataset generation finished.")
