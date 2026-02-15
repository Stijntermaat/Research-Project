import os
from pathlib import Path
import numpy as np
from PIL import Image
SRC_ROOT = Path("/scratch/s3777006/CP-CHILD/CP-CHILD-A")
DST_ROOT = Path("/scratch/s3777006/CP-CHILD/M1")

# M1 PARAMETERS
A_R = 0.65
A_G = 1.15
A_B = 1.10

for split in ["Train", "Test"]:
    for cls in ["Polyp", "Non-Polyp"]:
        src_dir = SRC_ROOT / split / cls
        dst_dir = DST_ROOT / split / cls

        dst_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(src_dir.glob("*.jpg"))

        for idx, src_path in enumerate(files, start=1):
            # read image
            img = Image.open(src_path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0

            # M1: channel-wise scaling
            arr[..., 0] *= A_R  # R
            arr[..., 1] *= A_G  # G
            arr[..., 2] *= A_B  # B

            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)

            # save image
            out_img = Image.fromarray(arr, mode="RGB")
            out_img.save(dst_dir / src_path.name, format="JPEG")

            if idx % 500 == 0:
                print(f"[M1] {split}/{cls}: {idx}/{len(files)} images processed")

print("M1 dataset generation finished.")
