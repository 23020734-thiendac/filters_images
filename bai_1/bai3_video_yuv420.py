from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


YUV_FILE = "RaceHorses_416x240_20.yuv"
WIDTH = 416
HEIGHT = 240
OUT = Path("ket_qua")
RGB_DIR = OUT / "bai3_yuv_rgb_frames"
NEG_DIR = OUT / "bai3_yuv_negative_frames"
OUT.mkdir(exist_ok=True)
RGB_DIR.mkdir(exist_ok=True)
NEG_DIR.mkdir(exist_ok=True)

# Bai 3: doc video YUV420, doi sang RGB, roi bien doi tung frame.
frame_size = WIDTH * HEIGHT * 3 // 2
data = np.fromfile(YUV_FILE, dtype=np.uint8)

if data.size % frame_size != 0:
    raise ValueError("Sai kich thuoc video hoac khong phai YUV420.")

frame_count = data.size // frame_size
samples = []

for i in range(frame_count):
    start = i * frame_size
    yuv = data[start : start + frame_size].reshape((HEIGHT * 3 // 2, WIDTH))
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
    negative = 255 - rgb

    cv2.imwrite(str(RGB_DIR / f"frame_{i:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(NEG_DIR / f"frame_{i:03d}.png"),
        cv2.cvtColor(negative, cv2.COLOR_RGB2BGR),
    )

    if i in {0, frame_count // 2, frame_count - 1}:
        samples.append((i, rgb, negative))

fig, axes = plt.subplots(2, len(samples), figsize=(12, 6))
fig.suptitle("Bai 3 - YUV420 -> RGB va bien doi 255 - pixel")

for col, (i, rgb, negative) in enumerate(samples):
    axes[0, col].imshow(rgb)
    axes[0, col].set_title(f"RGB frame {i}")
    axes[0, col].axis("off")

    axes[1, col].imshow(negative)
    axes[1, col].set_title(f"255 - frame {i}")
    axes[1, col].axis("off")

fig.tight_layout()
fig.savefig(OUT / "bai3_yuv420_rgb_va_bien_doi.png", dpi=200)
fig.savefig(OUT / "bai3_yuv420_rgb_va_bien_doi.pdf")
plt.close(fig)
print(f"Da xu ly {frame_count} frame. Ket qua nam trong thu muc ket_qua.")
