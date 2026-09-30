from pathlib import Path

import cv2
import matplotlib.pyplot as plt


IMAGE = "RaceHorses_416x240_frame001.png"
OUT = Path("ket_qua")
OUT.mkdir(exist_ok=True)

# Bai 1: doc anh va hien thi theo cac he mau.
bgr = cv2.imread(IMAGE)
if bgr is None:
    raise FileNotFoundError(f"Khong doc duoc anh: {IMAGE}")

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
views = [
    ("RGB", rgb, None),
    ("BGR", bgr, None),
    ("YUV", cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV), None),
    ("HSV", cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), None),
    ("Gray scale", cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), "gray"),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle("Bai 1 - Doc anh va hien thi cac he mau")

for ax, (title, img, cmap) in zip(axes.flat, views):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

axes.flat[-1].axis("off")
fig.tight_layout()
fig.savefig(OUT / "bai1_doc_anh_he_mau.png", dpi=200)
fig.savefig(OUT / "bai1_doc_anh_he_mau.pdf")
plt.close(fig)
print("Da luu ket qua bai 1 trong thu muc ket_qua.")
