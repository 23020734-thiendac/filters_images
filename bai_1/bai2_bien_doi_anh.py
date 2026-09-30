from pathlib import Path

import cv2
import matplotlib.pyplot as plt


IMAGE = "RaceHorses_416x240_frame001.png"
OUT = Path("ket_qua")
OUT.mkdir(exist_ok=True)

# Bai 2: bien doi anh theo cong thuc pixel_moi = 255 - pixel_cu.
bgr = cv2.imread(IMAGE)
if bgr is None:
    raise FileNotFoundError(f"Khong doc duoc anh: {IMAGE}")

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
negative = 255 - rgb

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Bai 2 - Bien doi anh 255 - pixel")

axes[0].imshow(rgb)
axes[0].set_title("Anh goc / RGB")
axes[0].axis("off")

axes[1].imshow(negative)
axes[1].set_title("Anh bien doi")
axes[1].axis("off")

fig.tight_layout()
fig.savefig(OUT / "bai2_bien_doi_255_pixel.png", dpi=200)
fig.savefig(OUT / "bai2_bien_doi_255_pixel.pdf")
plt.close(fig)
print("Da luu ket qua bai 2 trong thu muc ket_qua.")
