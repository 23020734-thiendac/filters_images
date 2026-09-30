"""Ba bo loc anh tu cai dat bang NumPy, khong dung OpenCV/SciPy."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).parent
IMAGES = [ROOT / "noisy_image.jpg", ROOT / "noisy_image2.jpg"]
SIZES = (3, 5, 7)


def windows(img, size):
    pad = size // 2
    return sliding_window_view(np.pad(img, pad, mode="reflect"), (size, size))


def mean_filter(img, size):
    return np.rint(windows(img, size).mean(axis=(-2, -1))).astype(np.uint8)


def median_filter(img, size):
    return np.median(windows(img, size), axis=(-2, -1)).astype(np.uint8)


def gaussian_filter(img, size):
    sigma = 0.3 * ((size - 1) / 2 - 1) + 0.8
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    result = np.einsum("ijkl,kl->ij", windows(img, size), kernel)
    return np.rint(result).astype(np.uint8)


FILTERS = {
    "trung_binh": mean_filter,
    "trung_vi": median_filter,
    "gaussian": gaussian_filter,
}


def run(name):
    output_dir = ROOT / "ket_qua_bo_loc" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in IMAGES:
        img = np.array(Image.open(path).convert("L"))
        results = [FILTERS[name](img, size) for size in SIZES]

        for size, result in zip(SIZES, results):
            Image.fromarray(result).save(output_dir / f"{path.stem}_{name}_{size}x{size}.png")

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ["Anh nhieu ban dau", "Kernel 3x3", "Kernel 5x5", "Kernel 7x7"]
        for ax, title, image in zip(axes, titles, [img, *results]):
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.set(title=title)
            ax.axis("off")
        fig.suptitle(f"{name.replace('_', ' ').title()} - {path.name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{path.stem}_{name}_so_sanh.png", dpi=150)
        plt.close(fig)

    print(f"Da xong bo loc {name}: {output_dir.relative_to(ROOT)}")
