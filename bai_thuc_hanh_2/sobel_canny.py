"""
Tu cai dat Sobel va Canny de loc hai ma tran I, J.
OpenCV chi duoc dung de ghi chu va hien thi, khong dung filter co san.
"""

from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


I = np.array([
    [0, 0, 0, 0, 200, 200, 0, 0, 0, 0],
    [0, 0, 0, 0, 200, 200, 0, 0, 0, 0],
    [0, 0, 0, 200, 200, 200, 200, 0, 0, 0],
    [0, 0, 200, 200, 200, 200, 200, 200, 0, 0],
    [200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
    [200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
    [0, 0, 200, 200, 200, 200, 200, 200, 0, 0],
    [0, 0, 0, 200, 200, 200, 200, 0, 0, 0],
    [0, 0, 0, 0, 200, 200, 0, 0, 0, 0],
    [0, 0, 0, 0, 200, 200, 0, 0, 0, 0],
], dtype=float)

J = np.array([
    [0, 0, 100, 100],
    [0, 0, 100, 100],
    [0, 0, 100, 100],
    [0, 0, 100, 100],
], dtype=float)

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)


def convolve2d(img, kernel):
    """Truot kernel tren ma tran; ngoai bien duoc them gia tri 0."""
    p = kernel.shape[0] // 2
    padded = np.pad(img, p)
    windows = sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel)


def gradient(img):
    """Tinh gradient Sobel theo x, y, do lon va huong (0-180 do)."""
    gx, gy = convolve2d(img, SOBEL_X), convolve2d(img, SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    direction = np.degrees(np.arctan2(gy, gx)) % 180
    return magnitude, gx, gy, direction


def sobel_filter(img):
    magnitude, gx, gy, _ = gradient(img)
    return np.clip(magnitude, 0, 255), gx, gy


def gaussian_blur(img, k=5, sigma=1.4):
    """Tao kernel Gaussian k x k va lam mo ma tran."""
    x = np.arange(-k // 2 + 1, k // 2 + 1)
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return convolve2d(img, kernel)


def canny_filter(img, low_thresh=50, high_thresh=150):
    """Canny: Gaussian -> Sobel -> NMS -> nguong kep -> hysteresis."""
    magnitude, _, _, direction = gradient(gaussian_blur(img))
    h, w = img.shape

    # Non-maximum suppression: chi giu cuc dai theo huong gradient.
    nms = np.zeros_like(img)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            angle = direction[i, j]
            if angle < 22.5 or angle >= 157.5:
                a, b = magnitude[i, j - 1], magnitude[i, j + 1]
            elif angle < 67.5:
                # Huong 45 do: duong cheo tren-trai <-> duoi-phai.
                a, b = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            elif angle < 112.5:
                a, b = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                # Huong 135 do: duong cheo tren-phai <-> duoi-trai.
                a, b = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
            if magnitude[i, j] >= max(a, b):
                nms[i, j] = magnitude[i, j]

    # Nguong kep va hysteresis: giu canh yeu neu noi 8 huong voi canh manh.
    edges = np.zeros_like(img, dtype=np.uint8)
    strong = nms >= high_thresh
    weak = (nms >= low_thresh) & ~strong
    edges[strong] = 255

    # Lap den khi khong con canh yeu nao ket noi duoc voi canh manh.
    while True:
        near_strong = sliding_window_view(np.pad(edges == 255, 1), (3, 3)).any((-2, -1))
        connected = weak & near_strong
        if not connected.any():
            break
        edges[connected] = 255
        weak[connected] = False
    return edges


def scale_up(img, factor):
    """Phong to ma tran bang noi suy nearest-neighbor."""
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def label(img, text):
    """Doi anh xam sang BGR va them nhan."""
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result, (0, 0), (result.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(result, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return result


def process_matrix(matrix, name, factor):
    """Chay Sobel, Canny, in ket qua va tao anh ghep."""
    sobel = sobel_filter(matrix)[0]
    canny = canny_filter(matrix)

    print(f"\nMa tran {name}:\n{matrix.astype(int)}")
    for filter_name, result in (("Sobel", sobel), ("Canny", canny)):
        print(f"\n--- {name}: {filter_name} ---\n{np.rint(result).astype(int)}")

    images = [matrix, sobel, canny]
    names = [f"{name} - Goc", f"{name} - Sobel", f"{name} - Canny"]
    displays = [
        label(scale_up(np.clip(img, 0, 255).astype(np.uint8), factor), title)
        for img, title in zip(images, names)
    ]
    return np.hstack(displays)


def main():
    output = Path(__file__).parent / "output"
    output.mkdir(exist_ok=True)

    for matrix, name, factor in ((I, "I", 40), (J, "J", 100)):
        grid = process_matrix(matrix, name, factor)
        title = f"Ma tran {name} - Sobel/Canny"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1200, 450)
        cv2.imshow(title, grid)
        # imencode + tofile ho tro duong dan Unicode tren Windows.
        success, encoded = cv2.imencode(".png", grid)
        if not success:
            raise RuntimeError(f"Khong ma hoa duoc result_{name}.png")
        encoded.tofile(output / f"result_{name}.png")

    print("\nDa luu output/result_I.png va output/result_J.png.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
