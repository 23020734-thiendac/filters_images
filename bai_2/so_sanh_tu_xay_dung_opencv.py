"""Y c: so sanh cac bo loc tu xay dung voi bo loc trong OpenCV."""

import csv
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from bo_loc_anh_tong_hop import loc_gaussian, loc_trung_binh, loc_trung_vi
from sobel_canny import canny_filter, sobel_filter


ROOT = Path(__file__).parent
OUTPUT = ROOT / "ket_qua_so_sanh_opencv"
IMAGES = ("noisy_image.jpg", "noisy_image2.jpg")
SIZES = (3, 5, 7)


def timed(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, (perf_counter() - start) * 1000


def metrics(custom, opencv, border=0):
    """Tinh sai so o noi vung, bo qua vien neu border > 0."""
    if border:
        custom = custom[border:-border, border:-border]
        opencv = opencv[border:-border, border:-border]
    difference = np.abs(custom.astype(float) - opencv.astype(float))
    return difference.mean(), difference.max(), (difference == 0).mean() * 100


def add_label(image, text):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return image


def save_comparison(path, original, custom, opencv, title):
    difference = cv2.absdiff(custom, opencv)
    panels = (
        add_label(original, "Anh dau vao"),
        add_label(custom, "Tu xay dung"),
        add_label(opencv, "OpenCV"),
        add_label(difference, "Sai khac tuyet doi"),
    )
    # imencode + tofile ho tro duong dan Unicode tren Windows.
    success, encoded = cv2.imencode(path.suffix, np.hstack(panels))
    if not success:
        raise RuntimeError(f"Khong ma hoa duoc anh {path.name}")
    encoded.tofile(path)


def compare_smoothing(image, image_name, rows):
    filters = {
        "Mean": (
            loc_trung_binh,
            lambda img, k: cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT_101),
        ),
        "Median": (loc_trung_vi, lambda img, k: cv2.medianBlur(img, k)),
        "Gaussian": (
            loc_gaussian,
            lambda img, k: cv2.GaussianBlur(
                img, (k, k), 0.3 * ((k - 1) / 2 - 1) + 0.8,
                borderType=cv2.BORDER_REFLECT_101,
            ),
        ),
    }

    for name, (custom_function, opencv_function) in filters.items():
        for k in SIZES:
            custom, custom_ms = timed(custom_function, image, k)
            opencv, opencv_ms = timed(opencv_function, image, k)
            mae, maximum, identical = metrics(custom, opencv, k // 2)
            rows.append((image_name, name, f"{k}x{k}", mae, maximum,
                         identical, custom_ms, opencv_ms))
            save_comparison(
                OUTPUT / f"{Path(image_name).stem}_{name.lower()}_{k}x{k}.png",
                image, custom, opencv, name,
            )


def compare_edges(image, image_name, rows, edge_rows):
    custom_sobel, custom_ms = timed(lambda: np.rint(sobel_filter(image)[0]).astype(np.uint8))

    def opencv_sobel():
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3,
                       borderType=cv2.BORDER_CONSTANT)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3,
                       borderType=cv2.BORDER_CONSTANT)
        return np.rint(np.clip(np.hypot(gx, gy), 0, 255)).astype(np.uint8)

    opencv_sobel_result, opencv_ms = timed(opencv_sobel)
    mae, maximum, identical = metrics(custom_sobel, opencv_sobel_result)
    rows.append((image_name, "Sobel", "3x3", mae, maximum,
                 identical, custom_ms, opencv_ms))
    save_comparison(OUTPUT / f"{Path(image_name).stem}_sobel.png", image,
                    custom_sobel, opencv_sobel_result, "Sobel")

    custom_canny, custom_ms = timed(canny_filter, image, 50, 150)

    def opencv_canny():
        blurred = cv2.GaussianBlur(image, (5, 5), 1.4,
                                   borderType=cv2.BORDER_CONSTANT)
        return cv2.Canny(blurred, 50, 150, apertureSize=3, L2gradient=True)

    opencv_canny_result, opencv_ms = timed(opencv_canny)
    mae, maximum, identical = metrics(custom_canny, opencv_canny_result)
    rows.append((image_name, "Canny", "5x5; 50/150", mae, maximum,
                 identical, custom_ms, opencv_ms))

    # Chi so danh rieng cho ban do canh (OpenCV duoc dung lam moc doi chieu).
    custom_edge, opencv_edge = custom_canny > 0, opencv_canny_result > 0
    intersection = np.count_nonzero(custom_edge & opencv_edge)
    custom_count = np.count_nonzero(custom_edge)
    opencv_count = np.count_nonzero(opencv_edge)
    union = np.count_nonzero(custom_edge | opencv_edge)
    precision = intersection / custom_count * 100 if custom_count else 0
    recall = intersection / opencv_count * 100 if opencv_count else 0
    iou = intersection / union * 100 if union else 100
    dice = 2 * intersection / (custom_count + opencv_count) * 100 \
        if custom_count + opencv_count else 100
    edge_rows.append((image_name, custom_count, opencv_count, precision,
                      recall, iou, dice))
    save_comparison(OUTPUT / f"{Path(image_name).stem}_canny.png", image,
                    custom_canny, opencv_canny_result, "Canny")


def main():
    OUTPUT.mkdir(exist_ok=True)
    rows, edge_rows = [], []
    for image_name in IMAGES:
        # imdecode doc duoc duong dan Unicode tren Windows on dinh hon imread.
        image = cv2.imdecode(np.fromfile(ROOT / image_name, dtype=np.uint8),
                             cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_name)
        compare_smoothing(image, image_name, rows)
        compare_edges(image, image_name, rows, edge_rows)

    csv_path = OUTPUT / "bang_so_sanh.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("Anh", "Bo loc", "Tham so", "MAE", "Sai khac lon nhat",
                         "Pixel giong nhau (%)", "Tu xay dung (ms)", "OpenCV (ms)"))
        writer.writerows(rows)

    with (OUTPUT / "bang_so_sanh_canny.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(("Anh", "So canh tu xay dung", "So canh OpenCV",
                         "Precision (%)", "Recall (%)", "IoU (%)", "Dice (%)"))
        writer.writerows(edge_rows)

    print(f"Da luu anh va bang ket qua tai: {OUTPUT.name}")
    for row in rows:
        print(f"{row[0]:16} {row[1]:8} {row[2]:12} "
              f"MAE={row[3]:7.3f}  giong={row[5]:6.2f}%")
    for row in edge_rows:
        print(f"Canny {row[0]}: Precision={row[3]:.2f}%  Recall={row[4]:.2f}%  "
              f"IoU={row[5]:.2f}%  Dice={row[6]:.2f}%")


if __name__ == "__main__":
    main()
