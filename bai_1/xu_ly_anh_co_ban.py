import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Cau hinh mac dinh: ten file anh, ten file video YUV va kich thuoc video.
DEFAULT_IMAGE = "RaceHorses_416x240_frame001.png"
DEFAULT_YUV_FILE = "RaceHorses_416x240_20.yuv"
DEFAULT_YUV_WIDTH = 416
DEFAULT_YUV_HEIGHT = 240


def read_image_bgr(image_path: Path):
    # Doc anh bang OpenCV. OpenCV mac dinh doc anh mau theo thu tu BGR.
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")
    return img_bgr


def build_color_views(image_path: Path):
    # Yeu cau 1: doc anh co san va chuyen sang cac he mau can hien thi.
    img_bgr = read_image_bgr(image_path)

    # Matplotlib hien thi dung theo RGB, nen can doi BGR -> RGB.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return {
        "Anh goc / RGB": img_rgb,
        "BGR": img_bgr,
        "YUV": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV),
        "HSV": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV),
        "Gray scale": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
    }


def build_rgb_channel_views(image_path: Path):
    # Phan bo sung: tach rieng 3 kenh mau R, G, B cua anh RGB.
    img_bgr = read_image_bgr(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Sau khi split, moi kenh chi con la anh 1 kenh nen hien thi dang xam.
    r_channel, g_channel, b_channel = cv2.split(img_rgb)

    return {
        "Anh goc / RGB": img_rgb,
        "Kenh R": r_channel,
        "Kenh G": g_channel,
        "Kenh B": b_channel,
    }


def build_negative_image_views(image_path: Path):
    # Yeu cau 2: bien doi anh theo cong thuc pixel_moi = 255 - pixel_cu.
    img_bgr = read_image_bgr(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Tru tung gia tri diem anh cho 255 de tao anh am ban.
    negative_rgb = 255 - img_rgb

    return {
        "Anh goc / RGB": img_rgb,
        "Anh bien doi = 255 - anh goc": negative_rgb,
    }


def read_yuv420_frame(file_path: Path, width: int, height: int, frame_index: int):
    # Yeu cau 3: doc 1 frame tu file video raw YUV420.
    # YUV420 co kich thuoc 1 frame = width * height * 3 / 2 byte.
    frame_size = width * height * 3 // 2
    offset = frame_index * frame_size

    # File .yuv la file raw, nen phai tu nhay den vi tri frame can doc.
    with file_path.open("rb") as file:
        file.seek(offset)
        frame_data = file.read(frame_size)

    if len(frame_data) != frame_size:
        raise ValueError(
            f"Khong doc duoc frame {frame_index}. "
            f"Can {frame_size} bytes nhung chi doc duoc {len(frame_data)} bytes."
        )

    # Doi day byte thanh mang numpy de OpenCV xu ly.
    yuv_frame = np.frombuffer(frame_data, dtype=np.uint8)

    # I420/YUV420p duoc OpenCV xem nhu anh co height * 3 / 2 dong.
    yuv_frame = yuv_frame.reshape((height * 3 // 2, width))

    # Chuyen frame tu YUV420 sang RGB de hien thi va bien doi anh.
    return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB_I420)


def count_yuv420_frames(file_path: Path, width: int, height: int):
    # Tinh so frame trong file YUV dua tren dung luong file.
    frame_size = width * height * 3 // 2
    file_size = file_path.stat().st_size

    # Neu dung luong file khong chia het cho kich thuoc 1 frame thi thong so sai.
    if file_size % frame_size != 0:
        raise ValueError(
            f"Kich thuoc file {file_size} bytes khong chia het cho "
            f"khi co frame YUV420 {frame_size} bytes."
        )

    return file_size // frame_size


def process_yuv420_video(file_path: Path, width: int, height: int, output_dir: Path):
    # Yeu cau 3: doc toan bo video YUV420, doi sang RGB, roi bien doi tung frame.
    frame_count = count_yuv420_frames(file_path, width, height)

    # Tao 2 thu muc ket qua: frame RGB va frame sau khi bien doi 255 - pixel.
    rgb_dir = output_dir / "yuv_rgb_frames"
    negative_dir = output_dir / "yuv_negative_frames"
    rgb_dir.mkdir(exist_ok=True)
    negative_dir.mkdir(exist_ok=True)

    # Lay 3 frame dai dien de ve anh tong hop: dau, giua, cuoi.
    sample_frames = []
    for frame_index in range(frame_count):
        # Doc 1 frame YUV420 va chuyen sang RGB.
        rgb_frame = read_yuv420_frame(file_path, width, height, frame_index)

        # Ap dung cong thuc cua yeu cau 2 cho tung frame video.
        negative_frame = 255 - rgb_frame

        rgb_output = rgb_dir / f"frame_{frame_index:03d}.png"
        negative_output = negative_dir / f"frame_{frame_index:03d}.png"

        # cv2.imwrite can anh theo BGR, nen phai doi RGB -> BGR truoc khi luu.
        cv2.imwrite(str(rgb_output), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            str(negative_output), cv2.cvtColor(negative_frame, cv2.COLOR_RGB2BGR)
        )

        if frame_index in {0, frame_count // 2, frame_count - 1}:
            sample_frames.append((frame_index, rgb_frame, negative_frame))

    return frame_count, sample_frames


def draw_result(views: dict[str, object], image_name: str):
    # Ve anh tong hop cho yeu cau 1: RGB, BGR, YUV, HSV va gray scale.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(f"Thuc hanh xu ly anh co ban - {image_name}", fontsize=16)

    items = list(views.items())
    for ax, (title, image) in zip(axes.flat, items):
        # Anh xam can cmap="gray" de hien thi dung dang den trang.
        if title == "Gray scale":
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    axes.flat[-1].axis("off")
    fig.tight_layout()
    return fig


def draw_negative_image(views: dict[str, object], image_name: str):
    # Ve anh goc va anh sau bien doi am ban cho yeu cau 2.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Bien doi diem anh: gia tri moi = 255 - gia tri cu - {image_name}")

    for ax, (title, image) in zip(axes.flat, views.items()):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    return fig


def draw_rgb_channels(views: dict[str, object], image_name: str):
    # Ve anh goc va 3 kenh mau R, G, B da tach rieng.
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(f"Tach tung kenh mau R - G - B - {image_name}", fontsize=15)

    for ax, (title, image) in zip(axes.flat, views.items()):
        if title == "Anh goc / RGB":
            ax.imshow(image)
        else:
            # Kenh R/G/B rieng le la anh 1 kenh, nen hien thi dang xam.
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    return fig


def draw_yuv_video_summary(sample_frames, file_name: str):
    # Ve anh tong hop minh hoa video YUV420: frame RGB va frame am ban.
    column_count = len(sample_frames)
    fig, axes = plt.subplots(2, column_count, figsize=(4 * column_count, 6))
    fig.suptitle(f"Video YUV420 -> RGB va bien doi am ban - {file_name}", fontsize=15)

    if column_count == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for column, (frame_index, rgb_frame, negative_frame) in enumerate(sample_frames):
        axes[0, column].imshow(rgb_frame)
        axes[0, column].set_title(f"RGB frame {frame_index}")
        axes[0, column].axis("off")

        axes[1, column].imshow(negative_frame)
        axes[1, column].set_title(f"255 - frame {frame_index}")
        axes[1, column].axis("off")

    fig.tight_layout()
    return fig


def main():
    # Khai bao cac tham so co the truyen khi chay chuong trinh.
    parser = argparse.ArgumentParser(description="Bai tap xu ly anh co ban.")
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Duong dan anh dau vao, mac dinh la anh trong thu muc hien tai.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mo cua so hien thi ket qua tren man hinh.",
    )
    parser.add_argument(
        "--yuv-file",
        default=DEFAULT_YUV_FILE,
        help="Duong dan video raw YUV420.",
    )
    parser.add_argument(
        "--yuv-width",
        type=int,
        default=DEFAULT_YUV_WIDTH,
        help="Chieu rong video YUV.",
    )
    parser.add_argument(
        "--yuv-height",
        type=int,
        default=DEFAULT_YUV_HEIGHT,
        help="Chieu cao video YUV.",
    )
    args = parser.parse_args()

    # Chuyen ten file thanh Path de lam viec voi duong dan de hon.
    image_path = Path(args.image)
    yuv_path = Path(args.yuv_file)

    # Tao thu muc ket_qua de luu toan bo anh/PDF dau ra.
    output_dir = Path("ket_qua")
    output_dir.mkdir(exist_ok=True)

    # Yeu cau 1: doc anh va hien thi theo cac he mau.
    views = build_color_views(image_path)
    fig = draw_result(views, image_path.name)

    # Phan bo sung: tach tung kenh R, G, B.
    channel_views = build_rgb_channel_views(image_path)
    channel_fig = draw_rgb_channels(channel_views, image_path.name)

    # Yeu cau 2: bien doi anh theo cong thuc 255 - pixel.
    negative_views = build_negative_image_views(image_path)
    negative_fig = draw_negative_image(negative_views, image_path.name)

    # Yeu cau 3: doc video YUV420, chuyen sang RGB va bien doi tung frame.
    frame_count, sample_frames = process_yuv420_video(
        yuv_path, args.yuv_width, args.yuv_height, output_dir
    )
    yuv_summary_fig = draw_yuv_video_summary(sample_frames, yuv_path.name)

    # Khai bao ten cac file ket qua se duoc xuat ra.
    png_path = output_dir / "ket_qua_cac_he_mau.png"
    pdf_path = output_dir / "ket_qua_cac_he_mau.pdf"
    channel_png_path = output_dir / "ket_qua_kenh_rgb.png"
    channel_pdf_path = output_dir / "ket_qua_kenh_rgb.pdf"
    negative_png_path = output_dir / "anh_goc_va_anh_bien_doi.png"
    negative_pdf_path = output_dir / "anh_goc_va_anh_bien_doi.pdf"
    yuv_summary_png_path = output_dir / "yuv420_rgb_va_bien_doi.png"
    yuv_summary_pdf_path = output_dir / "yuv420_rgb_va_bien_doi.pdf"

    # Luu cac hinh ket qua thanh PNG va PDF de dua vao bao cao.
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    channel_fig.savefig(channel_png_path, dpi=200)
    channel_fig.savefig(channel_pdf_path)
    negative_fig.savefig(negative_png_path, dpi=200)
    negative_fig.savefig(negative_pdf_path)
    yuv_summary_fig.savefig(yuv_summary_png_path, dpi=200)
    yuv_summary_fig.savefig(yuv_summary_pdf_path)

    # In thong bao de biet chuong trinh da tao nhung file nao.
    print(f"Da doc anh: {image_path}")
    print(f"Da luu anh ket qua: {png_path}")
    print(f"Da luu file PDF: {pdf_path}")
    print(f"Da luu anh tach kenh RGB: {channel_png_path}")
    print(f"Da luu file PDF tach kenh RGB: {channel_pdf_path}")
    print(f"Da luu anh bien doi 255 - pixel: {negative_png_path}")
    print(f"Da luu PDF bien doi 255 - pixel: {negative_pdf_path}")
    print(f"Da doc video YUV420: {yuv_path}")
    print(f"So frame da xu ly: {frame_count}")
    print(f"Frame RGB da luu trong: {output_dir / 'yuv_rgb_frames'}")
    print(f"Frame bien doi da luu trong: {output_dir / 'yuv_negative_frames'}")
    print(f"Da luu anh minh hoa video YUV: {yuv_summary_png_path}")
    print(f"Da luu PDF minh hoa video YUV: {yuv_summary_pdf_path}")

    # Neu co --show thi mo cua so hien thi, neu khong thi dong figure de tiet kiem bo nho.
    if args.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(channel_fig)
        plt.close(negative_fig)
        plt.close(yuv_summary_fig)


if __name__ == "__main__":
    main()
