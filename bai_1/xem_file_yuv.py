import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_YUV_FILE = "RaceHorses_416x240_20.yuv"
DEFAULT_WIDTH = 416
DEFAULT_HEIGHT = 240


def read_yuv420_frame(file_path: Path, width: int, height: int, frame_index: int):
    frame_size = width * height * 3 // 2
    offset = frame_index * frame_size

    with file_path.open("rb") as file:
        file.seek(offset)
        frame_data = file.read(frame_size)

    if len(frame_data) != frame_size:
        raise ValueError(
            f"Khong doc duoc frame {frame_index}. "
            f"File chi co {len(frame_data)} bytes tai vi tri nay."
        )

    yuv_frame = np.frombuffer(frame_data, dtype=np.uint8)
    yuv_frame = yuv_frame.reshape((height * 3 // 2, width))
    rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB_I420)
    return rgb_frame


def main():
    parser = argparse.ArgumentParser(description="Mo va xem file raw YUV420.")
    parser.add_argument("--file", default=DEFAULT_YUV_FILE, help="Duong dan file .yuv")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Chieu rong anh")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Chieu cao anh")
    parser.add_argument("--frame", type=int, default=0, help="Frame can xem, tinh tu 0")
    parser.add_argument("--show", action="store_true", help="Mo cua so hien thi anh")
    args = parser.parse_args()

    file_path = Path(args.file)
    rgb_frame = read_yuv420_frame(file_path, args.width, args.height, args.frame)

    output_dir = Path("ket_qua")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"yuv_frame_{args.frame:03d}.png"

    plt.imsave(output_path, rgb_frame)
    print(f"Da doc file YUV: {file_path}")
    print(f"Da luu frame {args.frame}: {output_path}")

    if args.show:
        plt.imshow(rgb_frame)
        plt.title(f"YUV420 frame {args.frame}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
