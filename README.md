# Thực hành Xử lý ảnh – Bài 1 và Bài 2

Repository chứa mã nguồn hai bài thực hành môn **Xử lý ảnh – Thị giác Robot**.

- **Bài 1:** đọc ảnh, chuyển đổi hệ màu, tạo ảnh âm bản và xử lý video raw YUV420.
- **Bài 2:** tự xây dựng các bộ lọc Mean, Median, Gaussian, Sobel, Canny và so sánh với OpenCV.

Các thuật toán lọc ở Bài 2 được cài đặt bằng NumPy. OpenCV không được dùng thay
cho thuật toán tự xây dựng; thư viện chỉ được dùng để đọc/ghi, hiển thị hoặc tạo
kết quả đối chiếu khi đề bài yêu cầu.

## Cấu trúc repository

```text
filters_images/
├── bai_1/
│   ├── bai1_doc_anh_he_mau.py
│   ├── bai2_bien_doi_anh.py
│   ├── bai3_video_yuv420.py
│   ├── xu_ly_anh_co_ban.py
│   ├── xem_file_yuv.py
│   ├── RaceHorses_416x240_frame001.png
│   └── RaceHorses_416x240_20.yuv
├── bai_2/
│   ├── bo_loc_anh_tong_hop.py
│   ├── bo_loc_utils.py
│   ├── chay_tat_ca_bo_loc.py
│   ├── trungbinh_Filter.py
│   ├── trungvi_Filter.py
│   ├── gau_Filter.py
│   ├── sobel_canny.py
│   ├── so_sanh_tu_xay_dung_opencv.py
│   ├── noisy_image.jpg
│   └── noisy_image2.jpg
├── requirements.txt
└── README.md
```

## Yêu cầu môi trường

- Python 3.10 trở lên.
- Các thư viện trong `requirements.txt`.

## Cài đặt

Clone repository:

```bash
git clone https://github.com/23020734-thiendac/filters_images.git
cd filters_images
```

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Trên Linux hoặc macOS:

```bash
source .venv/bin/activate
```

Cài thư viện:

```bash
python -m pip install -r requirements.txt
```

## Hướng dẫn chạy Bài 1

Di chuyển vào thư mục Bài 1:

```bash
cd bai_1
```

### 1. Đọc ảnh và hiển thị các hệ màu

```bash
python bai1_doc_anh_he_mau.py
```

Chương trình đọc ảnh đầu vào và biểu diễn ở các hệ màu RGB, BGR, YUV, HSV và
grayscale.

### 2. Biến đổi âm bản

```bash
python bai2_bien_doi_anh.py
```

Mỗi pixel được biến đổi theo công thức:

```text
pixel_moi = 255 - pixel_cu
```

### 3. Đọc video raw YUV420

```bash
python bai3_video_yuv420.py
```

Chương trình đọc toàn bộ 20 frame của file `RaceHorses_416x240_20.yuv`, chuyển
từ YUV420 sang RGB và tạo ảnh âm bản cho từng frame.

Có thể đọc riêng một frame:

```bash
python xem_file_yuv.py --frame 0
```

### 4. Chạy chương trình tổng hợp Bài 1

```bash
python xu_ly_anh_co_ban.py
```

Muốn mở cửa sổ xem kết quả:

```bash
python xu_ly_anh_co_ban.py --show
```

Kết quả Bài 1 được tạo trong thư mục `bai_1/ket_qua/`.

## Hướng dẫn chạy Bài 2

Từ thư mục gốc của repository:

```bash
cd bai_2
```

### 1. Chạy Mean, Median và Gaussian

Chạy tất cả bằng một file độc lập:

```bash
python bo_loc_anh_tong_hop.py
```

Chương trình xử lý `noisy_image.jpg` và `noisy_image2.jpg` với các kernel:

- `3×3`: giảm nhiễu nhẹ, giữ nhiều chi tiết.
- `5×5`: cân bằng giữa giảm nhiễu và giữ chi tiết.
- `7×7`: giảm nhiễu mạnh nhưng làm ảnh mờ hơn.

Kết quả được lưu tại:

```text
bai_2/ket_qua_bo_loc/
```

Ngoài ra có thể chạy phiên bản chia thành nhiều file:

```bash
python chay_tat_ca_bo_loc.py
```

Hoặc chạy riêng từng bộ lọc:

```bash
python trungbinh_Filter.py
python trungvi_Filter.py
python gau_Filter.py
```

### 2. Chạy Sobel và Canny trên ma trận

```bash
python sobel_canny.py
```

Chương trình thực hiện:

1. Sobel theo hai hướng `Gx`, `Gy`.
2. Tính độ lớn và hướng gradient.
3. Canny gồm Gaussian, Sobel, non-maximum suppression, ngưỡng kép và hysteresis.

Nhấn phím bất kỳ trên cửa sổ kết quả để kết thúc chương trình. Ảnh được lưu vào:

```text
bai_2/output/
```

### 3. So sánh bản tự xây dựng với OpenCV

```bash
python so_sanh_tu_xay_dung_opencv.py
```

Chương trình so sánh Mean, Median, Gaussian, Sobel và Canny bằng các chỉ số:

- MAE – sai số tuyệt đối trung bình.
- Sai khác pixel lớn nhất.
- Tỷ lệ pixel giống nhau.
- Precision, Recall, IoU và Dice đối với bản đồ cạnh Canny.

Ảnh và bảng CSV được lưu vào:

```text
bai_2/ket_qua_so_sanh_opencv/
```

## Ghi chú

- Phải chạy lệnh từ đúng thư mục `bai_1` hoặc `bai_2` để chương trình tìm thấy
  các file đầu vào theo đường dẫn tương đối.
- Các thư mục kết quả được tạo tự động và không được lưu trên Git để repository
  gọn nhẹ.
- `sobel_canny.py` mở cửa sổ đồ họa, vì vậy cần môi trường desktop để quan sát.
- Nếu dùng máy chủ không có giao diện, có thể bỏ phần `cv2.imshow()` và chỉ lưu
  ảnh kết quả.

## Tác giả

- Mã sinh viên: **23020734**
- GitHub: [23020734-thiendac](https://github.com/23020734-thiendac)
