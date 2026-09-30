# Bài thực hành 2 – Các bộ lọc ảnh

Repository chứa mã nguồn **Bài thực hành số 2** của môn Xử lý ảnh – Thị giác
Robot. Nội dung chính là tự xây dựng các bộ lọc làm trơn, khử nhiễu và phát hiện
biên, sau đó đối chiếu kết quả với OpenCV.

## Nội dung bài thực hành

- Bộ lọc trung bình (Mean).
- Bộ lọc trung vị (Median).
- Bộ lọc Gaussian.
- Bộ lọc Sobel.
- Bộ lọc Canny.
- Thử nghiệm kernel `3×3`, `5×5`, `7×7`.
- So sánh kết quả tự xây dựng với OpenCV bằng MAE, tỷ lệ pixel giống nhau,
  Precision, Recall, IoU và Dice.

Các thuật toán được cài đặt bằng NumPy. OpenCV không được dùng thay cho các bộ
lọc tự xây dựng; thư viện chỉ được dùng để hiển thị, lưu ảnh hoặc tạo kết quả đối
chiếu khi đề bài yêu cầu.

## Cấu trúc repository

```text
filters_images/
├── bai_thuc_hanh_2/
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
- NumPy.
- Pillow.
- Matplotlib.
- OpenCV Python.

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

Kích hoạt môi trường ảo trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Trên Linux hoặc macOS:

```bash
source .venv/bin/activate
```

Cài đặt thư viện:

```bash
python -m pip install -r requirements.txt
```

Di chuyển vào thư mục bài thực hành:

```bash
cd bai_thuc_hanh_2
```

## 1. Chạy Mean, Median và Gaussian

### Chạy bằng một file tổng hợp

```bash
python bo_loc_anh_tong_hop.py
```

Chương trình xử lý `noisy_image.jpg` và `noisy_image2.jpg` bằng ba bộ lọc với các
kernel:

- `3×3`: giảm nhiễu nhẹ, giữ nhiều chi tiết.
- `5×5`: cân bằng giữa giảm nhiễu và bảo toàn chi tiết.
- `7×7`: giảm nhiễu mạnh nhưng làm ảnh mờ hơn.

Kết quả được lưu trong:

```text
ket_qua_bo_loc/
```

### Chạy phiên bản chia thành nhiều file

Chạy cả ba bộ lọc:

```bash
python chay_tat_ca_bo_loc.py
```

Hoặc chạy riêng từng bộ lọc:

```bash
python trungbinh_Filter.py
python trungvi_Filter.py
python gau_Filter.py
```

Các file trên sử dụng chung các hàm trong `bo_loc_utils.py`.

## 2. Chạy Sobel và Canny trên ma trận

```bash
python sobel_canny.py
```

Chương trình tự thực hiện các bước:

1. Thêm biên 0 cho ma trận.
2. Tính Sobel theo hai hướng `Gx`, `Gy`.
3. Tính độ lớn và hướng gradient.
4. Làm mờ Gaussian cho Canny.
5. Non-maximum suppression.
6. Phân ngưỡng kép.
7. Hysteresis để nối cạnh yếu với cạnh mạnh.

Nhấn phím bất kỳ trên cửa sổ kết quả để thoát. Kết quả được lưu trong:

```text
output/
```

## 3. So sánh với OpenCV

```bash
python so_sanh_tu_xay_dung_opencv.py
```

Chương trình đối chiếu Mean, Median, Gaussian, Sobel và Canny tự xây dựng với các
hàm tương ứng của OpenCV.

Các chỉ số được sử dụng:

- **MAE:** sai số tuyệt đối trung bình.
- **Sai khác lớn nhất:** chênh lệch pixel lớn nhất giữa hai kết quả.
- **Tỷ lệ pixel giống nhau:** phần trăm pixel có cùng giá trị.
- **Precision, Recall, IoU, Dice:** đánh giá riêng bản đồ cạnh Canny.

Ảnh so sánh và bảng CSV được lưu trong:

```text
ket_qua_so_sanh_opencv/
```

## Nhận xét ngắn

- Mean đơn giản nhưng làm mờ cạnh và chi tiết.
- Median phù hợp nhất với nhiễu muối–tiêu.
- Gaussian phù hợp với nhiễu Gaussian và làm mượt tự nhiên hơn Mean.
- Sobel tính được độ lớn và hướng gradient nhưng tạo cạnh tương đối dày.
- Canny tạo cạnh mảnh và liên tục hơn nhờ Gaussian, NMS, ngưỡng kép và
  hysteresis.
- Kernel càng lớn thì khả năng giảm nhiễu càng mạnh, nhưng ảnh càng mất chi tiết.
