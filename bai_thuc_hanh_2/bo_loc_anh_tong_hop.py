"""
Loc nhieu cho noisy_image.jpg va noisy_image2.jpg.
Gom 3 bo loc: trung binh, trung vi va Gaussian.
Khong su dung OpenCV hoac SciPy.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Cho phep luu bieu do ma khong can mo cua so.
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view


# Cac anh dau vao va kich thuoc kernel can thu nghiem.
THU_MUC = Path(__file__).parent
CAC_ANH = ["noisy_image.jpg", "noisy_image2.jpg"]
CAC_KERNEL = (3, 5, 7)


def tao_cua_so(anh, kich_thuoc):
    """Tao mot cua so kernel quanh tung pixel cua anh."""
    le = kich_thuoc // 2

    # Them vien bang cach phan chieu pixel de ket qua khong bi thu nho.
    anh_them_vien = np.pad(anh, le, mode="reflect")
    return sliding_window_view(anh_them_vien, (kich_thuoc, kich_thuoc))


def loc_trung_binh(anh, kich_thuoc):
    """Thay pixel tam bang trung binh cac pixel trong kernel."""
    cac_cua_so = tao_cua_so(anh, kich_thuoc)
    ket_qua = np.mean(cac_cua_so, axis=(-2, -1))
    return np.rint(ket_qua).astype(np.uint8)


def loc_trung_vi(anh, kich_thuoc):
    """Thay pixel tam bang gia tri trung vi trong kernel."""
    cac_cua_so = tao_cua_so(anh, kich_thuoc)
    ket_qua = np.median(cac_cua_so, axis=(-2, -1))
    return ket_qua.astype(np.uint8)


def loc_gaussian(anh, kich_thuoc):
    """Nhan cac pixel voi trong so Gaussian; pixel gan tam co trong so lon hon."""
    # Sigma tu dong: kernel 3, 5, 7 tuong ung sigma 0.8, 1.1, 1.4.
    sigma = 0.3 * ((kich_thuoc - 1) / 2 - 1) + 0.8

    # Tao ma tran trong so Gaussian va chuan hoa de tong bang 1.
    toa_do = np.arange(-kich_thuoc // 2 + 1, kich_thuoc // 2 + 1)
    x, y = np.meshgrid(toa_do, toa_do)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    # Nhan tung cua so anh voi kernel roi cong cac gia tri lai.
    cac_cua_so = tao_cua_so(anh, kich_thuoc)
    ket_qua = np.einsum("ijkl,kl->ij", cac_cua_so, kernel)
    return np.rint(ket_qua).astype(np.uint8)


# Gan ten bo loc voi ham xu ly tuong ung.
CAC_BO_LOC = {
    "trung_binh": loc_trung_binh,
    "trung_vi": loc_trung_vi,
    "gaussian": loc_gaussian,
}


def xu_ly_anh(ten_anh):
    """Ap dung ca 3 bo loc cho mot anh va luu ket qua."""
    duong_dan = THU_MUC / ten_anh
    if not duong_dan.exists():
        print(f"Khong tim thay: {duong_dan}")
        return

    # Doc anh bang Pillow va chuyen thanh anh xam dang mang NumPy.
    with Image.open(duong_dan) as file_anh:
        anh_goc = np.array(file_anh.convert("L"))

    for ten_bo_loc, ham_loc in CAC_BO_LOC.items():
        thu_muc_ket_qua = THU_MUC / "ket_qua_bo_loc" / ten_bo_loc
        thu_muc_ket_qua.mkdir(parents=True, exist_ok=True)

        # Loc anh lan luot bang kernel 3x3, 5x5 va 7x7.
        cac_ket_qua = [ham_loc(anh_goc, k) for k in CAC_KERNEL]

        # Luu tung anh ket qua rieng.
        for k, anh_da_loc in zip(CAC_KERNEL, cac_ket_qua):
            ten_file = f"{duong_dan.stem}_{ten_bo_loc}_{k}x{k}.png"
            Image.fromarray(anh_da_loc).save(thu_muc_ket_qua / ten_file)

        # Tao them mot anh gom de so sanh truc quan.
        fig, cac_o = plt.subplots(1, 4, figsize=(16, 4))
        tieu_de = ["Anh nhieu ban dau", "Kernel 3x3", "Kernel 5x5", "Kernel 7x7"]
        for o, ten, anh in zip(cac_o, tieu_de, [anh_goc, *cac_ket_qua]):
            o.imshow(anh, cmap="gray", vmin=0, vmax=255)
            o.set_title(ten)
            o.axis("off")

        fig.suptitle(f"{ten_bo_loc.replace('_', ' ').title()} - {ten_anh}")
        fig.tight_layout()
        ten_so_sanh = f"{duong_dan.stem}_{ten_bo_loc}_so_sanh.png"
        fig.savefig(thu_muc_ket_qua / ten_so_sanh, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    for ten_anh in CAC_ANH:
        xu_ly_anh(ten_anh)
    print("Da loc xong. Ket qua nam trong thu muc ket_qua_bo_loc.")
