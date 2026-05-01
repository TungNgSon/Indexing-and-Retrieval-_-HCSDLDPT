README — HCSDLDPT demo

Mục đích
- Hướng dẫn nhanh để người khác chạy demo: tạo môi trường ảo, cài phụ thuộc, tạo index (hoặc tải artifacts), và chạy Streamlit demo.

Yêu cầu
- Python 3.11
- Git (khuyến nghị)
- Quyền mạng (tải model rembg nếu cần)

1) Tạo và kích hoạt môi trường ảo (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

(Trên CMD: `.venv\Scripts\activate.bat`; trên macOS/Linux: `source .venv/bin/activate`)

2) Cài thư viện

```powershell
pip install -r requirements.txt
```

Nếu không có `requirements.txt`, cài tay các gói chính: `opencv-python`, `scikit-learn`, `scikit-image`, `joblib`, `streamlit`, `chromadb`, `rembg`.
<!-- 
DUNG LAM -->
<!-- 3) (Tùy chọn) Tải artifacts đã tính sẵn

- Repo mặc định loại trừ thư mục `artifacts/`. Nếu bạn muốn tránh build tốn thời gian, hãy tải tập `artifacts/video_feature_store.npz` và `artifacts/models/*.joblib` từ nguồn được chia sẻ (GitHub Release / S3 / Google Drive) và đặt chúng vào thư mục `artifacts/` tương ứng.

Ví dụ PowerShell (thay URL bằng link thực tế):

```powershell
New-Item -ItemType Directory -Path artifacts\models -Force
Invoke-WebRequest -Uri "https://your-host/feature_store/video_feature_store.npz" -OutFile artifacts\video_feature_store.npz
Invoke-WebRequest -Uri "https://your-host/models/feature_scalers.joblib" -OutFile artifacts\models\feature_scalers.joblib
Invoke-WebRequest -Uri "https://your-host/models/hog_pca.joblib" -OutFile artifacts\models\hog_pca.joblib
``` -->

4) Nếu không có artifacts: build index từ đầu

```powershell
python scripts\build_chroma_index.py
```

Ghi chú: quá trình này có thể lâu — đặc biệt nếu phải tải model `rembg` (U2Net) và trích xuất HOG cho nhiều video. Bạn có thể chạy trên Colab và upload artifacts nếu cần.

5) Chạy demo Streamlit

```powershell
streamlit run app.py
```

6) Những file nên push lên GitHub

- `app.py`
- `src/` (tất cả `.py`)
- `scripts/` (trình build)
- `requirements.txt`
- `daphuongtien7_multi_rframe_pipeline.ipynb` (notebook tham khảo)
- `README.md`
- `.gitignore`

Không push: `data/`, `artifacts/`, lớn file `.npz`/`.joblib` (nên đưa vào Release hoặc lưu tách biệt).

7) Tùy chọn: tự động tải artifacts

Nếu muốn, tôi có thể thêm `scripts/download_artifacts.py` hoặc một PowerShell script để tải tự động từ URL bạn cung cấp.

Liên hệ / Ghi chú
- Nếu cần, hãy cho tôi URL nơi bạn muốn lưu artifacts (S3/Drive/GitHub Release), tôi tạo sẵn `download` script và cập nhật `README.md` với link.
