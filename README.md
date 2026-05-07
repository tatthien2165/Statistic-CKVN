# OpenClaw - Web-Based Stock Quant Dashboard

Ứng dụng này là một web app Flask cho phân tích kỹ thuật định lượng thị trường chứng khoán Việt Nam.

## Chức năng chính
- Tải dữ liệu từ `vnstock` cho cổ phiếu Viet Nam và VN-Index.
- Tính toán Log Returns, Z-Score, Skewness, Volume Profile, Velocity/Acceleration.
- Phát hiện tín hiệu `Fake Nuke`, `Real Nuke`, `Bullish Reversal`.
- Dự đoán xác suất tăng/giảm và kỳ vọng EV bằng Random Forest.
- Giao diện dark mode responsive, tích hợp Plotly chart.

## Chạy local
```bash
pip install -r requirements.txt
python app.py
```
Mở `http://localhost:5000` trong trình duyệt.

## Triển khai lên Web
Ứng dụng đã sẵn sàng cho deployment trên Render, Railway hoặc Heroku.

### Trên Render
1. Tạo repository trên GitHub và đẩy mã nguồn lên.
2. Đăng nhập Render và tạo Web Service mới.
3. Chọn GitHub repository.
4. Command build: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`
6. Render sẽ cung cấp URL truy cập.

## Làm web trên GitHub
Hiện tại, repo chưa có remote GitHub. Để hoàn tất:

```bash
git init
git add .
git commit -m "Initial commit: Flask stock quant dashboard"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Sau đó kết nối repository với Render hoặc Railway để có website trực tuyến.

## Lưu ý
- Dữ liệu thực tế phụ thuộc vào API `vnstock`.
- Nếu muốn bảo mật, hãy thêm `.env` và cấu hình token riêng.
