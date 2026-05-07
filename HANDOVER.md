# 📄 Bàn Giao Dự Án: Web-Based Stock Quant Dashboard

Dự án này là một hệ thống phân tích kỹ thuật định lượng (Quantitative Analysis) cho thị trường chứng khoán Việt Nam, được chuyển đổi từ script Python local sang nền tảng Web để có thể truy cập từ thiết bị di động.

## 🚀 Tổng quan Công việc đã thực hiện

### 1. Cấu trúc Back-end (`app.py`)
- **Framework**: Flask với kiến trúc hướng đối tượng cho các module phân tích.
- **Module đã hoàn thiện**:
    - `TimeSeriesProcessor`: Tính toán Log Returns và làm mượt dữ liệu.
    - `StatisticalAnalyzer`: Tính toán Z-Score, P-Value, Rolling Skewness.
    - `MarketMicrostructure`: Phân tích Volume Profile (POC, VAH, VAL).
    - `CalculusLinearAlgebra`: Tính toán Vận tốc (Velocity) và Gia tốc (Acceleration) của giá.
    - `AnomalyDetector`: Thuật toán phát hiện "Real Nuke" (Sập thật), "Fake Nuke" (Bẫy gấu) và "Bullish Reversal" (Hồi phục).
    - `MachineLearningPredictor`: Sử dụng Random Forest để dự đoán xác suất tăng/giảm và giá trị kỳ vọng (EV).
- **Data Source**: Tích hợp thư viện `vnstock` để lấy dữ liệu thời gian thực từ KBS và VCI.

### 2. Giao diện Front-end (`templates/index.html`)
- **Aesthetic**: Thiết kế giao diện Dark Mode hiện đại (Premium UI), tương thích với Mobile.
- **Visualization**: Sử dụng `Plotly.js` để vẽ biểu đồ đa lớp:
    - Biểu đồ nến/line kèm vùng Value Area.
    - Biểu đồ Volume Profile nằm ngang.
    - Biểu đồ phân phối chuẩn (Bell Curve) cho Z-Score.
    - Timeline Z-Score để theo dõi ngưỡng quá mua/quá bán.
- **Interactivity**: Chế độ chuyển đổi nhanh giữa STOCK và VN-INDEX, tùy chỉnh khung thời gian (1D, 1H, 15m).

### 3. Cấu hình hệ thống
- `requirements.txt`: Đã liệt kê đầy đủ các thư viện cần thiết.
- `Procfile` & `runtime.txt`: Sẵn sàng để deploy lên Heroku, Render hoặc Railway.

---

## 🛠 Hướng dẫn cho VS Code (Tiếp tục thực hiện)

### 1. Cách chạy dự án hiện tại
```bash
# Cài đặt thư viện
pip install -r requirements.txt

# Chạy server
python app.py
```
Truy cập: `http://localhost:5000`

### 2. Các Task cần thực hiện tiếp theo (Next Steps)
- [ ] **Deployment**: Đưa mã nguồn lên GitHub và kết nối với Render.com hoặc Railway.app để có URL truy cập từ điện thoại.
- [ ] **Authentication**: Thêm màn hình Login đơn giản hoặc xác thực Token để bảo mật dữ liệu cá nhân.
- [ ] **Backtest Integration**: Tích hợp logic từ `vnindex_backtest.py` vào web để xem hiệu suất chiến thuật ngay trên UI.
- [ ] **Multi-Ticker Comparison**: Cho phép so sánh tương quan giữa mã cổ phiếu và VN-INDEX trên cùng một biểu đồ.
- [ ] **Optimized Loading**: Lưu trữ (Cache) dữ liệu `vnstock` để tránh bị rate limit khi refresh liên tục.

### 3. Ghi chú về Logic quan trọng
- **Fake Nuke**: Xảy ra khi Z-Score cực thấp nhưng Gia tốc bắt đầu cải thiện và Skewness chuyển sang dương (hồi kỹ thuật).
- **Real Nuke**: Xảy ra khi Z-Score cực thấp kèm theo Skewness âm sâu và không có dấu hiệu cải thiện gia tốc.
- **ML Probability**: Dự đoán dựa trên các tính năng định lượng (Z-score, Accel, Volume) thay vì chỉ nhìn nến.

---
*File này được tạo để hỗ trợ AI hoặc lập trình viên tiếp quản dự án một cách nhanh nhất.*
