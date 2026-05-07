from vnstock import Vnstock
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
import sys

# Đảm bảo in Tiếng Việt không lỗi trên Terminal Windows
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# CẤU HÌNH & LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('QuantModel')
sns.set_theme(style="darkgrid")

# ==========================================
# 1. DATA TRANSFORMATION & TIME SERIES
# ==========================================
class TimeSeriesProcessor:
    """Xử lý Tiền xử lý dữ liệu chuẩn Time Series."""
    @staticmethod
    def calculate_log_returns(df, price_col='Close'):
        """Log Returns: R_t = ln(P_t / P_{t-1}) - Chuẩn hóa biến động, giúp tiệm cận Bell Curve."""
        df = df.copy()
        df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))
        return df

    @staticmethod
    def smooth_data(df, col='Close', window=5):
        """Làm mượt dữ liệu (Smoothing) để lọc nhiễu (Noise)."""
        df = df.copy()
        df[f'{col}_Smoothed'] = df[col].rolling(window=window).mean()
        return df

# ==========================================
# 2. STATISTICAL FOUNDATIONS
# ==========================================
class StatisticalAnalyzer:
    """Nền tảng Thống kê & Giá trị tương đối."""
    def __init__(self, window=20):
        self.window = window

    def calculate_rolling_stats(self, df, return_col='Log_Return'):
        df = df.copy()
        # Stdev & Bell Curve properties
        df['Rolling_Mean'] = df[return_col].rolling(window=self.window).mean()
        df['Rolling_Std'] = df[return_col].rolling(window=self.window).std()
        
        # Z-Score: Z = (X - \mu) / \sigma
        df['Z_Score'] = (df[return_col] - df['Rolling_Mean']) / df['Rolling_Std']
        
        # P-Value (Cumulative Distribution Function - xác suất của 1 biến cố giảm mành)
        df['P_Value'] = stats.norm.cdf(df['Z_Score'])
        
        # Rolling Skewness (Độ xiên) - Tìm kiếm phân phối lệch trái/phải
        df['Rolling_Skew'] = df[return_col].rolling(window=self.window).skew()
        
        # Skewness ngắn hạn để bắt đáy/đảo chiều (Shakeout)
        short_term = max(3, self.window // 4)
        df['Skewness_ShortTerm'] = df[return_col].rolling(window=short_term).skew()

        # Relative Value: So sánh Z-Score hiện tại với TB Lịch sử (Lookback lớn)
        df['Z_Score_Avg_Hist'] = df['Z_Score'].rolling(window=self.window * 3).mean()
        df['Relative_Z_Score'] = df['Z_Score'] - df['Z_Score_Avg_Hist']
        
        return df

    @staticmethod
    def plot_bell_curve(ax, returns, title='Fat Tails Real Life Return vs Normal Curve'):
        """Vẽ biểu đồ phân phối chuẩn (Bell Curve) đè lên biểu đồ thực tế."""
        returns = returns.dropna()
        mu, std = stats.norm.fit(returns)
        
        sns.histplot(returns, bins=50, stat='density', alpha=0.6, color='steelblue', label='Actual Returns', ax=ax)
        
        # Generate Normal Curve
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        
        ax.plot(x, p, 'k', linewidth=2, label=rf'Normal Curve ($\mu$={mu:.4f}, $\sigma$={std:.4f})')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend()

# ==========================================
# 3. MARKET MICROSTRUCTURE (VOLUME PROFILE)
# ==========================================
class MarketMicrostructure:
    """Cấu trúc vi mô thị trường thông qua Khối lượng."""
    def __init__(self, bins=50, val_area_pct=0.70):
        self.bins = bins
        self.val_area_pct = val_area_pct

    def calc_volume_profile(self, df, price_col='Close', vol_col='Volume'):
        """Tính toán Point of Control (POC), Value Area High (VAH) & Low (VAL)."""
        min_price = df[price_col].min()
        max_price = df[price_col].max()
        price_bins = np.linspace(min_price, max_price, self.bins)
        
        df = df.copy()
        # Gom nhóm mức giá (Price Bins/TPO)
        df['Price_Bin'] = np.digitize(df[price_col], price_bins)
        
        vp = df.groupby('Price_Bin')[vol_col].sum().reset_index()
        vp['Price'] = price_bins[vp['Price_Bin'] - 1]
        vp = vp.sort_values('Price').reset_index(drop=True)
        
        total_volume = vp[vol_col].sum()
        
        # Point of Control (POC)
        poc_idx = vp[vol_col].idxmax()
        poc_price = vp.loc[poc_idx, 'Price']
        
        # Value Area (70% Volume tập trung)
        target_vol = total_volume * self.val_area_pct
        current_vol = vp.loc[poc_idx, vol_col]
        upper_idx, lower_idx = poc_idx, poc_idx
        
        while current_vol < target_vol:
            up_vol = vp.loc[upper_idx + 1, vol_col] if upper_idx + 1 < len(vp) else -1
            down_vol = vp.loc[lower_idx - 1, vol_col] if lower_idx - 1 >= 0 else -1
            
            if up_vol == -1 and down_vol == -1:
                break
                
            if up_vol > down_vol:
                upper_idx += 1
                current_vol += up_vol
            else:
                lower_idx -= 1
                current_vol += down_vol
                
        vah = vp.loc[upper_idx, 'Price']
        val = vp.loc[lower_idx, 'Price']
        
        return vp, poc_price, vah, val

    @staticmethod
    def plot_volume_profile(ax, vp, poc_price, vah, val, title='Volume Profile \n(Liquidity Nodes)'):
        ax.barh(vp['Price'], vp['Volume'], height=(vp['Price'].max() - vp['Price'].min())/len(vp), color='cadetblue', edgecolor='white', alpha=0.7)
        ax.axhline(poc_price, color='red', linestyle='--', linewidth=2, label=f'POC: {poc_price:.2f}')
        ax.axhline(vah, color='forestgreen', linestyle='-', linewidth=1.5, label=f'VAH: {vah:.2f}')
        ax.axhline(val, color='forestgreen', linestyle='-', linewidth=1.5, label=f'VAL: {val:.2f}')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Volume')
        ax.set_ylabel('Price')
        ax.legend()

# ==========================================
# 4. CALCULUS & LINEAR ALGEBRA
# ==========================================
class CalculusLinearAlgebra:
    """Giải tích & Đại số tuyến tính trong Chuỗi thời gian."""
    @staticmethod
    def calculate_kinematics(df, col='Close_Smoothed'):
        df = df.copy()
        # Đạo hàm bậc 1: Vận tốc (Động lượng giá)
        df['Velocity'] = np.gradient(df[col].bfill().ffill())
        # Đạo hàm bậc 2: Gia tốc (Sự thay đổi của Động lượng)
        df['Acceleration'] = np.gradient(df['Velocity'])
        return df
        
    @staticmethod
    def build_feature_matrix(df, cols):
        """Đại số tuyến tính: Biến đổi Vector và tính Feature/Correlation Matrix."""
        df_clean = df[cols].dropna()
        feature_matrix = df_clean.values
        # Tính ma trận tương quan giữa Vector Giá, Vol, Động lượng
        corr_matrix = np.corrcoef(feature_matrix, rowvar=False)
        return feature_matrix, corr_matrix

# ==========================================
# 5. ANOMALY DETECTION (SIGNAL RULE)
# ==========================================
class AnomalyDetector:
    """Phân biệt Real Nuke (Phân phối), Fake Nuke (Rũ hàng) và Bullish Reversal (Đảo chiều)."""
    def __init__(self, z_thresh=-1.5, p_thresh=0.06):
        self.z_thresh = z_thresh
        self.p_thresh = p_thresh
        
    def detect(self, df, vp_val, vp_poc, vp_vah):
        df = df.copy()
        df['Signal'] = 'Normal'
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            if pd.isna(curr['Z_Score']) or pd.isna(curr['Skewness_ShortTerm']) or pd.isna(curr['Acceleration']):
                continue
                
            # 1. NHẬN DIỆN NUKE (GIẢM)
            is_nuke = (curr['Z_Score'] <= self.z_thresh) and (curr['P_Value'] <= self.p_thresh)
            
            if is_nuke:
                accel_improving = curr['Acceleration'] > prev['Acceleration']
                price_below_val = curr['Close'] < vp_val
                skew_positive = curr['Skewness_ShortTerm'] > 0
                skew_deep_neg = curr['Rolling_Skew'] < -1.0
                
                if accel_improving and price_below_val and skew_positive:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'Fake Nuke (Shakeout/Bear trap)'
                elif skew_deep_neg or not accel_improving:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'Real Nuke (Crash/Distribution)'
                continue

            # 2. NHẬN DIỆN ĐẢO CHIỀU TĂNG (BULLISH REVERSAL)
            recovering_z = (curr['Z_Score'] > -1.0) and (prev['Z_Score'] < -1.5)
            strong_accel = (curr['Acceleration'] > 0) and (curr['Acceleration'] > prev['Acceleration'])
            price_reclaiming_val = (curr['Close'] > vp_val) and (prev['Close'] <= vp_val)
            price_reclaiming_poc = (curr['Close'] > vp_poc) and (prev['Close'] <= vp_poc)
            
            if (recovering_z and strong_accel) or (price_reclaiming_val and strong_accel) or (price_reclaiming_poc and curr['Z_Score'] > 0):
                if curr['Log_Return'] > 0:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'Bullish Reversal (Trend Change)'
                
        return df

# ==========================================
# 6. MACHINE LEARNING PREDICTION
# ==========================================
class MachineLearningPredictor:
    """Mô hình Học máy dự báo xác suất Tăng giá (T+1) từ Correlation Matrix Features."""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
    def prepare_data(self, df):
        features = ['Log_Return', 'Velocity', 'Acceleration', 'Z_Score', 'Rolling_Skew', 'Volume']
        df_ml = df.dropna(subset=features).copy()
        
        # Nhãn (Label): 1 nếu T+1 tăng, 0 nếu giảm
        df_ml['Target'] = (df_ml['Close'].shift(-1) > df_ml['Close']).astype(int)
        
        # Bỏ đi dòng cuối do T+1 = NaN
        return df_ml.dropna(subset=['Target']), features
        
    def train_and_predict(self, df):
        df_ml, features = self.prepare_data(df)
        if len(df_ml) < 50:
            logger.warning("Không đủ dữ liệu cho Machine Learning.")
            return None, None
            
        # Chia train/test (80/20) - Walk-forward đơn giản
        split_idx = int(len(df_ml) * 0.8)
        train = df_ml.iloc[:split_idx]
        test = df_ml.iloc[split_idx:]
        
        X_train, y_train = train[features], train['Target']
        X_test, y_test = test[features], test['Target']
        
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Dự báo phân phối xác suất cho nến mới nhất df.iloc[-1]
        latest_features = df.iloc[-1:][features]
        if not latest_features.isna().any().any():
            prob_up = self.model.predict_proba(latest_features)[0][1]
        else:
            prob_up = 0.5
            
        return acc, prob_up

# ==========================================
# ORCHESTRATION ENGINE
# ==========================================
class QuantModelEngine:
    # Mapping interval -> adaptive rolling window
    INTERVAL_WINDOWS = {
        '1D': {'stats': 20, 'smooth': 5, 'vp_lookback': 60},
        '15m': {'stats': 30, 'smooth': 8, 'vp_lookback': 120},
        '1m':  {'stats': 60, 'smooth': 15, 'vp_lookback': 240},
    }

    def __init__(self, ticker='^VNINDEX', start_date='2018-01-01', end_date=None, interval='1D'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or pd.Timestamp.today().strftime('%Y-%m-%d')
        self.interval = interval
        
        params = self.INTERVAL_WINDOWS.get(interval, self.INTERVAL_WINDOWS['1D'])
        
        self.ts_proc = TimeSeriesProcessor()
        self.stats_analyzer = StatisticalAnalyzer(window=params['stats'])
        self.vp_analyzer = MarketMicrostructure(bins=60, val_area_pct=0.70)
        self.calc_alg = CalculusLinearAlgebra()
        self.detector = AnomalyDetector()
        self.ml_predictor = MachineLearningPredictor()
        self.vp_lookback = params['vp_lookback']
        self.smooth_window = params['smooth']

    def fetch_data(self):
        logger.info(f"Đang tải dữ liệu {self.ticker} | Interval: {self.interval} | {self.start_date} -> {self.end_date}")
        try:
            stock = Vnstock().stock(symbol="VNINDEX", source="VCI")
            df = stock.quote.history(start=self.start_date, end=self.end_date, interval=self.interval)
            
            if df is None or df.empty:
                raise ValueError("Không tìm thấy dữ liệu.")
                
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df.columns = df.columns.str.capitalize()
            
            # 1. Loại bỏ các nến NaN (giờ nghỉ trưa hoặc lỗi dữ liệu)
            df = df.dropna(subset=['Close'])
            
            # 2. Loại bỏ Thứ 7 (5) và Chủ Nhật (6) - Đảm bảo chỉ có ngày làm việc
            df = df[df.index.dayofweek < 5]
            
            # 3. Loại bỏ các ngày nghỉ lễ/không giao dịch (Volume = 0 hoặc cực thấp)
            # Thông thường VN-Index có Volume tối thiểu hàng chục triệu cổ phiếu
            df = df[df['Volume'] > 0]
            
            # 4. Loại bỏ các dữ liệu trùng lặp timestamp (nếu có)
            df = df[~df.index.duplicated(keep='first')]
            
            # 5. Sắp xếp lại theo thời gian để đảm bảo chuỗi chuẩn
            df = df.sort_index()
            
            logger.info(f"Tải và làm sạch dữ liệu thành công: {len(df)} nến ({self.interval}).")
            return df
        except Exception as e:
            logger.warning(f"Lỗi tải dữ liệu ({e}). Chuyển sang Data Mô phỏng (Mock Data)...")
            return self.generate_mock_data()

    def generate_mock_data(self):
        """Khởi tạo Mock Data chuẩn cấu trúc VN-Index khi fetch block."""
        dates = pd.date_range(start=self.start_date, end=pd.Timestamp.today(), freq='B')
        np.random.seed(42)
        # Giả lập lợi nhuận bình thường
        returns = np.random.normal(0.0002, 0.015, len(dates))
        
        # Tiêm Anomaly Events
        if len(returns) > 150: returns[150] = -0.05 # Fake Nuke
        if len(returns) > 350: returns[350] = -0.07 # Real Nuke
        if len(returns) > 351: returns[351] = -0.04 # Follow through
        
        price = 1000 * np.exp(np.cumsum(returns))
        volume = np.random.randint(200_000_000, 800_000_000, len(dates))
        
        # Simulate Shakeout: Vol thấp ở Fake Nuke
        if len(volume) > 150: volume[150] = 50_000_000 
        
        df = pd.DataFrame({'Close': price, 'Volume': volume}, index=dates)
        df['Open'] = df['Close'] * np.random.uniform(0.99, 1.01, len(dates))
        df['High'] = df[['Open', 'Close']].max(axis=1) * 1.01
        df['Low'] = df[['Open', 'Close']].min(axis=1) * 0.99
        return df

    def run(self):
        df = self.fetch_data()
        
        # 1. TIME SERIES
        df = self.ts_proc.calculate_log_returns(df)
        df = self.ts_proc.smooth_data(df, window=self.smooth_window)
        
        # 2. STATS
        df = self.stats_analyzer.calculate_rolling_stats(df)
        
        # 3. MICROSTRUCTURE VOL PROFILE
        vp_df = df.tail(self.vp_lookback) 
        vp, poc, vah, val = self.vp_analyzer.calc_volume_profile(vp_df)
        
        # 4. CALCULUS & LA
        df = self.calc_alg.calculate_kinematics(df, col='Close_Smoothed')
        feat_matrix, corr_matrix = self.calc_alg.build_feature_matrix(
            df, ['Log_Return', 'Velocity', 'Acceleration', 'Volume']
        )
        logger.info(f"Correlation Matrix Shape (Linear Algebra): {corr_matrix.shape}")
        
        # 5. ANOMALY DETECTION
        df = self.detector.detect(df, vp_val=val, vp_poc=poc, vp_vah=vah)
        
        # 6. MACHINE LEARNING PREDICTION
        logger.info("Đang huấn luyện mô hình Random Forest Classifier (ML)...")
        ml_acc, prob_up = self.ml_predictor.train_and_predict(df)
        
        # Thống kê toàn bộ chuỗi
        time_fmt = '%Y-%m-%d %H:%M' if self.interval != '1D' else '%Y-%m-%d'
        latest = df.iloc[-1]
        
        total_candles = len(df)
        mean_close_total = df['Close'].mean()
        std_close_total = df['Close'].std()
        skew_close_total = df['Close'].skew()
        skew_direction = "Lệch Phải (Tích cực)" if skew_close_total > 0 else ("Lệch Trái (Tiêu cực)" if skew_close_total < 0 else "Đối xứng")
        
        # IN KẾT QUẢ RÕ RÀNG
        logger.info(f"\n========== LATEST MARKET SNAPSHOT ({self.interval}) ==========")
        logger.info(f"Thời gian: {df.index[-1].strftime(time_fmt)}")
        logger.info(f"Tổng nến kiểm tra: {total_candles}")
        logger.info(f"Mean (Close): {mean_close_total:.2f}")
        logger.info(f"STDV (Close - Toàn cục): {std_close_total:.2f}")
        logger.info(f"Skew: {skew_close_total:.2f} ({skew_direction})")
        logger.info(f"Giá Close: {latest['Close']:.2f}")
        logger.info(f"Khối lượng: {latest['Volume']:,.0f}")
        logger.info(f"Z-Score (Độ cực đoan): {latest['Z_Score']:.2f}")
        logger.info(f"P-Value: {latest['P_Value']:.4f}")
        logger.info(f"Độ xiên (Skewness): {latest['Rolling_Skew']:.2f}")
        logger.info(f"Giá trị tương đối (Rel Z-Score): {latest['Relative_Z_Score']:.2f}")
        logger.info(f"Động lượng GIA TỐC (Calculus): {latest['Acceleration']:.4f}")
        logger.info(f"NHẬN DIỆN MÔ HÌNH: {latest['Signal']}")
        if ml_acc is not None:
            logger.info(f"ML ACCURACY (Out-of-sample Testing): {ml_acc:.2%}")
            logger.info(f"ML PREDICTION - XÁC SUẤT TĂNG GIÁ (T+1): {prob_up:.2%}")
        
        # Lịch sử Nuke
        nukes = df[df['Signal'] != 'Normal']
        if not nukes.empty:
            logger.info(f"\n=> TÌM THẤY {len(nukes)} SỰ KIỆN QUAN TRỌNG TRONG LỊCH SỬ:")
            for idx, row in nukes.iterrows():
                logger.info(f"[{idx.strftime(time_fmt)}] Drop: {row['Log_Return']:.2%} | Phân loại: {row['Signal']}")

        # Trực quan hóa
        self.plot_dashboard(df, vp, poc, vah, val, ml_acc, prob_up)
        return df
        
    def plot_dashboard(self, df, vp, poc, vah, val, ml_acc=None, prob_up=None):
        import matplotlib.ticker as ticker
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.6, 1, 1], width_ratios=[1, 1, 1.4], hspace=0.35, wspace=0.3)
        
        time_fmt = '%Y-%m-%d %H:%M' if self.interval != '1D' else '%Y-%m-%d'
        latest = df.iloc[-1]
        x_axis = np.arange(len(df))
        
        # Helper function for formatting x-axis dates
        def format_date(x, pos=None):
            idx = int(round(x))
            if idx >= 0 and idx < len(df):
                return df.index[idx].strftime(time_fmt)
            return ''

        # ====== 1. Biểu đồ Giá và Value Area (top, full width) ======
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(x_axis, df['Close'], label='Close Price', color='navy', linewidth=1.5)
        ax1.axhline(poc, color='r', linestyle='--', label=f'POC ({poc:.1f})')
        ax1.axhline(vah, color='g', alpha=0.5, label='Value Area')
        ax1.axhline(val, color='g', alpha=0.5)
        ax1.fill_between(x_axis, val, vah, color='green', alpha=0.05)
        
        # Scatter Signals
        for i in range(len(df)):
            row = df.iloc[i]
            idx = df.index[i]
            if 'Fake' in row['Signal']:
                ax1.scatter(i, row['Close'], color='orange', marker='^', s=150, zorder=5, label='Fake Nuke' if 'Fake Nuke' not in ax1.get_legend_handles_labels()[1] else '')
                ax1.text(i, row['Close'], f" {idx.strftime(time_fmt)}", color='darkorange', fontsize=8, fontweight='bold', va='bottom')
            elif 'Real' in row['Signal']:
                ax1.scatter(i, row['Close'], color='red', marker='v', s=150, zorder=5, label='Real Nuke' if 'Real Nuke' not in ax1.get_legend_handles_labels()[1] else '')
                ax1.text(i, row['Close'], f" {idx.strftime(time_fmt)}", color='darkred', fontsize=8, fontweight='bold', va='top')
            elif 'Bullish' in row['Signal']:
                ax1.scatter(i, row['Close'], color='green', marker='^', s=150, zorder=5, label='Bullish Reversal' if 'Bullish Reversal' not in ax1.get_legend_handles_labels()[1] else '')
                ax1.text(i, row['Close'], f" {idx.strftime(time_fmt)}", color='green', fontsize=8, fontweight='bold', va='bottom')

        ax1.set_title(f'Price Action, Anomalies & Value Area ({self.interval})', fontsize=14, fontweight='bold')
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax1.legend(loc='upper left')
        
        # ====== 2. Volume Profile (middle-left) ======
        ax2 = fig.add_subplot(gs[1, 0])
        self.vp_analyzer.plot_volume_profile(ax2, vp, poc, vah, val)
        
        # ====== 3. Bell Curve (middle-center) ======
        ax3 = fig.add_subplot(gs[1, 1])
        self.stats_analyzer.plot_bell_curve(ax3, df['Log_Return'])
        
        # ====== 4. LATEST MARKET SNAPSHOT Panel (middle-right) ======
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        
        # Determine signal color
        signal_text = latest['Signal']
        if 'Fake' in signal_text:
            signal_color = '#FF8C00'  # orange
        elif 'Real' in signal_text:
            signal_color = '#DC143C'  # crimson
        else:
            signal_color = '#228B22'  # green
        
        # Build info text
        mean_close_total = df['Close'].mean()
        std_close_total = df['Close'].std()
        skew_close_total = df['Close'].skew()
        skew_dir = "Lệch Phải" if skew_close_total > 0 else ("Lệch Trái" if skew_close_total < 0 else "Đối xứng")
        
        snapshot_lines = [
            ('Thời gian', df.index[-1].strftime(time_fmt)),
            ('Tổng nến', f'{len(df)}'),
            ('Mean(Close)', f'{mean_close_total:.2f}'),
            ('STDV(Close)', f'{std_close_total:.2f}'),
            ('Skew', f'{skew_close_total:.2f} ({skew_dir})'),
            ('Giá Close', f'{latest["Close"]:.2f}'),
            ('Khối lượng', f'{latest["Volume"]:,.0f}'),
            ('Z-Score', f'{latest["Z_Score"]:.2f}'),
            ('P-Value', f'{latest["P_Value"]:.4f}'),
            ('Roll Skew', f'{latest["Rolling_Skew"]:.2f}'),
            ('Rel Z-Score', f'{latest["Relative_Z_Score"]:.2f}'),
            ('Gia tốc', f'{latest["Acceleration"]:.4f}'),
        ]
        if ml_acc is not None:
            snapshot_lines.append(('ML Accuracy', f'{ml_acc:.2%}'))
            snapshot_lines.append(('Xác suất tăng T+1', f'{prob_up:.2%}'))
        
        # Title
        ax4.text(0.5, 0.98, f'MARKET SNAPSHOT ({self.interval})', transform=ax4.transAxes,
                 fontsize=12, fontweight='bold', ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='none'),
                 color='white')
        
        # Signal badge
        ax4.text(0.5, 0.88, signal_text, transform=ax4.transAxes,
                 fontsize=11, fontweight='bold', ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=signal_color, edgecolor='none', alpha=0.9),
                 color='white')
        
        # Data rows (2 columns)
        y_start = 0.72
        spacing = 0.1
        for i, (label, value) in enumerate(snapshot_lines):
            col = i % 2
            row = i // 2
            
            x_label = 0.04 if col == 0 else 0.54
            x_value = 0.46 if col == 0 else 0.96
            y = y_start - row * spacing
            
            ax4.text(x_label, y, label + ':', transform=ax4.transAxes,
                     fontsize=9, fontweight='bold', ha='left', va='top', color='#333333')
            ax4.text(x_value, y, value, transform=ax4.transAxes,
                     fontsize=9, ha='right', va='top', color='#1a1a2e',
                     fontfamily='monospace')
        
        # Background box
        ax4.add_patch(plt.Rectangle((0.02, 0.01), 0.96, 0.97, transform=ax4.transAxes,
                                     facecolor='#f0f4f8', edgecolor='#c0c0c0', linewidth=1.5,
                                     zorder=-1, clip_on=False))
        
        # ====== 5. Z-Score Timeline (bottom-left) ======
        ax5 = fig.add_subplot(gs[2, 0:2])
        ax5.plot(x_axis, df['Z_Score'], color='steelblue', linewidth=1, alpha=0.8)
        ax5.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax5.axhline(-1.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Nuke Threshold (-1.5)')
        ax5.axhline(1.5, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax5.fill_between(x_axis, -1.5, df['Z_Score'], where=df['Z_Score'] < -1.5,
                         color='red', alpha=0.3, label='Extreme Zone')
        ax5.set_title('Z-Score Timeline (Tail Risk Monitor)', fontsize=11, fontweight='bold')
        ax5.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax5.legend(loc='lower left', fontsize=8)
        ax5.set_ylabel('Z-Score')
        
        # ====== 6. Acceleration Timeline (bottom-right) ======
        ax6 = fig.add_subplot(gs[2, 2])
        accel_data = df['Acceleration'].dropna().tail(120)
        colors = ['#DC143C' if v < 0 else '#228B22' for v in accel_data.values]
        ax6.bar(range(len(accel_data)), accel_data.values, color=colors, alpha=0.7, width=1.0)
        ax6.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax6.set_title('Acceleration (Momentum Change)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Acceleration')
        ax6.set_xlabel(f'Last {len(accel_data)} bars')
        
        plt.tight_layout()
        plt.show()

import tkinter as tk
from tkinter import ttk, messagebox

class AppGUI:
    INTERVAL_OPTIONS = {
        'Ngày (1D)': '1D',
        '15 phút (15m)': '15m',
        '1 phút (1m)': '1m',
    }

    def __init__(self, root):
        self.root = root
        self.root.title("VN-Index Quant Anomaly Detector")
        self.root.geometry("500x320")
        self.root.resizable(False, False)
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="VN-Index Quantitative Research", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Start Date
        start_frame = ttk.Frame(main_frame)
        start_frame.pack(fill=tk.X, pady=4)
        ttk.Label(start_frame, text="Ngày bắt đầu (YYYY-MM-DD):", width=26).pack(side=tk.LEFT)
        self.start_entry = ttk.Entry(start_frame)
        self.start_entry.insert(0, "2018-01-01")
        self.start_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # End Date
        end_frame = ttk.Frame(main_frame)
        end_frame.pack(fill=tk.X, pady=4)
        ttk.Label(end_frame, text="Ngày kết thúc (YYYY-MM-DD):", width=26).pack(side=tk.LEFT)
        self.end_entry = ttk.Entry(end_frame)
        self.end_entry.insert(0, pd.Timestamp.today().strftime('%Y-%m-%d'))
        self.end_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Interval (Timeframe)
        interval_frame = ttk.Frame(main_frame)
        interval_frame.pack(fill=tk.X, pady=4)
        ttk.Label(interval_frame, text="Khung thời gian (Timeframe):", width=26).pack(side=tk.LEFT)
        self.interval_var = tk.StringVar(value='Ngày (1D)')
        self.interval_combo = ttk.Combobox(interval_frame, textvariable=self.interval_var, 
                                           values=list(self.INTERVAL_OPTIONS.keys()), state='readonly')
        self.interval_combo.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.interval_combo.bind('<<ComboboxSelected>>', self.on_interval_change)
        
        # Hint label
        self.hint_label = ttk.Label(main_frame, text="💡 Intraday (15m/1m): nên chọn khoảng thời gian ngắn (vài ngày)", 
                                    foreground="gray", font=("Arial", 8))
        self.hint_label.pack(pady=2)
        
        # Status Label
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.pack(pady=5)
        
        # Run Button
        self.run_btn = ttk.Button(main_frame, text="⚡ Run Backtest & Create Dashboard", command=self.run_model)
        self.run_btn.pack(pady=10)

    def on_interval_change(self, event=None):
        """Tự động đề xuất khoảng ngày phù hợp khi đổi timeframe."""
        interval = self.INTERVAL_OPTIONS[self.interval_var.get()]
        today = pd.Timestamp.today()
        if interval == '1m':
            suggested_start = (today - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, suggested_start)
        elif interval == '15m':
            suggested_start = (today - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, suggested_start)
        else:
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, '2018-01-01')
        
    def run_model(self):
        start_date = self.start_entry.get().strip()
        end_date = self.end_entry.get().strip()
        interval = self.INTERVAL_OPTIONS[self.interval_var.get()]
        
        if not start_date or not end_date:
            messagebox.showerror("Lỗi", "Vui lòng nhập ngày bắt đầu và ngày kết thúc.")
            return
            
        self.status_label.config(text=f"Đang lấy dữ liệu ({interval}) và phân tích...", foreground="blue")
        self.run_btn.config(state='disabled')
        self.root.update()
        
        try:
            engine = QuantModelEngine(ticker='^VNINDEX', start_date=start_date, end_date=end_date, interval=interval)
            engine.run()
            self.status_label.config(text="✅ Kiểm tra biểu đồ đã mở!", foreground="green")
        except Exception as e:
            messagebox.showerror("Lỗi hệ thống", f"Có lỗi xảy ra: {str(e)}")
            self.status_label.config(text="❌ Thất bại", foreground="red")
        finally:
            self.run_btn.config(state='normal')

if __name__ == "__main__":
    print("\n[+] ĐANG KHỞI ĐỘNG GIAO DIỆN HỆ THỐNG...")
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()
