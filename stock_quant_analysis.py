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
import os
import tkinter as tk
from tkinter import ttk, messagebox
from vnstock import Vnstock

# Đảm bảo in Tiếng Việt không lỗi trên Terminal Windows
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# CẤU HÌNH & LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('StockQuantModel')
sns.set_theme(style="darkgrid")

DATA_DIR = r"D:\Lập trình\Loc co phieu\data_library"

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
        
        # P-Value (Cumulative Distribution Function)
        df['P_Value'] = stats.norm.cdf(df['Z_Score'])
        
        # Rolling Skewness
        df['Rolling_Skew'] = df[return_col].rolling(window=self.window).skew()
        
        # Skewness ngắn hạn để bắt đáy/đảo chiều (Shakeout)
        short_term = max(3, self.window // 4)
        df['Skewness_ShortTerm'] = df[return_col].rolling(window=short_term).skew()

        # Relative Value: So sánh Z-Score hiện tại với TB Lịch sử
        df['Z_Score_Avg_Hist'] = df['Z_Score'].rolling(window=self.window * 3).mean()
        df['Relative_Z_Score'] = df['Z_Score'] - df['Z_Score_Avg_Hist']
        
        return df

    @staticmethod
    def plot_bell_curve(ax, returns, title='Fat Tails Real Life Return vs Normal Curve'):
        """Vẽ biểu đồ phân phối chuẩn (Bell Curve) đè lên biểu đồ thực tế."""
        returns = returns.dropna()
        if returns.empty: return
        mu, std = stats.norm.fit(returns)
        
        sns.histplot(returns, bins=50, stat='density', alpha=0.6, color='steelblue', label='Actual Returns', ax=ax)
        
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
        if df.empty: return None, 0, 0, 0
        min_price = df[price_col].min()
        max_price = df[price_col].max()
        if min_price == max_price: return None, min_price, min_price, min_price
        
        price_bins = np.linspace(min_price, max_price, self.bins)
        
        df = df.copy()
        df['Price_Bin'] = np.digitize(df[price_col], price_bins)
        
        vp = df.groupby('Price_Bin')[vol_col].sum().reset_index()
        vp['Price'] = price_bins[vp['Price_Bin'] - 1]
        vp = vp.sort_values('Price').reset_index(drop=True)
        
        total_volume = vp[vol_col].sum()
        poc_idx = vp[vol_col].idxmax()
        poc_price = vp.loc[poc_idx, 'Price']
        
        target_vol = total_volume * self.val_area_pct
        current_vol = vp.loc[poc_idx, vol_col]
        upper_idx, lower_idx = poc_idx, poc_idx
        
        while current_vol < target_vol:
            up_vol = vp.loc[upper_idx + 1, vol_col] if upper_idx + 1 < len(vp) else -1
            down_vol = vp.loc[lower_idx - 1, vol_col] if lower_idx - 1 >= 0 else -1
            
            if up_vol == -1 and down_vol == -1: break
                
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
        if vp is None or vp.empty: return
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
    @staticmethod
    def calculate_kinematics(df, col='Close_Smoothed'):
        df = df.copy()
        if col not in df.columns: return df
        # Đạo hàm bậc 1: Vận tốc
        df['Velocity'] = np.gradient(df[col].bfill().ffill())
        # Đạo hàm bậc 2: Gia tốc
        df['Acceleration'] = np.gradient(df['Velocity'])
        return df
        
    @staticmethod
    def build_feature_matrix(df, cols):
        df_clean = df[cols].dropna()
        if df_clean.empty: return None, None
        feature_matrix = df_clean.values
        corr_matrix = np.corrcoef(feature_matrix, rowvar=False)
        return feature_matrix, corr_matrix

# ==========================================
# 5. ANOMALY DETECTION (SIGNAL RULE)
# ==========================================
class AnomalyDetector:
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
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
    def prepare_data(self, df):
        features = ['Log_Return', 'Velocity', 'Acceleration', 'Z_Score', 'Rolling_Skew', 'Volume']
        df_ml = df.dropna(subset=features).copy()
        df_ml['Target'] = (df_ml['Close'].shift(-1) > df_ml['Close']).astype(int)
        return df_ml.dropna(subset=['Target']), features
        
    def train_and_predict(self, df):
        df_ml, features = self.prepare_data(df)
        if len(df_ml) < 50:
            logger.warning("Không đủ dữ liệu cho Machine Learning.")
            return None, None, None
            
        split_idx = int(len(df_ml) * 0.8)
        train = df_ml.iloc[:split_idx]
        test = df_ml.iloc[split_idx:]
        
        X_train, y_train = train[features], train['Target']
        X_test, y_test = test[features], test['Target']
        
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        latest_features = df.iloc[-1:][features]
        if not latest_features.isna().any().any():
            prob_up = self.model.predict_proba(latest_features)[0][1]
        else:
            prob_up = 0.5
            
        # --- Bổ sung tính toán EV (Expected Value) ---
        # Tính mức lợi nhuận trung bình khi tăng và mức lỗ trung bình khi giảm (lookback 120 bars)
        lookback = 120
        recent_returns = df['Log_Return'].tail(lookback)
        avg_gain = recent_returns[recent_returns > 0].mean()
        avg_loss = recent_returns[recent_returns < 0].mean()
        
        # Xử lý trường hợp NaN
        avg_gain = avg_gain if not pd.isna(avg_gain) else 0.01
        avg_loss = avg_loss if not pd.isna(avg_loss) else -0.01
        
        # EV = (P_up * Avg_Gain) + (P_down * Avg_Loss)
        ev = (prob_up * avg_gain) + ((1 - prob_up) * avg_loss)
            
        return acc, prob_up, ev

# ==========================================
# ORCHESTRATION ENGINE
# ==========================================
class QuantModelEngine:
    INTERVAL_WINDOWS = {
        '1D': {'stats': 20, 'smooth': 5, 'vp_lookback': 60},
        '1H': {'stats': 24, 'smooth': 6, 'vp_lookback': 120},
        '15m': {'stats': 30, 'smooth': 8, 'vp_lookback': 160},
    }

    def __init__(self, ticker, start_date, end_date, interval='1D'):
        self.ticker = ticker
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
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
        logger.info(f"Đang tải dữ liệu {self.ticker} trực tiếp | Interval: {self.interval} | {self.start_date.strftime('%Y-%m-%d')} -> {self.end_date.strftime('%Y-%m-%d')}")
        try:
            stock = Vnstock().stock(symbol=self.ticker, source="KBS")
            df = stock.quote.history(start=self.start_date.strftime('%Y-%m-%d'), end=self.end_date.strftime('%Y-%m-%d'), interval=self.interval)
            
            if df is None or df.empty:
                raise ValueError("Không tìm thấy dữ liệu.")
                
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df.columns = df.columns.str.capitalize()
            
            # Loại bỏ nến lỗi/nghỉ
            df = df.dropna(subset=['Close'])
            df = df[df.index.dayofweek < 5]
            df = df[df['Volume'] > 0]
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            
            logger.info(f"Tải và lọc dữ liệu thành công: {len(df)} nến ({self.interval}).")
            return df
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu từ vnstock: {e}")
            raise e

    def run(self):
        df = self.fetch_data()
        
        df = self.ts_proc.calculate_log_returns(df)
        df = self.ts_proc.smooth_data(df, window=self.smooth_window)
        df = self.stats_analyzer.calculate_rolling_stats(df)
        
        vp_df = df.tail(self.vp_lookback) 
        vp, poc, vah, val = self.vp_analyzer.calc_volume_profile(vp_df)
        
        df = self.calc_alg.calculate_kinematics(df, col='Close_Smoothed')
        feat_matrix, corr_matrix = self.calc_alg.build_feature_matrix(
            df, ['Log_Return', 'Velocity', 'Acceleration', 'Volume']
        )
        
        df = self.detector.detect(df, vp_val=val, vp_poc=poc, vp_vah=vah)
        
        logger.info("Đang huấn luyện mô hình Machine Learning...")
        ml_acc, prob_up, ev = self.ml_predictor.train_and_predict(df)
        
        self.print_results(df, ml_acc, prob_up, ev)
        self.plot_dashboard(df, vp, poc, vah, val, ml_acc, prob_up, ev)
        return df

    def print_results(self, df, ml_acc, prob_up, ev):
        total_candles = len(df)
        latest = df.iloc[-1]
        mean_close_total = df['Close'].mean()
        std_close_total = df['Close'].std()
        skew_close_total = df['Close'].skew()
        skew_direction = "Lệch Phải (Tích cực)" if skew_close_total > 0 else ("Lệch Trái (Tiêu cực)" if skew_close_total < 0 else "Đối xứng")

        logger.info(f"\n========== LATEST SNAPSHOT: {self.ticker} ({self.interval}) ==========")
        logger.info(f"Thời gian: {df.index[-1]}")
        logger.info(f"Tổng nến kiểm tra: {total_candles}")
        logger.info(f"Mean (Close): {mean_close_total:.2f}")
        logger.info(f"STDV (Close - Toàn cục): {std_close_total:.2f}")
        logger.info(f"Skew: {skew_close_total:.2f} ({skew_direction})")
        logger.info(f"Giá Close: {latest['Close']:.2f}")
        logger.info(f"Z-Score: {latest['Z_Score']:.2f}")
        logger.info(f"P-Value: {latest['P_Value']:.4f}")
        logger.info(f"Độ xiên (Roll Skew): {latest['Rolling_Skew']:.2f}")
        logger.info(f"Giá trị tương đối (Rel Z-Score): {latest['Relative_Z_Score']:.2f}")
        logger.info(f"Động lượng GIA TỐC: {latest['Acceleration']:.4f}")
        logger.info(f"NHẬN DIỆN: {latest['Signal']}")
        if ml_acc is not None:
            logger.info(f"ML Prediction T+1: {prob_up:.2%}")
            logger.info(f"Expected Value (EV): {ev:.4%}")

    def plot_dashboard(self, df, vp, poc, vah, val, ml_acc=None, prob_up=None, ev=None):
        import matplotlib.ticker as mticker
        
        fig = plt.figure(figsize=(20, 12))
        fig.canvas.manager.set_window_title(f"Analysis: {self.ticker}")
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.6, 1, 1], width_ratios=[1, 1, 1.4], hspace=0.35, wspace=0.3)
        
        time_fmt = '%Y-%m-%d %H:%M' if 'd' not in self.interval else '%Y-%m-%d'
        latest = df.iloc[-1]
        x_axis = np.arange(len(df))
        
        def format_date(x, pos=None):
            idx = int(round(x))
            if 0 <= idx < len(df):
                return df.index[idx].strftime(time_fmt)
            return ''

        # Biểu đồ giá
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(x_axis, df['Close'], label='Close Price', color='navy', linewidth=1.5)
        ax1.axhline(poc, color='r', linestyle='--', label=f'POC ({poc:.1f})')
        ax1.axhline(vah, color='g', alpha=0.5, label='Value Area')
        ax1.axhline(val, color='g', alpha=0.5)
        ax1.fill_between(x_axis, val, vah, color='green', alpha=0.05)
        
        for i in range(len(df)):
            row = df.iloc[i]
            idx = df.index[i]
            if 'Fake' in row['Signal']:
                ax1.scatter(i, row['Close'], color='orange', marker='^', s=100)
            elif 'Real' in row['Signal']:
                ax1.scatter(i, row['Close'], color='red', marker='v', s=100)
            elif 'Bullish' in row['Signal']:
                ax1.scatter(i, row['Close'], color='green', marker='^', s=100)

        ax1.set_title(f'{self.ticker} - Price Action & Anomalies ({self.interval})', fontsize=14, fontweight='bold')
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(format_date))
        ax1.legend(loc='upper left')
        
        # Volume Profile
        ax2 = fig.add_subplot(gs[1, 0])
        self.vp_analyzer.plot_volume_profile(ax2, vp, poc, vah, val)
        
        # Bell Curve
        ax3 = fig.add_subplot(gs[1, 1])
        self.stats_analyzer.plot_bell_curve(ax3, df['Log_Return'])
        
        # Snapshot Panel
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        signal_color = '#FF8C00' if 'Fake' in latest['Signal'] else ('#DC143C' if 'Real' in latest['Signal'] else '#228B22')
        
        ax4.text(0.5, 0.95, f'SNAPSHOT: {self.ticker}', transform=ax4.transAxes, fontsize=12, fontweight='bold', ha='center', bbox=dict(facecolor='#1a1a2e', alpha=0.1), color='black')
        ax4.text(0.5, 0.85, latest['Signal'], transform=ax4.transAxes, fontsize=11, fontweight='bold', ha='center', bbox=dict(facecolor=signal_color, alpha=0.8), color='white')
        
        mean_close_total = df['Close'].mean()
        std_close_total = df['Close'].std()
        skew_close_total = df['Close'].skew()
        skew_dir = "Lệch Phải" if skew_close_total > 0 else ("Lệch Trái" if skew_close_total < 0 else "Đối xứng")

        info = [
            ('Time', df.index[-1].strftime(time_fmt)),
            ('Tổng nến', f"{len(df)}"),
            ('Mean(Close)', f"{mean_close_total:.2f}"),
            ('STDV(Close)', f"{std_close_total:.2f}"),
            ('Skew', f"{skew_close_total:.2f} ({skew_dir})"),
            ('Close', f"{latest['Close']:.2f}"),
            ('Z-Score', f"{latest['Z_Score']:.2f}"),
            ('P-Value', f"{latest['P_Value']:.4f}"),
            ('Roll Skew', f"{latest['Rolling_Skew']:.2f}"),
            ('Rel Z-Score', f"{latest['Relative_Z_Score']:.2f}"),
            ('Acceleration', f"{latest['Acceleration']:.4f}")
        ]
        if ml_acc is not None:
            info.append(('ML Prob Up', f"{prob_up:.2%}"))
            info.append(('Exp Value (EV)', f"{ev:.4%}"))

        y_start = 0.72
        spacing = 0.12
        for i, (label, val_text) in enumerate(info):
            col = i % 2
            row = i // 2
            
            x_label = 0.04 if col == 0 else 0.54
            x_value = 0.46 if col == 0 else 0.96
            y = y_start - row * spacing
            
            ax4.text(x_label, y, f"{label}:", transform=ax4.transAxes, fontsize=9, fontweight='bold', ha='left', va='top')
            ax4.text(x_value, y, val_text, transform=ax4.transAxes, fontsize=9, ha='right', va='top', fontfamily='monospace')
        
        # Z-Score timeline
        ax5 = fig.add_subplot(gs[2, 0:2])
        ax5.plot(x_axis, df['Z_Score'], color='steelblue')
        ax5.axhline(-1.5, color='red', linestyle='--')
        ax5.set_title('Z-Score Monitor')
        ax5.xaxis.set_major_formatter(mticker.FuncFormatter(format_date))
        
        # Acceleration
        ax6 = fig.add_subplot(gs[2, 2])
        accel = df['Acceleration'].tail(60)
        ax6.bar(range(len(accel)), accel, color=['g' if x>0 else 'r' for x in accel])
        ax6.set_title('Acceleration (Momentum)')
        
        plt.tight_layout()
        plt.show()

# ==========================================
# GUI APPLICATION
# ==========================================
class StockAppGUI:
    INTERVAL_OPTIONS = {
        'Ngày (1D)': '1D',
        '1 giờ (1H)': '1H',
        '15 phút (15m)': '15m',
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Stock Quant Anomaly Detector")
        self.root.geometry("500x400")
        
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Stock Quantitative Analysis", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Symbol
        symbol_frame = ttk.Frame(main_frame)
        symbol_frame.pack(fill=tk.X, pady=5)
        ttk.Label(symbol_frame, text="Mã cổ phiếu (Symbol):", width=25).pack(side=tk.LEFT)
        self.symbol_entry = ttk.Entry(symbol_frame)
        self.symbol_entry.insert(0, "HPG")
        self.symbol_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Start Date
        start_frame = ttk.Frame(main_frame)
        start_frame.pack(fill=tk.X, pady=5)
        ttk.Label(start_frame, text="Ngày bắt đầu (YYYY-MM-DD):", width=25).pack(side=tk.LEFT)
        self.start_entry = ttk.Entry(start_frame)
        self.start_entry.insert(0, "2023-01-01")
        self.start_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # End Date
        end_frame = ttk.Frame(main_frame)
        end_frame.pack(fill=tk.X, pady=5)
        ttk.Label(end_frame, text="Ngày kết thúc (YYYY-MM-DD):", width=25).pack(side=tk.LEFT)
        self.end_entry = ttk.Entry(end_frame)
        self.end_entry.insert(0, pd.Timestamp.today().strftime('%Y-%m-%d'))
        self.end_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Interval
        int_frame = ttk.Frame(main_frame)
        int_frame.pack(fill=tk.X, pady=5)
        ttk.Label(int_frame, text="Khung thời gian:", width=25).pack(side=tk.LEFT)
        self.int_var = tk.StringVar(value='Ngày (1D)')
        self.int_combo = ttk.Combobox(int_frame, textvariable=self.int_var, values=list(self.INTERVAL_OPTIONS.keys()), state='readonly')
        self.int_combo.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.pack(pady=10)
        
        ttk.Button(main_frame, text="⚡ RUN ANALYSIS", command=self.run_analysis).pack(pady=10, fill=tk.X)

    def run_analysis(self):
        symbol = self.symbol_entry.get().strip().upper()
        start = self.start_entry.get().strip()
        end = self.end_entry.get().strip()
        interval = self.INTERVAL_OPTIONS[self.int_var.get()]
        
        if not symbol:
            messagebox.showerror("Error", "Vui lòng nhập mã cổ phiếu")
            return
            
        self.status_label.config(text=f"Analyzing {symbol}...", foreground="blue")
        self.root.update()
        
        try:
            engine = QuantModelEngine(symbol, start, end, interval)
            engine.run()
            self.status_label.config(text="✅ Done!", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="❌ Error", foreground="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockAppGUI(root)
    root.mainloop()
