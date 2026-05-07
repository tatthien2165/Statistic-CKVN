import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('WebQuantModel')

app = Flask(__name__)
CORS(app)

# ==========================================
# CORE ANALYSIS CLASSES (no matplotlib/tkinter)
# ==========================================
class TimeSeriesProcessor:
    @staticmethod
    def calculate_log_returns(df, price_col='Close'):
        df = df.copy()
        df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))
        return df

    @staticmethod
    def smooth_data(df, col='Close', window=5):
        df = df.copy()
        df[f'{col}_Smoothed'] = df[col].rolling(window=window).mean()
        return df


class StatisticalAnalyzer:
    def __init__(self, window=20):
        self.window = window

    def calculate_rolling_stats(self, df, return_col='Log_Return'):
        df = df.copy()
        df['Rolling_Mean'] = df[return_col].rolling(window=self.window).mean()
        df['Rolling_Std'] = df[return_col].rolling(window=self.window).std()
        df['Z_Score'] = (df[return_col] - df['Rolling_Mean']) / df['Rolling_Std']
        df['P_Value'] = stats.norm.cdf(df['Z_Score'])
        df['Rolling_Skew'] = df[return_col].rolling(window=self.window).skew()
        short_term = max(3, self.window // 4)
        df['Skewness_ShortTerm'] = df[return_col].rolling(window=short_term).skew()
        df['Z_Score_Avg_Hist'] = df['Z_Score'].rolling(window=self.window * 3).mean()
        df['Relative_Z_Score'] = df['Z_Score'] - df['Z_Score_Avg_Hist']
        return df


class MarketMicrostructure:
    def __init__(self, bins=50, val_area_pct=0.70):
        self.bins = bins
        self.val_area_pct = val_area_pct

    def calc_volume_profile(self, df, price_col='Close', vol_col='Volume'):
        if df.empty: return None, 0, 0, 0
        min_price, max_price = df[price_col].min(), df[price_col].max()
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


class CalculusLinearAlgebra:
    @staticmethod
    def calculate_kinematics(df, col='Close_Smoothed'):
        df = df.copy()
        if col not in df.columns: return df
        df['Velocity'] = np.gradient(df[col].bfill().ffill())
        df['Acceleration'] = np.gradient(df['Velocity'])
        return df


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
            is_nuke = (curr['Z_Score'] <= self.z_thresh) and (curr['P_Value'] <= self.p_thresh)
            if is_nuke:
                accel_improving = curr['Acceleration'] > prev['Acceleration']
                price_below_val = curr['Close'] < vp_val
                skew_positive = curr['Skewness_ShortTerm'] > 0
                skew_deep_neg = curr['Rolling_Skew'] < -1.0
                if accel_improving and price_below_val and skew_positive:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'Fake Nuke'
                elif skew_deep_neg or not accel_improving:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'Real Nuke'
                continue
            recovering_z = (curr['Z_Score'] > -1.0) and (prev['Z_Score'] < -1.5)
            strong_accel = (curr['Acceleration'] > 0) and (curr['Acceleration'] > prev['Acceleration'])
            price_reclaiming_val = (curr['Close'] > vp_val) and (prev['Close'] <= vp_val)
            price_reclaiming_poc = (curr['Close'] > vp_poc) and (prev['Close'] <= vp_poc)
            if (recovering_z and strong_accel) or (price_reclaiming_val and strong_accel) or (price_reclaiming_poc and curr['Z_Score'] > 0):
                if curr['Log_Return'] > 0:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'Bullish Reversal'
        return df


class MachineLearningPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    def train_and_predict(self, df):
        features = ['Log_Return', 'Velocity', 'Acceleration', 'Z_Score', 'Rolling_Skew', 'Volume']
        df_ml = df.dropna(subset=features).copy()
        df_ml['Target'] = (df_ml['Close'].shift(-1) > df_ml['Close']).astype(int)
        df_ml = df_ml.dropna(subset=['Target'])
        if len(df_ml) < 50:
            return None, None, None
        split_idx = int(len(df_ml) * 0.8)
        train, test = df_ml.iloc[:split_idx], df_ml.iloc[split_idx:]
        X_train, y_train = train[features], train['Target']
        X_test, y_test = test[features], test['Target']
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        latest_features = df.iloc[-1:][features]
        prob_up = self.model.predict_proba(latest_features)[0][1] if not latest_features.isna().any().any() else 0.5
        lookback = 120
        recent_returns = df['Log_Return'].tail(lookback)
        avg_gain = recent_returns[recent_returns > 0].mean()
        avg_loss = recent_returns[recent_returns < 0].mean()
        avg_gain = avg_gain if not pd.isna(avg_gain) else 0.01
        avg_loss = avg_loss if not pd.isna(avg_loss) else -0.01
        ev = (prob_up * avg_gain) + ((1 - prob_up) * avg_loss)
        return acc, prob_up, ev


# ==========================================
# ANALYSIS ENGINE
# ==========================================
INTERVAL_WINDOWS = {
    '1D': {'stats': 20, 'smooth': 5, 'vp_lookback': 60},
    '1H': {'stats': 24, 'smooth': 6, 'vp_lookback': 120},
    '15m': {'stats': 30, 'smooth': 8, 'vp_lookback': 160},
}

def fetch_stock_data(ticker, start_date, end_date, interval):
    from vnstock import Vnstock
    stock = Vnstock().stock(symbol=ticker, source="KBS")
    df = stock.quote.history(start=start_date, end=end_date, interval=interval)
    if df is None or df.empty:
        raise ValueError(f"Không tìm thấy dữ liệu cho {ticker}")
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.columns = df.columns.str.capitalize()
    df = df.dropna(subset=['Close'])
    df = df[df.index.dayofweek < 5]
    df = df[df['Volume'] > 0]
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    return df

def fetch_vnindex_data(start_date, end_date, interval):
    from vnstock import Vnstock
    stock = Vnstock().stock(symbol="VNINDEX", source="VCI")
    df = stock.quote.history(start=start_date, end=end_date, interval=interval)
    if df is None or df.empty:
        raise ValueError("Không tìm thấy dữ liệu VN-Index")
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.columns = df.columns.str.capitalize()
    df = df.dropna(subset=['Close'])
    df = df[df.index.dayofweek < 5]
    df = df[df['Volume'] > 0]
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    return df

def run_analysis(df, interval):
    params = INTERVAL_WINDOWS.get(interval, INTERVAL_WINDOWS['1D'])
    ts = TimeSeriesProcessor()
    sa = StatisticalAnalyzer(window=params['stats'])
    mm = MarketMicrostructure(bins=60, val_area_pct=0.70)
    cla = CalculusLinearAlgebra()
    det = AnomalyDetector()
    ml = MachineLearningPredictor()

    df = ts.calculate_log_returns(df)
    df = ts.smooth_data(df, window=params['smooth'])
    df = sa.calculate_rolling_stats(df)
    vp_df = df.tail(params['vp_lookback'])
    vp, poc, vah, val = mm.calc_volume_profile(vp_df)
    df = cla.calculate_kinematics(df, col='Close_Smoothed')
    df = det.detect(df, vp_val=val, vp_poc=poc, vp_vah=vah)
    ml_acc, prob_up, ev = ml.train_and_predict(df)

    return df, vp, poc, vah, val, ml_acc, prob_up, ev

def build_plotly_figure(df, vp, poc, vah, val, ticker, interval):
    time_fmt = '%Y-%m-%d %H:%M' if interval != '1D' else '%Y-%m-%d'
    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.5, 0.25, 0.25],
        column_widths=[0.65, 0.35],
        specs=[[{"colspan": 2}, None],
               [{"type": "xy"}, {"type": "xy"}],
               [{"colspan": 2}, None]],
        subplot_titles=[
            f'{ticker} — Price Action & Anomalies ({interval})',
            'Volume Profile', 'Z-Score Distribution',
            'Z-Score Timeline'
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.06
    )

    # ---- Row 1: Price chart ----
    fig.add_trace(go.Scatter(
        x=df.index.strftime(time_fmt), y=df['Close'],
        mode='lines', name='Close', line=dict(color='#60a5fa', width=1.5)
    ), row=1, col=1)

    # Value Area fill
    fig.add_hrect(y0=val, y1=vah, fillcolor='rgba(34,197,94,0.08)',
                  line_width=0, row=1, col=1)
    fig.add_hline(y=poc, line_dash='dash', line_color='#f87171',
                  annotation_text=f'POC {poc:.1f}', row=1, col=1)
    fig.add_hline(y=vah, line_color='#4ade80', line_width=1,
                  annotation_text=f'VAH {vah:.1f}', row=1, col=1)
    fig.add_hline(y=val, line_color='#4ade80', line_width=1,
                  annotation_text=f'VAL {val:.1f}', row=1, col=1)

    # Signals
    for sig, color, sym, name in [
        ('Fake Nuke', '#fb923c', 'triangle-up', 'Fake Nuke'),
        ('Real Nuke', '#ef4444', 'triangle-down', 'Real Nuke'),
        ('Bullish Reversal', '#22c55e', 'triangle-up', 'Bullish Reversal'),
    ]:
        mask = df['Signal'] == sig
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df[mask].index.strftime(time_fmt),
                y=df[mask]['Close'],
                mode='markers', name=name,
                marker=dict(color=color, size=10, symbol=sym,
                            line=dict(width=1, color='white'))
            ), row=1, col=1)

    # ---- Row 2 left: Volume Profile ----
    if vp is not None and not vp.empty:
        fig.add_trace(go.Bar(
            y=vp['Price'], x=vp['Volume'],
            orientation='h', name='Volume Profile',
            marker_color='rgba(96,165,250,0.5)',
            showlegend=False
        ), row=2, col=1)
        fig.add_hline(y=poc, line_dash='dash', line_color='#f87171', row=2, col=1)
        fig.add_hline(y=vah, line_color='#4ade80', row=2, col=1)
        fig.add_hline(y=val, line_color='#4ade80', row=2, col=1)

    # ---- Row 2 right: Bell Curve ----
    returns = df['Log_Return'].dropna()
    mu, std_val = stats.norm.fit(returns)
    hist_vals, bin_edges = np.histogram(returns, bins=40, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=bin_centers, y=hist_vals, name='Actual Returns',
        marker_color='rgba(96,165,250,0.5)', showlegend=False
    ), row=2, col=2)
    x_norm = np.linspace(returns.min(), returns.max(), 100)
    fig.add_trace(go.Scatter(
        x=x_norm, y=stats.norm.pdf(x_norm, mu, std_val),
        mode='lines', name='Normal Curve',
        line=dict(color='#fbbf24', width=2), showlegend=False
    ), row=2, col=2)

    # ---- Row 3: Z-Score timeline ----
    fig.add_trace(go.Scatter(
        x=df.index.strftime(time_fmt), y=df['Z_Score'],
        mode='lines', name='Z-Score',
        line=dict(color='#a78bfa', width=1.2), showlegend=False
    ), row=3, col=1)
    fig.add_hline(y=-1.5, line_dash='dash', line_color='#ef4444',
                  annotation_text='Threshold -1.5', row=3, col=1)
    fig.add_hline(y=0, line_color='rgba(255,255,255,0.3)', row=3, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font=dict(family='Inter, sans-serif', color='#e2e8f0'),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        height=900,
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')

    return json.loads(fig.to_json())

def build_snapshot(df, poc, vah, val, ml_acc, prob_up, ev, ticker, interval):
    latest = df.iloc[-1]
    time_fmt = '%Y-%m-%d %H:%M' if interval != '1D' else '%Y-%m-%d'
    skew_total = df['Close'].skew()
    signal = latest['Signal']
    signal_class = 'fake' if 'Fake' in signal else ('real' if 'Real' in signal else ('bull' if 'Bullish' in signal else 'normal'))

    return {
        'ticker': ticker,
        'interval': interval,
        'signal': signal,
        'signal_class': signal_class,
        'time': df.index[-1].strftime(time_fmt),
        'total_candles': len(df),
        'mean_close': round(df['Close'].mean(), 2),
        'std_close': round(df['Close'].std(), 2),
        'skew': round(skew_total, 2),
        'close': round(latest['Close'], 2),
        'volume': int(latest['Volume']),
        'z_score': round(float(latest['Z_Score']), 2) if not pd.isna(latest['Z_Score']) else None,
        'p_value': round(float(latest['P_Value']), 4) if not pd.isna(latest['P_Value']) else None,
        'roll_skew': round(float(latest['Rolling_Skew']), 2) if not pd.isna(latest['Rolling_Skew']) else None,
        'rel_z_score': round(float(latest['Relative_Z_Score']), 2) if not pd.isna(latest['Relative_Z_Score']) else None,
        'acceleration': round(float(latest['Acceleration']), 4) if not pd.isna(latest['Acceleration']) else None,
        'poc': round(poc, 2), 'vah': round(vah, 2), 'val': round(val, 2),
        'ml_acc': round(ml_acc * 100, 1) if ml_acc else None,
        'prob_up': round(prob_up * 100, 2) if prob_up is not None else None,
        'ev': round(ev * 100, 4) if ev is not None else None,
        'events': [
            {'time': idx.strftime(time_fmt), 'signal': row['Signal'], 'return': round(row['Log_Return'] * 100, 2)}
            for idx, row in df[df['Signal'] != 'Normal'].iterrows()
        ]
    }

# ==========================================
# ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    mode = data.get('mode', 'stock')  # 'stock' or 'vnindex'
    ticker = data.get('ticker', 'HPG').strip().upper()
    start_date = data.get('start_date', '2026-01-01')
    end_date = data.get('end_date', datetime.today().strftime('%Y-%m-%d'))
    interval = data.get('interval', '1D')

    try:
        if mode == 'vnindex':
            df = fetch_vnindex_data(start_date, end_date, interval)
            ticker = 'VNINDEX'
        else:
            df = fetch_stock_data(ticker, start_date, end_date, interval)

        df, vp, poc, vah, val, ml_acc, prob_up, ev = run_analysis(df, interval)
        snapshot = build_snapshot(df, poc, vah, val, ml_acc, prob_up, ev, ticker, interval)
        chart = build_plotly_figure(df, vp, poc, vah, val, ticker, interval)
        return jsonify({'success': True, 'snapshot': snapshot, 'chart': chart})
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
