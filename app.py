import sys
sys.stdout.reconfigure(encoding='utf-8')

from datetime import datetime
import json
import logging
import os
import pickle
import types
import time
import sys
import urllib.error
import urllib.request

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import scipy.stats as stats


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('WebQuantModel')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)


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
        if df.empty:
            return None, 0, 0, 0
        min_price, max_price = df[price_col].min(), df[price_col].max()
        if min_price == max_price:
            return None, min_price, min_price, min_price

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


class CalculusLinearAlgebra:
    @staticmethod
    def calculate_kinematics(df, col='Close_Smoothed'):
        df = df.copy()
        if col not in df.columns:
            return df
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
            prev = df.iloc[i - 1]
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
                if curr['Log_Return'] > 0 and curr['Close'] > vp_poc:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'Bullish Reversal'
                continue

            # Bearish Reversal conditions
            overbought_z = (curr['Z_Score'] < 1.0) and (prev['Z_Score'] > 1.5)
            strong_decel = (curr['Acceleration'] < 0) and (curr['Acceleration'] < prev['Acceleration'])
            price_breaking_val = (curr['Close'] < vp_val) and (prev['Close'] >= vp_val)
            price_breaking_poc = (curr['Close'] < vp_poc) and (prev['Close'] >= vp_poc)
            if (overbought_z and strong_decel) or (price_breaking_val and strong_decel) or (price_breaking_poc and curr['Z_Score'] < 0):
                if curr['Log_Return'] < 0 and curr['Close'] < vp_poc:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'Bearish Reversal'
        return df


class MachineLearningPredictor:
    def train_and_predict(self, df):
        features = ['Log_Return', 'Velocity', 'Acceleration', 'Z_Score', 'Rolling_Skew', 'Volume']
        df_ml = df.dropna(subset=features).copy()
        if len(df_ml) < 50:
            return None, None, None

        latest = df_ml.iloc[-1]
        recent = df_ml.tail(120).copy()
        next_up = (recent['Close'].shift(-1) > recent['Close']).astype(float)
        aligned = recent.iloc[:-1].copy()
        target = next_up.iloc[:-1]
        if len(aligned) < 20:
            return None, None, None

        signal_score = (
            np.tanh(aligned['Acceleration'].fillna(0) * 8.0) * 0.30
            + np.tanh((-aligned['Z_Score'].fillna(0)) / 2.0) * 0.25
            + np.tanh((-aligned['Rolling_Skew'].fillna(0)) / 2.0) * 0.15
            + np.tanh(aligned['Log_Return'].fillna(0) * 12.0) * 0.15
            + np.tanh(aligned['Velocity'].fillna(0) * 8.0) * 0.15
        )
        pred = (signal_score > 0).astype(int)
        acc = float((pred.values == target.values).mean())

        latest_score = (
            np.tanh(float(latest['Acceleration']) * 8.0) * 0.30
            + np.tanh(float(-latest['Z_Score']) / 2.0) * 0.25
            + np.tanh(float(-latest['Rolling_Skew']) / 2.0) * 0.15
            + np.tanh(float(latest['Log_Return']) * 12.0) * 0.15
            + np.tanh(float(latest['Velocity']) * 8.0) * 0.15
        )
        prob_up = float(np.clip(50 + latest_score * 35, 5, 95)) / 100.0
        recent_returns = df['Log_Return'].tail(120)
        avg_gain = recent_returns[recent_returns > 0].mean()
        avg_loss = recent_returns[recent_returns < 0].mean()
        avg_gain = avg_gain if not pd.isna(avg_gain) else 0.01
        avg_loss = avg_loss if not pd.isna(avg_loss) else -0.01
        ev = (prob_up * avg_gain) + ((1 - prob_up) * avg_loss)
        return acc, prob_up, ev


INTERVAL_WINDOWS = {
    '1D': {'stats': 20, 'smooth': 5, 'vp_lookback': 60},
    '1H': {'stats': 24, 'smooth': 6, 'vp_lookback': 120},
    '15m': {'stats': 30, 'smooth': 8, 'vp_lookback': 160},
}
MAX_FETCH_ROWS = {
    '1D': 400,
    '1H': 800,
    '15m': 1200,
}
CACHE_DIR = '.cache_vnstock'
CACHE_TTL_SECONDS = 900
GITHUB_MODELS_ENDPOINT = 'https://models.github.ai/inference/chat/completions'
GITHUB_MODELS_API_VERSION = '2022-11-28'
GITHUB_MODELS_NAME = os.getenv('GITHUB_MODELS_NAME', 'openai/gpt-4o-mini')


def patch_vnstock_visual_modules():
    if 'vnstock.common.viz' not in sys.modules:
        dummy_viz = types.ModuleType('vnstock.common.viz')

        class DummyChart:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        dummy_viz.Chart = DummyChart
        sys.modules['vnstock.common.viz'] = dummy_viz

    if 'vnstock_ezchart' not in sys.modules:
        sys.modules['vnstock_ezchart'] = types.ModuleType('vnstock_ezchart')

    if 'vnstock_chart' not in sys.modules:
        sys.modules['vnstock_chart'] = types.ModuleType('vnstock_chart')


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def build_cache_path(symbol, start_date, end_date, interval, source):
    safe_symbol = symbol.replace('/', '_')
    return os.path.join(CACHE_DIR, f'{safe_symbol}_{source}_{interval}_{start_date}_{end_date}.pkl')


def load_cached_frame(symbol, start_date, end_date, interval, source):
    ensure_cache_dir()
    cache_path = build_cache_path(symbol, start_date, end_date, interval, source)
    if not os.path.exists(cache_path):
        return None
    age = time.time() - os.path.getmtime(cache_path)
    if age > CACHE_TTL_SECONDS:
        return None
    try:
        with open(cache_path, 'rb') as cache_file:
            df = pickle.load(cache_file)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        return None
    return None


def save_cached_frame(df, symbol, start_date, end_date, interval, source):
    ensure_cache_dir()
    cache_path = build_cache_path(symbol, start_date, end_date, interval, source)
    try:
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(df, cache_file)
    except Exception:
        logger.warning('Unable to write cache file %s', cache_path)


def optimize_dataframe(df, interval):
    limit = MAX_FETCH_ROWS.get(interval, 400)
    df = df.tail(limit).copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].astype('float32')
    return df


def normalize_quote_frame(df, interval):
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.columns = df.columns.str.capitalize()
    df = df.dropna(subset=['Close'])
    df = df[df.index.dayofweek < 5]
    if 'Volume' in df.columns:
        df = df[df['Volume'] > 0]
    df = df[~df.index.duplicated(keep='first')]
    return optimize_dataframe(df.sort_index(), interval)


def fetch_stock_data(ticker, start_date, end_date, interval):
    patch_vnstock_visual_modules()
    from vnstock import Quote

    errors = []
    for source in ['kbs', 'vci']:
        cached_df = load_cached_frame(ticker, start_date, end_date, interval, source.lower())
        if cached_df is not None:
            logger.info('Loaded %s %s from cache (%s)', ticker, interval, source)
            return normalize_quote_frame(cached_df.copy(), interval)

        try:
            quote = Quote(source=source, symbol=ticker, show_log=False)
            df = quote.history(start=start_date, end=end_date, interval=interval)
            if df is None or df.empty:
                raise ValueError('empty dataframe')
            save_cached_frame(df, ticker, start_date, end_date, interval, source.lower())
            logger.info('Fetched %s %s from vnstock quote source=%s rows=%s', ticker, interval, source, len(df))
            return normalize_quote_frame(df, interval)
        except Exception as exc:
            logger.warning('vnstock quote fetch failed for %s source=%s interval=%s: %s', ticker, source, interval, exc)
            errors.append(f'{source}: {exc}')
            time.sleep(0.6)

    raise ValueError(f'Khong the lay du lieu {ticker}. Sources tried: {" | ".join(errors)}')


def fetch_vnindex_data(start_date, end_date, interval):
    patch_vnstock_visual_modules()
    from vnstock import Quote

    errors = []
    for source in ['vci', 'kbs']:
        cached_df = load_cached_frame('VNINDEX', start_date, end_date, interval, source.lower())
        if cached_df is not None:
            logger.info('Loaded VNINDEX %s from cache (%s)', interval, source)
            return normalize_quote_frame(cached_df.copy(), interval)

        try:
            quote = Quote(source=source, symbol='VNINDEX', show_log=False)
            df = quote.history(start=start_date, end=end_date, interval=interval)
            if df is None or df.empty:
                raise ValueError('empty dataframe')
            save_cached_frame(df, 'VNINDEX', start_date, end_date, interval, source.lower())
            logger.info('Fetched VNINDEX %s from vnstock quote source=%s rows=%s', interval, source, len(df))
            return normalize_quote_frame(df, interval)
        except Exception as exc:
            logger.warning('vnstock quote fetch failed for VNINDEX source=%s interval=%s: %s', source, interval, exc)
            errors.append(f'{source}: {exc}')
            time.sleep(0.6)

    raise ValueError(f'Khong the lay du lieu VNINDEX. Sources tried: {" | ".join(errors)}')


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


def build_plotly_figure(df, poc, vah, val, ticker, interval):
    time_fmt = '%Y-%m-%d %H:%M' if interval != '1D' else '%Y-%m-%d'
    x_values = df.index.strftime(time_fmt).tolist()
    volume_colors = np.where(df['Close'] >= df['Open'], '#22c55e', '#ef4444').tolist()
    accel_values = df['Acceleration'].astype(float).fillna(0).tolist()
    accel_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in accel_values]
    skew_values = df['Rolling_Skew'].astype(float).fillna(0).tolist()
    traces = [
        {
            'type': 'candlestick',
            'x': x_values,
            'open': df['Open'].astype(float).tolist(),
            'high': df['High'].astype(float).tolist(),
            'low': df['Low'].astype(float).tolist(),
            'close': df['Close'].astype(float).tolist(),
            'name': 'Candles',
            'xaxis': 'x',
            'yaxis': 'y',
            'increasing': {'line': {'color': '#22c55e'}, 'fillcolor': '#22c55e'},
            'decreasing': {'line': {'color': '#ef4444'}, 'fillcolor': '#ef4444'},
        },
        {
            'type': 'scatter',
            'x': x_values,
            'y': df['Close_Smoothed'].astype(float).bfill().ffill().tolist(),
            'mode': 'lines',
            'name': 'Smoothed',
            'xaxis': 'x',
            'yaxis': 'y',
            'line': {'color': '#f8fafc', 'width': 1.2, 'dash': 'dot'},
        },
        {
            'type': 'bar',
            'x': x_values,
            'y': df['Volume'].astype(float).tolist(),
            'name': 'Volume',
            'xaxis': 'x2',
            'yaxis': 'y2',
            'marker': {'color': volume_colors},
            'opacity': 0.75,
        },
        {
            'type': 'scatter',
            'x': x_values,
            'y': df['Z_Score'].astype(float).fillna(0).tolist(),
            'mode': 'lines',
            'name': 'Z-Score',
            'xaxis': 'x3',
            'yaxis': 'y3',
            'line': {'color': '#a78bfa', 'width': 1.6},
            'fill': 'tozeroy',
            'fillcolor': 'rgba(167,139,250,0.10)',
        },
        {
            'type': 'bar',
            'x': x_values,
            'y': accel_values,
            'name': 'Acceleration',
            'xaxis': 'x4',
            'yaxis': 'y4',
            'marker': {'color': accel_colors},
            'opacity': 0.8,
        },
        {
            'type': 'scatter',
            'x': x_values,
            'y': skew_values,
            'mode': 'lines',
            'name': 'Rolling Skew',
            'xaxis': 'x5',
            'yaxis': 'y5',
            'line': {'color': '#f59e0b', 'width': 1.5},
            'fill': 'tozeroy',
            'fillcolor': 'rgba(245,158,11,0.08)',
        },
    ]

    for sig, color, sym, name in [
        ('Fake Nuke', '#fb923c', 'triangle-up', 'Fake Nuke'),
        ('Real Nuke', '#ef4444', 'triangle-down', 'Real Nuke'),
        ('Bullish Reversal', '#22c55e', 'triangle-up', 'Bullish Reversal'),
        ('Bearish Reversal', '#ef4444', 'triangle-down', 'Bearish Reversal'),
    ]:
        mask = df['Signal'] == sig
        if mask.any():
            traces.append({
                'type': 'scatter',
                'x': df[mask].index.strftime(time_fmt).tolist(),
                'y': df[mask]['Close'].astype(float).tolist(),
                'mode': 'markers',
                'name': name,
                'xaxis': 'x',
                'yaxis': 'y',
                'marker': {'color': color, 'size': 10, 'symbol': sym, 'line': {'width': 1, 'color': 'white'}},
            })

    layout = {
        'paper_bgcolor': '#0f172a',
        'plot_bgcolor': '#172033',
        'font': {'family': 'Inter, sans-serif', 'color': '#e2e8f0'},
        'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0},
        'margin': {'l': 30, 'r': 20, 't': 70, 'b': 30},
        'height': 1080,
        'showlegend': True,
        'xaxis': {'domain': [0, 1], 'anchor': 'y', 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.05)', 'rangeslider': {'visible': False}},
        'yaxis': {'domain': [0.56, 1.0], 'anchor': 'x', 'title': {'text': 'Price'}, 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.05)'},
        'xaxis2': {'domain': [0, 1], 'anchor': 'y2', 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.05)'},
        'yaxis2': {'domain': [0.43, 0.53], 'anchor': 'x2', 'title': {'text': 'Volume'}, 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.05)'},
        'xaxis3': {'domain': [0, 1], 'anchor': 'y3', 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.05)'},
        'yaxis3': {'domain': [0.28, 0.40], 'anchor': 'x3', 'title': {'text': 'Z-Score'}, 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.05)'},
        'xaxis4': {'domain': [0, 1], 'anchor': 'y4', 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.05)'},
        'yaxis4': {'domain': [0.14, 0.25], 'anchor': 'x4', 'title': {'text': 'Acceleration'}, 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.05)'},
        'xaxis5': {'domain': [0, 1], 'anchor': 'y5', 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.05)'},
        'yaxis5': {'domain': [0.0, 0.11], 'anchor': 'x5', 'title': {'text': 'Skew'}, 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.05)'},
        'annotations': [
            {'text': f'{ticker} - Candlestick & Signals ({interval})', 'xref': 'paper', 'yref': 'paper', 'x': 0, 'y': 1.08, 'showarrow': False, 'font': {'size': 16}},
            {'text': 'Volume', 'xref': 'paper', 'yref': 'paper', 'x': 0, 'y': 0.55, 'showarrow': False, 'font': {'size': 12}},
            {'text': 'Z-Score Monitor', 'xref': 'paper', 'yref': 'paper', 'x': 0, 'y': 0.41, 'showarrow': False, 'font': {'size': 12}},
            {'text': 'Acceleration', 'xref': 'paper', 'yref': 'paper', 'x': 0, 'y': 0.26, 'showarrow': False, 'font': {'size': 12}},
            {'text': 'Rolling Skew', 'xref': 'paper', 'yref': 'paper', 'x': 0, 'y': 0.12, 'showarrow': False, 'font': {'size': 12}},
            {'text': f'POC {poc:.2f}', 'xref': 'paper', 'yref': 'y', 'x': 1, 'y': float(poc), 'showarrow': False, 'xanchor': 'right', 'font': {'size': 11, 'color': '#f87171'}},
            {'text': f'VAH {vah:.2f}', 'xref': 'paper', 'yref': 'y', 'x': 1, 'y': float(vah), 'showarrow': False, 'xanchor': 'right', 'font': {'size': 11, 'color': '#4ade80'}},
            {'text': f'VAL {val:.2f}', 'xref': 'paper', 'yref': 'y', 'x': 1, 'y': float(val), 'showarrow': False, 'xanchor': 'right', 'font': {'size': 11, 'color': '#4ade80'}},
        ],
        'shapes': [
            {'type': 'rect', 'xref': 'x', 'yref': 'y', 'x0': x_values[0], 'x1': x_values[-1], 'y0': float(val), 'y1': float(vah), 'fillcolor': 'rgba(34,197,94,0.10)', 'line': {'width': 0}},
            {'type': 'line', 'xref': 'x', 'yref': 'y', 'x0': x_values[0], 'x1': x_values[-1], 'y0': float(poc), 'y1': float(poc), 'line': {'color': '#f87171', 'width': 1, 'dash': 'dash'}},
            {'type': 'line', 'xref': 'x', 'yref': 'y', 'x0': x_values[0], 'x1': x_values[-1], 'y0': float(vah), 'y1': float(vah), 'line': {'color': '#4ade80', 'width': 1}},
            {'type': 'line', 'xref': 'x', 'yref': 'y', 'x0': x_values[0], 'x1': x_values[-1], 'y0': float(val), 'y1': float(val), 'line': {'color': '#4ade80', 'width': 1}},
            {'type': 'line', 'xref': 'x3', 'yref': 'y3', 'x0': x_values[0], 'x1': x_values[-1], 'y0': -1.5, 'y1': -1.5, 'line': {'color': '#ef4444', 'width': 1, 'dash': 'dash'}},
            {'type': 'line', 'xref': 'x3', 'yref': 'y3', 'x0': x_values[0], 'x1': x_values[-1], 'y0': 0, 'y1': 0, 'line': {'color': 'rgba(255,255,255,0.3)', 'width': 1}},
            {'type': 'line', 'xref': 'x4', 'yref': 'y4', 'x0': x_values[0], 'x1': x_values[-1], 'y0': 0, 'y1': 0, 'line': {'color': 'rgba(255,255,255,0.3)', 'width': 1}},
            {'type': 'line', 'xref': 'x5', 'yref': 'y5', 'x0': x_values[0], 'x1': x_values[-1], 'y0': 0, 'y1': 0, 'line': {'color': 'rgba(255,255,255,0.3)', 'width': 1}},
        ],
    }

    return {'data': traces, 'layout': layout}


def to_py_float(value, digits=None):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return round(value, digits) if digits is not None else value


def build_snapshot(df, poc, vah, val, ml_acc, prob_up, ev, ticker, interval):
    latest = df.iloc[-1]
    time_fmt = '%Y-%m-%d %H:%M' if interval != '1D' else '%Y-%m-%d'
    signal = latest['Signal']
    signal_class = 'fake' if 'Fake' in signal else ('real' if 'Real' in signal else ('bull' if 'Bullish' in signal else 'normal'))

    def future_signal_return(event_index, periods=3):
        if event_index >= len(df) - 1:
            return None
        target_index = min(event_index + periods, len(df) - 1)
        entry_price = df.iloc[event_index]['Close']
        exit_price = df.iloc[target_index]['Close']
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
            return None
        return to_py_float((exit_price / entry_price - 1) * 100, 2)

    events = []
    for idx, row in df[df['Signal'] != 'Normal'].iterrows():
        event_index = df.index.get_loc(idx)
        events.append({
            'time': idx.strftime(time_fmt),
            'signal': row['Signal'],
            'return': future_signal_return(event_index, periods=3),
        })

    return {
        'ticker': ticker,
        'interval': interval,
        'signal': signal,
        'signal_class': signal_class,
        'time': df.index[-1].strftime(time_fmt),
        'total_candles': len(df),
        'mean_close': to_py_float(df['Close'].mean(), 2),
        'std_close': to_py_float(df['Close'].std(), 2),
        'mean_log_return': to_py_float(df['Log_Return'].mean(), 4),
        'std_log_return': to_py_float(df['Log_Return'].std(), 4),
        'skew': to_py_float(df['Close'].skew(), 2),
        'close': to_py_float(latest['Close'], 2),
        'volume': int(latest['Volume']),
        'z_score': to_py_float(latest['Z_Score'], 2),
        'p_value': to_py_float(latest['P_Value'], 4),
        'roll_skew': to_py_float(latest['Rolling_Skew'], 2),
        'rel_z_score': to_py_float(latest['Relative_Z_Score'], 2),
        'acceleration': to_py_float(latest['Acceleration'], 4),
        'poc': to_py_float(poc, 2),
        'vah': to_py_float(vah, 2),
        'val': to_py_float(val, 2),
        'ml_acc': to_py_float(ml_acc * 100, 1) if ml_acc is not None else None,
        'prob_up': to_py_float(prob_up * 100, 2) if prob_up is not None else None,
        'ev': to_py_float(ev * 100, 4) if ev is not None else None,
        'events': events,
    }


def build_fallback_ai_analysis(snapshot):
    close = snapshot['close']
    vah = snapshot['vah']
    val = snapshot['val']
    poc = snapshot['poc']
    mean_close = snapshot['mean_close']
    std_close = snapshot['std_close']
    z_score = snapshot['z_score'] if snapshot['z_score'] is not None else 0
    mean_log_return = snapshot['mean_log_return']
    std_log_return = snapshot['std_log_return']
    roll_skew = snapshot['roll_skew'] if snapshot['roll_skew'] is not None else 0
    acceleration = snapshot['acceleration'] if snapshot['acceleration'] is not None else 0
    prob_up = snapshot['prob_up']
    ev = snapshot['ev']
    p_value = snapshot['p_value']
    skew = snapshot['skew']

    # Phần 1: Vị thế giá so với cấu trúc thị trường
    position_analysis = f"""1. Vị thế giá so với cấu trúc thị trường (POC & Value Area)
Close hiện tại = {close}

POC (Điểm kiểm soát) = {poc} → giá đang {'trên' if close > poc else 'dưới' if close < poc else 'tại'} POC.

Value Area High = {vah}, Value Area Low = {val} → giá đang nằm {'trên biên trên' if close > vah else 'trong vùng giá trị' if val <= close <= vah else 'sát biên dưới' if close >= val - (vah - val) * 0.1 else 'dưới biên dưới'} của vùng giá trị.

Mean(Close) = {mean_close}, StdDev(Close) = {std_close} → giá hiện tại {'cao hơn' if close > mean_close else 'thấp hơn'} trung bình khoảng {abs((close - mean_close) / std_close):.1f} độ lệch chuẩn. Vị thế này cho thấy giá đang {'chiết khấu về vùng hỗ trợ quan trọng' if close < mean_close else 'trong vùng kháng cự mạnh'}."""

    # Phần 2: Các chỉ báo động lượng & phân phối lợi suất
    momentum_analysis = f"""2. Các chỉ báo động lượng & phân phối lợi suất
Z-Score = {z_score} (dựa trên Log Return với μ={mean_log_return}, σ={std_log_return}): Lợi suất gần đây {'dương' if z_score > 0 else 'âm'} gần {abs(z_score):.1f} độ lệch chuẩn, phản ánh {'áp lực mua' if z_score > 0 else 'áp lực bán'} ngắn hạn, {'chưa đến ngưỡng cực đoan' if abs(z_score) < 2 else 'đã đến ngưỡng cực đoan'} (thường >2 hoặc <-2 mới là quá mua/bán mạnh).

Skew = {skew} ({'lệch phải' if skew > 0 else 'lệch trái'} tổng thể): Phân phối lợi suất dài hạn vẫn {'lệch phải' if skew > 0 else 'lệch trái'}, {'đuôi phải dày hơn' if skew > 0 else 'đuôi trái dày hơn'}, tức là cổ phiếu có xu hướng xuất hiện các nhịp {'tăng mạnh' if skew > 0 else 'giảm mạnh'} khi hồi phục.

Roll Skew = {roll_skew} (độ lệch cuộn ngắn hạn): {'Khá cao' if roll_skew > 1 else 'Thấp'}, cho thấy giai đoạn gần đây có những phiên {'tăng đột biến' if roll_skew > 0 else 'giảm đột biến'} (lợi suất {'dương' if roll_skew > 0 else 'âm'} lớn). Điều này cảnh báo rằng thị trường đã có những nhịp {'hưng phấn' if roll_skew > 0 else 'bán tháo'} cục bộ, và cần thời gian điều chỉnh hoặc tích lũy trở lại.

Acceleration (Gia tốc) = {acceleration}: Gia tốc {'dương' if acceleration > 0 else 'âm'}, {'đang cải thiện' if acceleration > 0 else 'vẫn suy yếu'}. Đây là tín hiệu cho thấy đà {'tăng' if acceleration > 0 else 'giảm'} đã {'bứt phá' if abs(acceleration) > 0.01 else 'chững lại'}, động lượng đang {'cải thiện dần' if acceleration >= 0 else 'suy yếu'}."""

    # Phần 3: Xác suất mô hình và kỳ vọng
    model_analysis = f"""3. Xác suất mô hình và kỳ vọng
ML Prob Up = {prob_up}%: Mô hình học máy đưa ra xác suất tăng giá ~{prob_up}%, đây là mức {'khá tích cực' if prob_up > 60 else 'trung lập' if 40 <= prob_up <= 60 else 'tiêu cực'}, cho thấy dữ liệu lịch sử và các yếu tố định lượng {'ủng hộ khả năng tăng' if prob_up > 50 else 'ủng hộ khả năng giảm'} trong những phiên tới.

Exp Value (EV) = {ev}%: Kỳ vọng lợi nhuận trung bình hàng ngày là {ev}%, {'dương' if ev > 0 else 'âm'}, phù hợp với thị trường đang trong giai đoạn {'hồi phục' if ev > 0 else 'suy yếu'}.

P-Value = {p_value} (từ kiểm định bất thường): {'<' if p_value < 0.05 else '>'}0.05, {'có bất thường có ý nghĩa thống kê cần lo ngại' if p_value < 0.05 else 'không có bất thường có ý nghĩa thống kê cần lo ngại'}."""

    # Kết luận
    conclusion = f"Kết luận: Dựa trên phân tích trên, khuyến nghị {'MUA' if prob_up > 60 and close > poc else 'BÁN' if prob_up < 40 and close < poc else 'GIỮ'} với độ tin cậy {72 if prob_up > 60 or prob_up < 40 else 58}%. Thị trường đang {'trong giai đoạn tích cực' if prob_up > 55 else 'trung lập' if 45 <= prob_up <= 55 else 'tiêu cực'}, cần theo dõi sát sao các ngưỡng POC và VAL."

    summary = f"{position_analysis}\n\n{momentum_analysis}\n\n{model_analysis}\n\n{conclusion}"

    risks = []
    if z_score < -1.5:
        risks.append('Z-Score quá bán, có thể hồi phục.')
    if roll_skew < 0:
        risks.append('Rolling skew âm, xu hướng giảm ngắn hạn.')
    if acceleration < 0:
        risks.append('Gia tốc âm, động lượng suy yếu.')
    if p_value < 0.05:
        risks.append('Có bất thường thống kê, cần cẩn trọng.')
    if not risks:
        risks.append('Không có rủi ro nổi bật hiện tại.')

    action = 'BUY' if prob_up > 60 and close >= poc else 'SELL' if prob_up < 40 and close < poc else 'HOLD'
    confidence = 72 if action != 'HOLD' else 58

    return {
        'title': f'AI Analysis - {snapshot["ticker"]}',
        'summary': summary,
        'bullets': risks[:3],
        'indicators': [
            {'name': 'Close vs Value Area', 'status': 'positive' if close > poc else 'negative', 'analysis': f'Giá tại {close}, POC {poc}, VAH {vah}, VAL {val}.'},
            {'name': 'Z-Score', 'status': 'positive' if z_score < 0 else 'neutral', 'analysis': f'Z-Score {z_score}, phản ánh động lượng.'},
            {'name': 'Rolling Skew', 'status': 'negative' if roll_skew < 0 else 'positive', 'analysis': f'Roll Skew {roll_skew}, phân phối ngắn hạn.'},
            {'name': 'Acceleration', 'status': 'positive' if acceleration > 0 else 'negative', 'analysis': f'Acceleration {acceleration}, gia tốc giá.'},
            {'name': 'Prob Up / EV', 'status': 'positive' if prob_up > 55 else 'negative', 'analysis': f'Xác suất tăng {prob_up}%, EV {ev}%. P-Value {p_value}.'},
        ],
        'recommendation': {
            'action': action,
            'confidence': confidence,
            'reason': f'Dựa trên xác suất mô hình {prob_up}% và vị thế giá.',
            'disclaimer': 'Đây chỉ là góc nhìn định lượng từ AI, không phải khuyến nghị đầu tư bắt buộc.'
        }
    }


def get_github_models_token():
    return os.getenv('GITHUB_MODELS_TOKEN') or os.getenv('GITHUB_PAT')


def build_indicator_prompt(snapshot):
    return json.dumps({
        'ticker': snapshot['ticker'],
        'interval': snapshot['interval'],
        'signal': snapshot['signal'],
        'close': snapshot['close'],
        'poc': snapshot['poc'],
        'vah': snapshot['vah'],
        'val': snapshot['val'],
        'z_score': snapshot['z_score'],
        'p_value': snapshot['p_value'],
        'roll_skew': snapshot['roll_skew'],
        'rel_z_score': snapshot['rel_z_score'],
        'acceleration': snapshot['acceleration'],
        'prob_up': snapshot['prob_up'],
        'ev': snapshot['ev'],
        'ml_acc': snapshot['ml_acc'],
        'mean_close': snapshot['mean_close'],
        'std_close': snapshot['std_close'],
        'skew': snapshot['skew'],
    }, ensure_ascii=False)


def build_ai_analysis(snapshot):
    token = get_github_models_token()
    fallback = build_fallback_ai_analysis(snapshot)
    if not token:
        return fallback

    system_prompt = (
        'You are a cautious quantitative market analyst. '
        'Analyze each indicator separately, then return a concise JSON object only. '
        'Never promise profit. Recommendation action must be one of BUY, SELL, HOLD.'
    )
    user_prompt = (
        'Analyze the following market snapshot. '
        'Return JSON with keys: title, summary, bullets, indicators, recommendation. '
        'indicators must be an array of objects with keys name, status, analysis. '
        'recommendation must include action, confidence, reason, disclaimer. '
        'Keep text concise and practical.\n\n'
        f'{build_indicator_prompt(snapshot)}'
    )
    payload = {
        'model': GITHUB_MODELS_NAME,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': 0.2,
        'max_tokens': 700,
        'response_format': {'type': 'json_object'},
    }

    request = urllib.request.Request(
        GITHUB_MODELS_ENDPOINT,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Accept': 'application/vnd.github+json',
            'Authorization': f'Bearer {token}',
            'X-GitHub-Api-Version': GITHUB_MODELS_API_VERSION,
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = json.loads(response.read().decode('utf-8'))
        content = body['choices'][0]['message']['content']
        parsed = json.loads(content)
        parsed.setdefault('title', fallback['title'])
        parsed.setdefault('summary', fallback['summary'])
        parsed.setdefault('bullets', fallback['bullets'])
        parsed.setdefault('indicators', fallback['indicators'])
        parsed.setdefault('recommendation', fallback['recommendation'])
        return parsed
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        logger.warning('GitHub Models analysis failed: %s', exc)
        fallback['bullets'] = [f'GitHub Models API fallback: {exc}'] + fallback['bullets'][:2]
        return fallback


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    mode = data.get('mode', 'stock')
    ticker = data.get('ticker', 'HPG').strip().upper()
    start_date = data.get('start_date', '2026-01-01')
    end_date = data.get('end_date', datetime.today().strftime('%Y-%m-%d'))
    interval = data.get('interval', '1D')

    try:
        if mode == 'vnindex':
            raw_df = fetch_vnindex_data(start_date, end_date, interval)
            ticker = 'VNINDEX'
        else:
            raw_df = fetch_stock_data(ticker, start_date, end_date, interval)

        df, vp, poc, vah, val, ml_acc, prob_up, ev = run_analysis(raw_df, interval)
        snapshot = build_snapshot(df, poc, vah, val, ml_acc, prob_up, ev, ticker, interval)
        chart = build_plotly_figure(df, poc, vah, val, ticker, interval)
        ai_analysis = build_ai_analysis(snapshot)
        output = (
            f'Analysis completed for {ticker} ({interval}).\n'
            f'Candle count: {len(df)}\n'
            f'Latest signal: {snapshot["signal"]}\n'
            f'POC/VAH/VAL: {snapshot["poc"]} / {snapshot["vah"]} / {snapshot["val"]}\n'
            f'Model mode: lightweight heuristic'
        )
        return jsonify({
            'success': True,
            'snapshot': snapshot,
            'chart': chart,
            'ai_analysis': ai_analysis,
            'output': output,
        })
    except Exception as e:
        logger.error(f'Analysis error: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error('Unhandled exception', exc_info=True)
    return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
