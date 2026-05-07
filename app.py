import sys
sys.stdout.reconfigure(encoding='utf-8')

from datetime import datetime
import json
import logging
import os
import pickle
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('WebQuantModel')

app = Flask(__name__)
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
                if curr['Log_Return'] > 0:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'Bullish Reversal'
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
    x_values = df.index.strftime(time_fmt)
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.62, 0.16, 0.22],
        specs=[[{'type': 'xy'}], [{'type': 'xy'}], [{'type': 'xy'}]],
        subplot_titles=[
            f'{ticker} - Candlestick & Signals ({interval})',
            'Volume',
            'Z-Score Monitor'
        ],
        vertical_spacing=0.06,
    )

    fig.add_trace(
        go.Candlestick(
            x=x_values,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candles',
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444',
            increasing_fillcolor='#22c55e',
            decreasing_fillcolor='#ef4444',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=df['Close_Smoothed'],
            mode='lines',
            name='Smoothed',
            line=dict(color='#f8fafc', width=1.2, dash='dot'),
        ),
        row=1,
        col=1,
    )
    volume_colors = np.where(df['Close'] >= df['Open'], '#22c55e', '#ef4444')
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=df['Volume'],
            name='Volume',
            marker_color=volume_colors,
            opacity=0.75,
        ),
        row=2,
        col=1,
    )
    fig.add_hrect(y0=val, y1=vah, fillcolor='rgba(34,197,94,0.10)', line_width=0, row=1, col=1)
    fig.add_hline(y=poc, line_dash='dash', line_color='#f87171', annotation_text=f'POC {poc:.2f}', row=1, col=1)
    fig.add_hline(y=vah, line_color='#4ade80', line_width=1, annotation_text=f'VAH {vah:.2f}', row=1, col=1)
    fig.add_hline(y=val, line_color='#4ade80', line_width=1, annotation_text=f'VAL {val:.2f}', row=1, col=1)

    for sig, color, sym, name in [
        ('Fake Nuke', '#fb923c', 'triangle-up', 'Fake Nuke'),
        ('Real Nuke', '#ef4444', 'triangle-down', 'Real Nuke'),
        ('Bullish Reversal', '#22c55e', 'triangle-up', 'Bullish Reversal'),
    ]:
        mask = df['Signal'] == sig
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df[mask].index.strftime(time_fmt),
                    y=df[mask]['Close'],
                    mode='markers',
                    name=name,
                    marker=dict(color=color, size=10, symbol=sym, line=dict(width=1, color='white')),
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=df['Z_Score'],
            mode='lines',
            name='Z-Score',
            line=dict(color='#a78bfa', width=1.6),
            fill='tozeroy',
            fillcolor='rgba(167,139,250,0.10)',
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=-1.5, line_dash='dash', line_color='#ef4444', annotation_text='Risk threshold', row=3, col=1)
    fig.add_hline(y=0, line_color='rgba(255,255,255,0.3)', row=3, col=1)

    fig.update_layout(
        paper_bgcolor='#0f172a',
        plot_bgcolor='#172033',
        font=dict(family='Inter, sans-serif', color='#e2e8f0'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=30, r=20, t=70, b=30),
        height=820,
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickangle=0)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_yaxes(title_text='Z-Score', row=3, col=1)
    return json.loads(fig.to_json())


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

    return {
        'ticker': ticker,
        'interval': interval,
        'signal': signal,
        'signal_class': signal_class,
        'time': df.index[-1].strftime(time_fmt),
        'total_candles': len(df),
        'mean_close': to_py_float(df['Close'].mean(), 2),
        'std_close': to_py_float(df['Close'].std(), 2),
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
        'events': [
            {'time': idx.strftime(time_fmt), 'signal': row['Signal'], 'return': to_py_float(row['Log_Return'] * 100, 2)}
            for idx, row in df[df['Signal'] != 'Normal'].iterrows()
        ],
    }


def build_ai_analysis(snapshot):
    close = snapshot['close']
    vah = snapshot['vah']
    val = snapshot['val']
    z_score = snapshot['z_score'] if snapshot['z_score'] is not None else 0
    rel_z_score = snapshot['rel_z_score'] if snapshot['rel_z_score'] is not None else 0
    roll_skew = snapshot['roll_skew'] if snapshot['roll_skew'] is not None else 0
    acceleration = snapshot['acceleration'] if snapshot['acceleration'] is not None else 0
    prob_up = snapshot['prob_up']
    ev = snapshot['ev']
    signal = snapshot['signal']

    if close > vah:
        location = 'Gia dang nam tren value area, dong luc tang van chiem uu the.'
    elif close < val:
        location = 'Gia dang nam duoi value area, rui ro suy yeu van cao.'
    else:
        location = 'Gia dang di trong value area, xu huong hien tai nghieng ve can bang.'

    if z_score <= -1.5:
        momentum = 'Z-Score da vao vung qua ban theo nguong he thong, can theo doi nhip hoi.'
    elif z_score >= 1.5:
        momentum = 'Z-Score dang cao, gia de rung lac neu mat dong luc.'
    else:
        momentum = 'Z-Score trung tinh, gia chua lech qua xa khoi mean ngan han.'

    if prob_up is None:
        model_view = 'Mo hinh xac suat chua du du lieu de dua ra xac suat huong gia ke tiep.'
    else:
        direction = 'tang' if prob_up >= 55 else ('giam' if prob_up <= 45 else 'di ngang')
        ev_text = f' EV uoc tinh {ev:.2f}%.' if ev is not None else ''
        model_view = f'Mo hinh xac suat nghieng ve kich ban {direction} voi xac suat {prob_up:.2f}%.' + ev_text

    risks = []
    if signal != 'Normal':
        risks.append(f'Tin hieu hien tai la {signal}, can xac nhan them bang nen va thanh khoan.')
    if acceleration < 0:
        risks.append('Gia toc gia dang am, nhip tang neu co co the chua ben.')
    if roll_skew < 0:
        risks.append('Rolling skew am cho thay loi suat ngan han van lech ve phia giam.')
    if abs(rel_z_score) > 1:
        risks.append('Relative Z-Score dang lech ro khoi POC, can de phong nhip mean reversion.')
    if not risks:
        risks.append('Rui ro ngan han chua noi bat, van nen quan tri vi the theo POC va VAL.')

    return {
        'title': f'AI Analysis - {snapshot["ticker"]}',
        'summary': f'{location} {momentum} {model_view}',
        'bullets': risks[:3],
    }


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
