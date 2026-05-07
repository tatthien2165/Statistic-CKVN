from vnindex_quant_analysis import QuantModelEngine, logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import sys

# Đảm bảo in Tiếng Việt không lỗi trên Terminal Windows
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# BACKTESTING ENGINE
# ==========================================
class Backtester:
    """Hệ thống kiểm thử chiến thuật (Backtest) dựa trên tín hiệu định lượng VN-Index."""
    def __init__(self, initial_capital=100_000):
        self.initial_capital = initial_capital
        
    def run_backtest(self, df):
        capital = self.initial_capital
        position = 0  # 1 = Long, -1 = Short
        entry_price = 0
        equity_curve = [capital]
        trades = []
        
        # Mapping tín hiệu sang hành động trading
        # STRONG BUY: Bullish Reversal
        # BUY: Fake Nuke (Shakeout - điểm mua khi rũ bỏ)
        # STRONG SELL: Real Nuke (Crash - thoát hàng/Short)
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            price = curr['Close']
            signal = curr['Signal']
            
            # --- 1. MỞ VỊ THẾ (ENTRY) ---
            # LONG (Mua)
            if (signal == 'Bullish Reversal (Trend Change)' or signal == 'Fake Nuke (Shakeout/Bear trap)') and position <= 0:
                if position == -1: # Cover Short
                    capital *= entry_price / price
                position = 1
                entry_price = price
                trades.append({'time': df.index[i], 'type': 'LONG', 'price': price})
                
            # SHORT (Bán khống / Thoát hàng)
            elif signal == 'Real Nuke (Crash/Distribution)' and position >= 0:
                if position == 1: # Sell Long
                    capital *= price / entry_price
                position = -1
                entry_price = price
                trades.append({'time': df.index[i], 'type': 'SHORT', 'price': price})
                
            # --- 2. ĐÓNG VỊ THẾ (EXIT / TAKE PROFIT) ---
            # Chốt lời Long khi Z-Score quá cao (Hưng phấn quá đà)
            elif position == 1 and curr['Z_Score'] > 1.8:
                capital *= price / entry_price
                position = 0
                trades.append({'time': df.index[i], 'type': 'EXIT_LONG', 'price': price})
                
            # Chốt lời Short khi Z-Score phục hồi
            elif position == -1 and curr['Z_Score'] > -0.5:
                capital *= entry_price / price
                position = 0
                trades.append({'time': df.index[i], 'type': 'EXIT_SHORT', 'price': price})
                
            equity_curve.append(capital)
            
        # Tính toán Win Rate
        win_count = 0
        total_finished_trades = 0
        for j in range(0, len(trades)-1, 2):
            entry = trades[j]
            exit = trades[j+1]
            if entry['type'] == 'LONG':
                if exit['price'] > entry['price']: win_count += 1
            else: # SHORT
                if exit['price'] < entry['price']: win_count += 1
            total_finished_trades += 1
            
        win_rate = (win_count / total_finished_trades) if total_finished_trades > 0 else 0
        final_equity = equity_curve[-1]
        total_return = (final_equity / self.initial_capital) - 1
        max_dd = self.calculate_max_drawdown(equity_curve)
        
        return pd.Series(equity_curve, index=df.index), win_rate, total_finished_trades, total_return, max_dd

    @staticmethod
    def calculate_max_drawdown(equity_curve):
        equity_series = pd.Series(equity_curve)
        roll_max = equity_series.cummax()
        drawdown = (equity_series - roll_max) / roll_max
        return drawdown.min()

def plot_backtest_results(df, equity_curve, win_rate, total_trades, total_return, max_dd):
    """Vẽ biểu đồ kết quả Backtest so sánh với VN-Index."""
    import matplotlib.ticker as ticker
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3)
    
    x_axis = np.arange(len(df))
    time_fmt = '%Y-%m-%d'
    
    # Helper function for formatting x-axis dates
    def format_date(x, pos=None):
        idx = int(round(x))
        if idx >= 0 and idx < len(df):
            return df.index[idx].strftime(time_fmt)
        return ''

    # 1. Price Action & Signals
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x_axis, df['Close'], color='navy', alpha=0.6, label='VN-Index Price')
    
    # Vẽ các điểm mua/bán
    nukes = df[df['Signal'].str.contains('Nuke')]
    reversals = df[df['Signal'].str.contains('Bullish')]
    
    # Tìm index nguyên cho các điểm tín hiệu
    reversal_indices = df.index.get_indexer(reversals.index)
    nuke_real_indices = df.index.get_indexer(nukes[nukes['Signal'].str.contains('Real')].index)
    
    ax1.scatter(reversal_indices, reversals['Close'], color='green', marker='^', s=100, label='Buy/Reversal')
    ax1.scatter(nuke_real_indices, 
                nukes[nukes['Signal'].str.contains('Real')]['Close'], 
                color='red', marker='v', s=100, label='Sell/Crash')
    
    ax1.set_title('VN-Index Backtest Signals History', fontsize=14, fontweight='bold')
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax1.legend()
    
    # 2. Equity Curve vs Benchmark
    ax2 = fig.add_subplot(gs[1])
    benchmark = (df['Close'] / df['Close'].iloc[0]) * equity_curve.iloc[0]
    
    ax2.plot(x_axis, equity_curve.values, color='forestgreen', linewidth=2.5, label='Strategy Equity')
    ax2.plot(x_axis, benchmark.values, color='gray', linestyle='--', alpha=0.7, label='Buy & Hold Benchmark')
    ax2.fill_between(x_axis, equity_curve.iloc[0], equity_curve.values, color='green', alpha=0.1)
    
    # Annotations
    stats_text = (f"Win Rate: {win_rate:.2%}\n"
                  f"Total Return: {total_return:.2%}\n"
                  f"Total Trades: {total_trades}\n"
                  f"Max Drawdown: {max_dd:.2%}")
    
    ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_title('Growth of Capital: Strategy vs VN-Index Benchmark', fontsize=12, fontweight='bold')
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax2.set_ylabel('Capital')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" VN-INDEX QUANTITATIVE BACKTESTING SYSTEM ")
    print("="*50 + "\n")
    
    # 1. Cấu hình thời gian
    START_DATE = '2018-01-01'
    END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    # 2. Chạy Engine để lấy dữ liệu và tín hiệu
    engine = QuantModelEngine(start_date=START_DATE, end_date=END_DATE, interval='1D')
    df = engine.run()
    
    # 3. Chạy Backtest
    print("\n[+] Đang chạy mô phỏng giao dịch...")
    bt = Backtester(initial_capital=100_000)
    equity, win_rate, trades, ret, mdd = bt.run_backtest(df)
    
    # 4. In kết quả tóm tắt
    print("\n" + "*"*30)
    print(" KẾT QUẢ BACKTEST TÓM TẮT ")
    print(f" Khoảng thời gian: {START_DATE} -> {END_DATE}")
    print(f" Vốn ban đầu: 100,000")
    print(f" Vốn cuối cùng: {equity.iloc[-1]:,.0f}")
    print(f" Tổng lợi nhuận: {ret:.2%}")
    print(f" Tỷ lệ thắng (Win Rate): {win_rate:.2%}")
    print(f" Tổng số lệnh đã đóng: {trades}")
    print(f" Sụt giảm lớn nhất (MDD): {mdd:.2%}")
    print("*"*30 + "\n")
    
    # 5. Vẽ biểu đồ
    plot_backtest_results(df, equity, win_rate, trades, ret, mdd)
