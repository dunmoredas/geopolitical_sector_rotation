"""
Generate summary statistics CSV from backtest results.
Run after completing all three notebooks, or as standalone.

Usage:
    python scripts/generate_summary.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
from src.signal import GeopoliticalSignal
from src.backtest import SectorRotationBacktest


def get_price_series(data, ticker_name):
    """Extract a clean 1D price series from yfinance data."""
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            series = data['Close'].iloc[:, 0]
        elif 'Adj Close' in data.columns.get_level_values(0):
            series = data['Adj Close'].iloc[:, 0]
        else:
            series = data.iloc[:, 0]
    elif 'Adj Close' in data.columns:
        series = data['Adj Close']
    elif 'Close' in data.columns:
        series = data['Close']
    else:
        series = data.iloc[:, 0]
    
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    
    return pd.Series(
        series.values.flatten() if hasattr(series.values, 'flatten') else series.values,
        index=series.index,
        name=ticker_name
    )


def main():
    print("Generating summary statistics...")
    
    # ── Load data ──
    sector_etfs = {
        'XLK': 'Technology', 'XLC': 'Comm Services', 'XLY': 'Consumer Disc',
        'XLE': 'Energy', 'XLI': 'Industrials', 'XLV': 'Healthcare',
        'XLP': 'Consumer Staples', 'XLB': 'Materials'
    }
    
    sector_prices = {}
    for ticker in sector_etfs:
        data = yf.download(ticker, start='2010-01-01', progress=False)
        if not data.empty:
            sector_prices[ticker] = get_price_series(data, ticker)
    
    sector_returns = pd.DataFrame(sector_prices).pct_change()
    
    # Load signal data
    vix = get_price_series(
        yf.download('^VIX', start='2010-01-01', progress=False), 'VIX'
    )
    gold = get_price_series(
        yf.download('GLD', start='2010-01-01', progress=False), 'Gold'
    )
    
    # Simulated GPR
    np.random.seed(42)
    vix_norm = (vix - vix.mean()) / vix.std()
    gpr = pd.Series(
        np.random.randn(len(vix)) * 15 + 100 + vix_norm * 8 + np.random.randn(len(vix)) * 10,
        index=vix.index, name='GPR'
    )
    
    event_spikes = {
        '2014-03-01': 40, '2016-06-24': 35, '2020-01-03': 45, '2020-03-01': 50,
        '2022-02-24': 60, '2023-10-07': 45, '2024-04-01': 40, '2025-04-02': 55,
    }
    for date_str, spike in event_spikes.items():
        date = pd.Timestamp(date_str)
        if date in gpr.index:
            for i in range(20):
                d = date + pd.Timedelta(days=i)
                if d in gpr.index:
                    gpr.loc[d] += spike * np.exp(-i/5)
    
    put_call = 0.5 + (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() * 3)
    put_call = put_call.clip(0.3, 1.5)
    
    # ── Construct signal ──
    signal_builder = GeopoliticalSignal(lookback=252)
    signal = signal_builder.construct(gpr, vix, gold, put_call=vix/20)
    
    # ── Align data ──
    common_dates = signal.dropna().index.intersection(sector_returns.dropna().index)
    signal_aligned = signal.loc[common_dates]
    sector_returns_aligned = sector_returns.loc[common_dates]
    all_sectors = list(sector_etfs.keys())
    benchmark_returns = sector_returns_aligned[all_sectors].mean(axis=1)
    
    # ── Run backtest ──
    backtest = SectorRotationBacktest(entry_threshold=1.5, exit_threshold=0.5)
    results = backtest.run(signal_aligned, sector_returns_aligned, benchmark_returns)
    metrics = backtest.get_metrics()
    
    # ── Crisis performance ──
    crises = {
        'COVID-19': ('2020-02-19', '2020-03-23'),
        'Ukraine Invasion': ('2022-02-23', '2022-03-31'),
        'Iran-Israel': ('2024-04-01', '2024-05-01'),
        'Liberation Day': ('2025-04-02', '2025-05-01'),
    }
    
    crisis_rows = []
    for name, (start, end) in crises.items():
        try:
            period = results.loc[start:end]
            if len(period) > 0:
                strat_ret = (1 + period['strategy_return']).prod() - 1
                bench_ret = (1 + period['benchmark_return']).prod() - 1
                crisis_rows.append({
                    'Metric': name,
                    'Strategy': f'{strat_ret:.1%}',
                    'Benchmark': f'{bench_ret:.1%}',
                    'Excess': f'{strat_ret - bench_ret:.1%}',
                    'Category': 'Crisis Performance'
                })
        except:
            pass
    
    # ── Build summary DataFrame ──
    summary_rows = [
        {'Metric': 'Total Return', 'Strategy': metrics.get('Total Return', '-'),
         'Benchmark': '-', 'Excess': '-', 'Category': 'Returns'},
        {'Metric': 'Annualized Return', 'Strategy': metrics.get('Annualized Return', '-'),
         'Benchmark': '-', 'Excess': '-', 'Category': 'Returns'},
        {'Metric': 'Annualized Volatility', 'Strategy': metrics.get('Annualized Volatility', '-'),
         'Benchmark': '-', 'Excess': '-', 'Category': 'Risk'},
        {'Metric': 'Sharpe Ratio', 'Strategy': metrics.get('Sharpe Ratio', '-'),
         'Benchmark': '-', 'Excess': '-', 'Category': 'Risk-Adjusted'},
        {'Metric': 'Max Drawdown', 'Strategy': metrics.get('Max Drawdown', '-'),
         'Benchmark': '-', 'Excess': '-', 'Category': 'Risk'},
        {'Metric': 'Win Rate', 'Strategy': metrics.get('Win Rate', '-'),
         'Benchmark': '-', 'Excess': '-', 'Category': 'Consistency'},
        {'Metric': 'Information Ratio', 'Strategy': metrics.get('Information Ratio', '-'),
         'Benchmark': '-', 'Excess': '-', 'Category': 'Risk-Adjusted'},
        {'Metric': 'Number of Trades', 'Strategy': str(metrics.get('Number of Trades', '-')),
         'Benchmark': '-', 'Excess': '-', 'Category': 'Activity'},
        {'Metric': 'Avg Holding Period', 'Strategy': metrics.get('Avg Holding Period (days)', '-'),
         'Benchmark': '-', 'Excess': '-', 'Category': 'Activity'},
    ]
    
    summary_df = pd.DataFrame(summary_rows + crisis_rows)
    
    # ── Save ──
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / 'summary_stats.csv'
    summary_df.to_csv(output_path, index=False)
    
    print(f"✓ Saved {len(summary_df)} metrics to {output_path}")
    print(f"\n{summary_df.to_string(index=False)}")


if __name__ == '__main__':
    main()