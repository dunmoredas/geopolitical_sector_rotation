"""
Visualization utilities for the Geopolitical Sector Rotation Strategy.
Centralizes plotting logic so notebooks stay clean and focused on analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


# ── Global Style ──────────────────────────────────────────────────────────

# Consistent color palette
COLORS = {
    'navy': '#1a237e',
    'red': '#e74c3c',
    'green': '#2ecc71',
    'orange': '#f39c12',
    'gray': '#95a5a6',
    'dark': '#2c3e50',
    'light_gray': '#ecf0f1',
    'white': '#ffffff',
}

# Regime colors
REGIME_COLORS = {
    'calm': '#2ecc71',
    'elevated': '#f39c12',
    'crisis': '#e74c3c',
}

# Quintile gradient (low → high signal)
QUINTILE_COLORS = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']


def set_style():
    """Apply consistent matplotlib style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': COLORS['light_gray'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
    })


def remove_spines(ax, keep_left=True, keep_bottom=True):
    """Clean up chart spines for a professional look."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not keep_left:
        ax.spines['left'].set_visible(False)
    else:
        ax.spines['left'].set_alpha(0.3)
    if not keep_bottom:
        ax.spines['bottom'].set_visible(False)
    else:
        ax.spines['bottom'].set_alpha(0.3)


def format_xaxis_dates(ax, years_every=2):
    """Format x-axis for date-based charts."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(years_every))
    ax.tick_params(axis='x', rotation=45)


def add_watermark(ax, text='Sirocco Quant', alpha=0.03):
    """Add subtle watermark to chart."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            fontsize=40, color='black', alpha=alpha,
            ha='center', va='center', weight='bold')


# ── Signal Charts ─────────────────────────────────────────────────────────

def plot_signal_overview(signal, save_path=None):
    """
    Three-panel signal overview chart:
    1. Raw vs smoothed signal with thresholds
    2. Regime classification timeline
    3. Signal with annotated historical events
    """
    signal_clean = signal.dropna()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # ── Panel 1: Signal with thresholds ──
    axes[0].plot(signal_clean.index, signal_clean['raw'],
                 alpha=0.3, color=COLORS['gray'], label='Raw Signal', linewidth=1)
    axes[0].plot(signal_clean.index, signal_clean['smooth'],
                 linewidth=2, color=COLORS['navy'], label='Smoothed (5-day)')
    axes[0].axhline(y=1.5, color=COLORS['red'], linestyle='--', linewidth=1.5,
                    label='Entry Threshold (1.5σ)', alpha=0.7)
    axes[0].axhline(y=0.5, color=COLORS['green'], linestyle='--', linewidth=1.5,
                    label='Exit Threshold (0.5σ)', alpha=0.7)

    # Shade crisis zones
    if (signal_clean['smooth'] > 1.5).any():
        axes[0].fill_between(signal_clean.index, 1.5, signal_clean['smooth'].max(),
                             where=signal_clean['smooth'] > 1.5,
                             alpha=0.1, color=COLORS['red'], label='Crisis Zone')

    axes[0].set_title('Geopolitical Risk Signal', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Signal (Z-score)')
    axes[0].legend(loc='upper left', fontsize=8)
    remove_spines(axes[0])

    # ── Panel 2: Regime timeline ──
    regime_colors_list = []
    for date in signal_clean.index:
        regime = signal_clean.loc[date, 'regime']
        if pd.notna(regime) and regime in REGIME_COLORS:
            regime_colors_list.append(REGIME_COLORS[regime])
        else:
            regime_colors_list.append('#cccccc')

    axes[1].scatter(signal_clean.index, [1] * len(signal_clean),
                    c=regime_colors_list, marker='s', s=2, alpha=0.8)

    legend_elements = [
        Patch(facecolor=REGIME_COLORS['calm'], label='Calm'),
        Patch(facecolor=REGIME_COLORS['elevated'], label='Elevated'),
        Patch(facecolor=REGIME_COLORS['crisis'], label='Crisis'),
    ]
    axes[1].legend(handles=legend_elements, loc='center right', fontsize=8)
    axes[1].set_title('Regime Classification', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_yticks([])
    remove_spines(axes[1], keep_left=False)

    # ── Panel 3: Historical events ──
    axes[2].plot(signal_clean.index, signal_clean['smooth'],
                 color=COLORS['navy'], linewidth=1.5)

    events = {
        '2014-03-01': 'Crimea',
        '2016-06-24': 'Brexit',
        '2020-01-03': 'Soleimani',
        '2020-03-01': 'COVID-19',
        '2022-02-24': 'Ukraine',
        '2023-10-07': 'Hamas\nAttack',
        '2024-04-01': 'Iran-\nIsrael',
        '2025-04-02': 'Liberation\nDay',
    }

    for date_str, label in events.items():
        event_date = pd.Timestamp(date_str)
        if event_date in signal_clean.index:
            signal_val = signal_clean.loc[event_date, 'smooth']
        else:
            closest_idx = signal_clean.index.searchsorted(event_date)
            if closest_idx < len(signal_clean):
                event_date = signal_clean.index[closest_idx]
                signal_val = signal_clean.loc[event_date, 'smooth']
            else:
                continue

        axes[2].axvline(x=event_date, color=COLORS['red'], alpha=0.4,
                        linestyle=':', linewidth=1)
        axes[2].annotate(label, xy=(event_date, signal_val),
                        xytext=(10, 20), textcoords='offset points',
                        fontsize=8, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', alpha=0.7),
                        arrowprops=dict(arrowstyle='->',
                                       color=COLORS['red'], alpha=0.5))

    axes[2].set_title('Signal vs Historical Geopolitical Events',
                      fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Smoothed Signal (Z-score)')
    remove_spines(axes[2])

    # Format x-axis for all panels
    for ax in axes:
        format_xaxis_dates(ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


# ── Backtest Charts ───────────────────────────────────────────────────────

def plot_backtest_overview(results, save_path=None):
    """
    Four-panel backtest overview:
    1. Cumulative returns (strategy vs benchmark)
    2. Drawdown
    3. Position status
    4. Rolling 1-year Sharpe
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: Cumulative Returns ──
    axes[0, 0].plot(results.index, results['cumulative_return'],
                    label='Strategy', linewidth=2, color=COLORS['navy'])
    axes[0, 0].plot(results.index, results['benchmark_cumulative'],
                    label='Equal-Weight Benchmark', linewidth=1,
                    color=COLORS['gray'], alpha=0.7)
    axes[0, 0].set_title('Cumulative Returns', fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel('Growth of $1')
    remove_spines(axes[0, 0])

    # ── Panel 2: Drawdown ──
    cumulative = results['cumulative_return']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    axes[0, 1].fill_between(drawdown.index, 0, drawdown,
                            color=COLORS['red'], alpha=0.3)
    axes[0, 1].plot(drawdown.index, drawdown,
                    color=COLORS['red'], linewidth=0.5)
    axes[0, 1].set_title('Strategy Drawdown', fontweight='bold')
    axes[0, 1].set_ylabel('Drawdown %')
    axes[0, 1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    remove_spines(axes[0, 1])

    # ── Panel 3: Position Status ──
    in_position = results['in_position'].astype(int)
    axes[1, 0].fill_between(results.index, 0, in_position,
                            color=COLORS['orange'], alpha=0.3)
    axes[1, 0].set_title('Position Status', fontweight='bold')
    axes[1, 0].set_ylabel('In Position')
    axes[1, 0].set_ylim(0, 1.2)
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(['Out', 'In'])
    remove_spines(axes[1, 0])

    # ── Panel 4: Rolling Sharpe ──
    rolling_sharpe = (
        results['strategy_return'].rolling(252).mean() /
        results['strategy_return'].rolling(252).std() * np.sqrt(252)
    )
    axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe,
                    color='teal', linewidth=1.5)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].fill_between(rolling_sharpe.index, 0, rolling_sharpe,
                            where=rolling_sharpe > 0,
                            color=COLORS['green'], alpha=0.1)
    axes[1, 1].fill_between(rolling_sharpe.index, 0, rolling_sharpe,
                            where=rolling_sharpe < 0,
                            color=COLORS['red'], alpha=0.1)
    axes[1, 1].set_title('Rolling 1-Year Sharpe Ratio', fontweight='bold')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    remove_spines(axes[1, 1])

    # Format x-axis for all panels
    for ax in axes.flatten():
        format_xaxis_dates(ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_signal_intensity_bars(regime_performance, save_path=None):
    """
    Bar chart showing strategy performance by signal intensity quintile.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_abs_return = max(abs(regime_performance['annualized_mean'].min()),
                         abs(regime_performance['annualized_mean'].max()))
    y_limit_return = max_abs_return * 1.4

    max_abs_sharpe = max(abs(regime_performance['sharpe'].min()),
                         abs(regime_performance['sharpe'].max()))
    y_limit_sharpe = max_abs_sharpe * 1.4

    # ── Panel 1: Annualized Return ──
    bars1 = axes[0].bar(range(5), regime_performance['annualized_mean'],
                        color=QUINTILE_COLORS, edgecolor='white',
                        linewidth=1.5, width=0.6)

    for i, bar in enumerate(bars1):
        height = bar.get_height()
        offset = y_limit_return * 0.03
        if height >= 0:
            axes[0].text(bar.get_x() + bar.get_width()/2.,
                        height + offset, f'{height:.1%}',
                        ha='center', va='bottom',
                        fontweight='bold', fontsize=10, color=COLORS['dark'])
        else:
            axes[0].text(bar.get_x() + bar.get_width()/2.,
                        height - offset, f'{height:.1%}',
                        ha='center', va='top',
                        fontweight='bold', fontsize=10, color=COLORS['dark'])

    axes[0].axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    axes[0].set_ylim(-y_limit_return, y_limit_return)
    axes[0].set_title('Annualized Return by Signal Intensity',
                      fontweight='bold', fontsize=13, pad=15)
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(['Lowest\nSignal', 'Low', 'Medium',
                             'High', 'Highest\nSignal'], fontsize=10)
    axes[0].set_ylabel('Annualized Return', fontsize=11)
    remove_spines(axes[0])

    # ── Panel 2: Sharpe Ratio ──
    bars2 = axes[1].bar(range(5), regime_performance['sharpe'],
                        color=QUINTILE_COLORS, edgecolor='white',
                        linewidth=1.5, width=0.6)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        offset = y_limit_sharpe * 0.03
        if height >= 0:
            axes[1].text(bar.get_x() + bar.get_width()/2.,
                        height + offset, f'{height:.2f}',
                        ha='center', va='bottom',
                        fontweight='bold', fontsize=10, color=COLORS['dark'])
        else:
            axes[1].text(bar.get_x() + bar.get_width()/2.,
                        height - offset, f'{height:.2f}',
                        ha='center', va='top',
                        fontweight='bold', fontsize=10, color=COLORS['dark'])

    axes[1].axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    axes[1].set_ylim(-y_limit_sharpe, y_limit_sharpe)
    axes[1].set_title('Sharpe Ratio by Signal Intensity',
                      fontweight='bold', fontsize=13, pad=15)
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(['Lowest\nSignal', 'Low', 'Medium',
                             'High', 'Highest\nSignal'], fontsize=10)
    axes[1].set_ylabel('Sharpe Ratio', fontsize=11)
    remove_spines(axes[1])

    plt.suptitle('Strategy Performance Increases with Signal Strength',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_sector_attribution(sector_analysis, sell_sectors, buy_sectors, save_path=None):
    """
    Horizontal bar chart showing sector return differences
    between crisis and calm periods.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color bars based on strategy direction
    colors = []
    for sector in sector_analysis.index:
        if sector in sell_sectors:
            colors.append(COLORS['red'])
        elif sector in buy_sectors:
            colors.append(COLORS['green'])
        else:
            colors.append(COLORS['gray'])

    ax.barh(sector_analysis.index, sector_analysis['Difference'],
            color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)

    ax.set_title('Sector Return Difference: Crisis vs Calm Periods',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Annualized Return Difference')
    ax.axvline(x=0, color='black', linewidth=0.8)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['green'], alpha=0.7, label='Strategy Long'),
        Patch(facecolor=COLORS['red'], alpha=0.7, label='Strategy Short'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    remove_spines(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


# ── Quick Helpers ─────────────────────────────────────────────────────────

def print_metrics_table(metrics):
    """Pretty-print a metrics dictionary."""
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"  {key:30s}: {value}")


def print_regime_summary(signal):
    """Print summary of signal regime distribution."""
    signal_clean = signal.dropna()
    print("\n" + "=" * 50)
    print("REGIME DISTRIBUTION")
    print("=" * 50)
    total = len(signal_clean)
    for regime in ['calm', 'elevated', 'crisis']:
        count = (signal_clean['regime'] == regime).sum()
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {regime:10s}: {count:5d} days ({pct:4.1f}%)  {bar}")