"""
Simplified backtesting engine for geopolitical sector rotation.
Focus: Prove the concept works, not production readiness.
"""

import numpy as np
import pandas as pd

class SectorRotationBacktest:
    """
    Rules-based sector rotation strategy.
    
    Entry: Signal > threshold → rotate from growth to defensive sectors
    Exit: Signal < threshold → revert to neutral
    """
    
    def __init__(self, entry_threshold=1.5, exit_threshold=0.5):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # Sector definitions from the strategy document
        self.sell_sectors = ['XLK', 'XLC', 'XLY']  # Growth/Tech
        self.buy_sectors = ['XLE', 'XLI', 'XLV', 'XLP', 'XLB']  # Defensive
        
    def run(self, signal, sector_returns, benchmark_returns=None):
        """
        Execute the strategy.
        
        Returns DataFrame with strategy performance and trade log.
        """
        # Align data
        common_idx = signal.index.intersection(sector_returns.index)
        signal = signal.loc[common_idx]
        sector_returns = sector_returns.loc[common_idx]
        
        # Initialize tracking
        results = pd.DataFrame(index=common_idx)
        results['signal'] = signal['smooth']
        results['in_position'] = False
        
        # Strategy returns
        strategy_returns = pd.Series(0.0, index=common_idx)
        trade_log = []
        
        in_position = False
        entry_date = None
        
        for i, date in enumerate(common_idx):
            sig_val = signal.loc[date, 'smooth']
            
            # Monthly evaluation (first of month only - reduces noise)
            if i > 0 and date.month != common_idx[i-1].month:
                
                # Entry logic
                if not in_position and sig_val > self.entry_threshold:
                    in_position = True
                    entry_date = date
                    trade_log.append({
                        'date': date, 'action': 'ENTRY',
                        'signal': sig_val, 'regime': signal.loc[date, 'regime']
                    })
                
                # Exit logic
                elif in_position and sig_val < self.exit_threshold:
                    in_position = False
                    trade_log.append({
                        'date': date, 'action': 'EXIT',
                        'signal': sig_val, 'reason': 'Signal reversion',
                        'days_held': (date - entry_date).days
                    })
            
            results.loc[date, 'in_position'] = in_position
            
            # Calculate daily return if in position
            if in_position:
                # Short growth sectors, long defensive sectors
                daily_ret = 0
                for sector in self.sell_sectors:
                    if sector in sector_returns.columns:
                        daily_ret -= 0.05 * sector_returns.loc[date, sector]
                for sector in self.buy_sectors:
                    if sector in sector_returns.columns:
                        daily_ret += 0.02 * sector_returns.loc[date, sector]
                
                strategy_returns.loc[date] = daily_ret
        
        # Build results
        results['strategy_return'] = strategy_returns
        results['cumulative_return'] = (1 + strategy_returns).cumprod()
        
        # Benchmark comparison (equal-weight sector portfolio)
        if benchmark_returns is None:
            all_sectors = self.sell_sectors + self.buy_sectors
            benchmark_returns = sector_returns[all_sectors].mean(axis=1)
        
        results['benchmark_return'] = benchmark_returns
        results['benchmark_cumulative'] = (1 + benchmark_returns).cumprod()
        results['excess_return'] = strategy_returns - benchmark_returns
        
        self.results = results
        self.trade_log = pd.DataFrame(trade_log)
        
        return results
    
    def get_metrics(self):
        """Calculate key performance metrics."""
        rets = self.results['strategy_return'].dropna()
        bench_rets = self.results['benchmark_return'].dropna()
        
        # Returns
        total_ret = (1 + rets).prod() - 1
        ann_ret = (1 + total_ret) ** (252 / len(rets)) - 1
        ann_vol = rets.std() * np.sqrt(252)
        
        # Risk-adjusted
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        
        # Drawdown
        cumulative = (1 + rets).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win rate
        win_rate = (rets > 0).mean()
        
        # Information ratio
        excess = self.results['excess_return'].dropna()
        ir = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0
        
        # Trade statistics
        n_trades = len(self.trade_log[self.trade_log['action'] == 'ENTRY'])
        avg_holding = self.trade_log[
            self.trade_log['action'] == 'EXIT'
        ]['days_held'].mean() if n_trades > 0 else 0
        
        return {
            'Total Return': f'{total_ret:.2%}',
            'Annualized Return': f'{ann_ret:.2%}',
            'Annualized Volatility': f'{ann_vol:.2%}',
            'Sharpe Ratio': f'{sharpe:.2f}',
            'Max Drawdown': f'{max_dd:.2%}',
            'Win Rate': f'{win_rate:.1%}',
            'Information Ratio': f'{ir:.2f}',
            'Number of Trades': n_trades,
            'Avg Holding Period (days)': f'{avg_holding:.0f}'
        }