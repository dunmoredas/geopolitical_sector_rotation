# src/signal.py
"""
Geopolitical Risk Signal Construction
Core intellectual property: Composite signal from GPR + VIX + FTQ proxies
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class GeopoliticalSignal:
    """
    Multi-component signal that identifies geopolitical stress regimes.
    
    Key innovation: Removes VIX component from GPR to isolate 
    pure geopolitical signal vs general market fear.
    """
    
    def __init__(self, lookback=252):
        self.lookback = lookback
        self.weights = {
            'gpr_residual': 0.30,
            'vix': 0.20, 
            'gold': 0.25,
            'put_call_proxy': 0.25
        }
        
    def construct(self, gpr, vix, gold, put_call):
        """
        Build composite signal from components.
        
        Parameters:
        - gpr: Geopolitical Risk Index (Caldara & Iacoviello)
        - vix: CBOE Volatility Index
        - gold: Gold price (flight-to-quality proxy)
        - put_call: Put/call ratio or proxy
        
        Returns:
        - DataFrame with raw and smoothed composite signal
        """
        # Step 1: Remove VIX from GPR (address multicollinearity)
        gpr_clean = self._deconfound_gpr(gpr, vix)
        
        # Step 2: Normalize all components to Z-scores
        components = pd.DataFrame({
            'gpr_z': self._rolling_zscore(gpr_clean),
            'vix_z': self._rolling_zscore(vix),
            'gold_z': self._rolling_zscore(gold.pct_change().rolling(5).mean()),
            'pc_z': self._rolling_zscore(put_call)
        })
        
        # Step 3: Weighted composite
        signal = pd.DataFrame(index=components.index)
        signal['raw'] = (
            components['gpr_z'] * self.weights['gpr_residual'] +
            components['vix_z'] * self.weights['vix'] +
            components['gold_z'] * self.weights['gold'] +
            components['pc_z'] * self.weights['put_call_proxy']
        )
        
        # Step 4: Smooth to reduce noise
        signal['smooth'] = signal['raw'].rolling(5, min_periods=1).mean()
        
        # Step 5: Classify regimes
        signal['regime'] = pd.cut(
            signal['smooth'], 
            bins=[-np.inf, 0.5, 1.5, np.inf],
            labels=['calm', 'elevated', 'crisis']
        )
        
        return signal
    
    def _deconfound_gpr(self, gpr, vix):
        """Regress GPR on VIX, return residuals as clean geopolitical signal."""
        combined = pd.DataFrame({'gpr': gpr, 'vix': vix}).dropna()
        X = combined[['vix']].values
        y = combined['gpr'].values
        
        reg = LinearRegression().fit(X, y)
        residuals = y - reg.predict(X)
        
        r2 = reg.score(X, y)
        print(f"VIX explains {r2:.1%} of GPR variance")
        print(f"Using residuals as pure geopolitical signal")
        
        return pd.Series(residuals, index=combined.index)
    
    def _rolling_zscore(self, series):
        """Calculate rolling Z-score normalization."""
        roll_mean = series.rolling(self.lookback, min_periods=63).mean()
        roll_std = series.rolling(self.lookback, min_periods=63).std()
        return (series - roll_mean) / roll_std