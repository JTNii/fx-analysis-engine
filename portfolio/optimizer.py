"""
FX Analysis Engine — Portfolio Optimizer
Risk parity + volatility targeting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Risk parity portfolio with volatility targeting."""

    def __init__(self, vol_target: float = 0.10, max_leverage: float = 1.0):
        self.vol_target = vol_target
        self.max_leverage = max_leverage

    def optimize(self, signals: pd.DataFrame, vols: Dict[str, float]) -> Dict[str, float]:
        """
        Compute risk-parity weights from signals and volatilities.
        
        Args:
            signals: DataFrame with signal columns per pair
            vols: Dict of pair -> volatility
            
        Returns:
            Dict of pair -> weight (clipped to [-max_leverage, max_leverage])
        """
        weights = {}
        for pair in signals.columns:
            sig = signals[pair].iloc[-1] if len(signals) > 0 else 0.0
            vol = vols.get(pair, 0.02)
            
            # Risk parity: weight proportional to 1/vol
            inv_vol = 1.0 / (vol + 1e-6)
            weight = sig * inv_vol
            weight = np.clip(weight, -self.max_leverage, self.max_leverage)
            weights[pair] = float(weight)
        
        return weights

    def position_size(self, signal: float, vol: float) -> float:
        """Calculate position size from signal and volatility."""
        weight = self.vol_target / (vol + 1e-6)
        return float(np.clip(signal * weight, -self.max_leverage, self.max_leverage))
