"""
FX Analysis Engine — Signal Engine
Multi-factor alpha signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SignalEngine:
    """Generate trading signals from features and regimes."""

    def __init__(self, weights: Optional[Dict] = None):
        # Default weights for different signal components
        self.weights = weights or {
            "pc1": 0.4,
            "pc2": 0.3,
            "pc3": 0.3,
            "cot": 0.3,
            "momentum": 0.2,
        }

    def generate_signals(self, features: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
        """
        Generate signals for all currency pairs.
        
        Args:
            features: Feature matrix from FXFeatures.compute_all()
            regimes: Regime labels from RegimeModel
            
        Returns:
            DataFrame with signal columns per pair
        """
        if features.empty or len(features) == 0:
            return pd.DataFrame()

        # Identify PC columns
        pc_cols = [c for c in features.columns if c.startswith("PC")]
        cot_cols = [c for c in features.columns if c.startswith("cot_")]
        mom_cols = [c for c in features.columns if c.startswith("mom_1m_")]
        vol_cols = [c for c in features.columns if c.startswith("vol_")]
        
        # Extract pair names from column prefixes
        pairs = set()
        for col in features.columns:
            if "_" in col and not col.startswith("PC"):
                pair = col.split("_")[1]
                pairs.add(pair)
        
        signals = {}
        for pair in pairs:
            # Get pair-specific features
            pair_vol = features[f"vol_{pair}"] if f"vol_{pair}" in features.columns else pd.Series(1.0, index=features.index)
            pair_cot = features[f"cot_{pair}"] if f"cot_{pair}" in features.columns else pd.Series(0.0, index=features.index)
            pair_mom = features[f"mom_1m_{pair}"] if f"mom_1m_{pair}" in features.columns else pd.Series(0.0, index=features.index)
            
            # Base signal from PCs (common across all pairs)
            if pc_cols:
                pc_signal = sum(
                    self.weights.get(f"pc{i+1}", 1/len(pc_cols)) * features[pc_cols[i]]
                    for i in range(min(len(pc_cols), 3))
                ) / max(sum(self.weights.get(f"pc{i+1}", 1/3) for i in range(min(len(pc_cols), 3))), 1e-8)
            else:
                pc_signal = pd.Series(0.0, index=features.index)
            
            # Pair-specific adjustments
            signal = (
                0.5 * pc_signal +
                0.3 * pair_cot +
                0.2 * pair_mom
            )
            
            # Volatility scaling
            signal = signal / (pair_vol + 1e-8)
            
            # Regime adjustment
            regime_adj = np.where(regimes == 0, 1.1,  # Low vol trend: slight leverage
                                np.where(regimes == 1, 1.5,  # High vol range: reduce
                                         0.5))  # Crisis: heavy reduction
            
            signal = signal * regime_adj
            
            # Tanh squashing to [-1, 1]
            signal = np.tanh(signal.values)
            
            signals[pair] = pd.Series(signal, index=features.index)
        
        return pd.DataFrame(signals)

    def generate_single_signal(
        self, 
        pcs: np.ndarray, 
        regime: int, 
        vol: float, 
        cot: float = 0.0, 
        momentum: float = 0.0
    ) -> float:
        """
        Generate signal for a single point (for live trading).
        
        Args:
            pcs: [pc1, pc2, pc3] principal component values
            regime: Current regime (0, 1, or 2)
            vol: Current volatility
            cot: COT positioning z-score
            momentum: Recent momentum
            
        Returns:
            Signal in [-1, 1]
        """
        signal = 0.4 * pcs[0] + 0.3 * pcs[1] + 0.3 * pcs[2]
        signal += 0.3 * cot
        signal += 0.2 * momentum
        
        # Regime adjustment
        if regime == 0:
            signal *= 1.1
        elif regime == 1:
            signal *= 1.5
        else:
            signal *= 0.5
        
        # Volatility scaling
        signal = signal / (vol + 1e-6)
        
        return float(np.tanh(signal))
