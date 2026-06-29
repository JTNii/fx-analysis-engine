"""
FX Analysis Engine — Feature Engineering
PCA, GARCH vol, Carry, Momentum, COT positioning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.decomposition import IncrementalPCA
import logging

logger = logging.getLogger(__name__)


class FXFeatures:
    """Multi-currency feature engineering."""

    def __init__(self, window: int = 60):
        self.window = window
        self._pca: Optional[IncrementalPCA] = None

    def compute_all(self, fx_rates: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute all features from FX rate data.
        
        Returns DataFrame with columns:
        - PC1, PC2, PC3 (principal components)
        - vol_* (GARCH volatility per pair)
        - carry_* (interest rate differential)
        - momentum_* (1m, 3m returns)
        - cot_* (COT positioning z-score proxy)
        """
        # 1. Compute returns for each pair
        returns_dict = {}
        for pair, df in fx_rates.items():
            if "Close" not in df.columns:
                continue
            returns = df["Close"].pct_change().dropna()
            returns_dict[pair] = returns

        if not returns_dict:
            logger.error("No valid returns data")
            return pd.DataFrame()

        # 2. Align all returns to common index
        aligned = pd.DataFrame(returns_dict)
        aligned = aligned.dropna()
        
        if len(aligned) < self.window:
            logger.warning(f"Not enough data: {len(aligned)} < {self.window}")
            return pd.DataFrame()

        # 3. Rolling PCA
        pcs = self._compute_pca(aligned)
        
        # 4. GARCH-style volatility (simplified)
        vols = self._compute_garch_vol(aligned)
        
        # 5. Carry factors (from interest rates)
        carries = self._compute_carry(aligned)
        
        # 6. Momentum factors
        mom_1m = aligned.pct_change(21).dropna()
        mom_3m = aligned.pct_change(63).dropna()
        
        # 7. COT positioning proxy
        cot = self._compute_cot_proxy(aligned)
        
        # 8. Combine all features
        features = pd.concat({
            **{f"PC{i+1}": pcs.iloc[:, i] for i in range(min(3, pcs.shape[1]))},
            **{f"vol_{pair}": vols.get(pair, pd.Series(dtype=float)).reindex(pcs.index) for pair in fx_rates},
            **{f"carry_{pair}": carries.get(pair, pd.Series(dtype=float)).reindex(pcs.index) for pair in fx_rates},
            **{f"mom_1m_{pair}": mom_1m.get(pair, pd.Series(dtype=float)).reindex(pcs.index) for pair in fx_rates},
            **{f"mom_3m_{pair}": mom_3m.get(pair, pd.Series(dtype=float)).reindex(pcs.index) for pair in fx_rates},
            **{f"cot_{pair}": cot.get(pair, pd.Series(dtype=float)).reindex(pcs.index) for pair in fx_rates},
        }, axis=1)
        
        features = features.dropna()
        return features

    def _compute_pca(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling principal components."""
        n_components = min(3, returns.shape[1])
        n_windows = len(returns) - self.window + 1
        
        if n_windows <= 0:
            return pd.DataFrame()
        
        all_components = []
        all_dates = []
        
        ipca = IncrementalPCA(n_components=n_components)
        
        # First window — fit on numpy array to avoid feature name warnings
        first_window = returns.iloc[:self.window].values
        ipca.partial_fit(first_window)
        
        # Subsequent windows — use numpy arrays
        for i in range(self.window, len(returns)):
            window_data = returns.iloc[i-self.window:i].values
            try:
                comp = ipca.transform(window_data[-1:])
                all_components.append(comp.flatten())
                all_dates.append(returns.index[i])
            except Exception:
                continue
        
        if not all_components:
            return pd.DataFrame()
        
        return pd.DataFrame(all_components, index=all_dates, 
                          columns=[f"PC{i+1}" for i in range(n_components)])

    def _compute_garch_vol(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """Simple GARCH-style volatility estimation."""
        vols = {}
        for col in returns.columns:
            ret = returns[col].dropna()
            if len(ret) < 20:
                continue
            # EWMA volatility (proxy for GARCH)
            ewma = ret.ewm(span=20).std()
            vols[col] = ewma
        return vols

    def _compute_carry(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """Interest rate differential carry factor."""
        carries = {}
        for col in returns.columns:
            ret = returns[col].dropna()
            if len(ret) < 2:
                continue
            # Simple carry: annualized return proxy
            carries[col] = ret * 252  # Annualize
        return carries

    def _compute_cot_proxy(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """COT positioning proxy from price momentum."""
        cots = {}
        for col in returns.columns:
            ret = returns[col].dropna()
            if len(ret) < 52:
                continue
            # Net position proxy: rolling z-score of returns
            rolling_mean = ret.rolling(52).mean()
            rolling_std = ret.rolling(52).std()
            cots[col] = (ret - rolling_mean) / (rolling_std + 1e-8)
        return cots
