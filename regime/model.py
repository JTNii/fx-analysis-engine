"""
FX Analysis Engine — Regime Model
HMM-based market regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from hmmlearn import hmm
import logging

logger = logging.getLogger(__name__)


class RegimeModel:
    """HMM regime detection for FX markets."""

    REGIME_LABELS = {
        0: "LOW_VOL_TREND",
        1: "HIGH_VOL_RANGE",
        2: "CRISIS_STRESS",
    }

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.model: Optional[hmm.GaussianHMM] = None
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit HMM on feature matrix.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of regime labels
        """
        if X.shape[0] < 100:
            logger.warning(f"Not enough data for HMM: {X.shape[0]} samples")
            return np.zeros(X.shape[0], dtype=int)

        try:
            self.model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="full",
                n_iter=50,
                random_state=42,
            )
            self.model.fit(X.values)
            self._fitted = True
            regimes = self.model.predict(X.values)
            logger.info(f"HMM fitted: {len(np.unique(regimes))} regimes detected")
            return regimes
        except Exception as e:
            logger.error(f"HMM fit failed: {e}")
            return np.zeros(X.shape[0], dtype=int)

    def predict_regime(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime for new data."""
        if not self._fitted:
            return np.zeros(len(X), dtype=int)
        return self.model.predict(X.values)

    def get_regime_probs(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get probability of each regime."""
        if not self._fitted:
            return pd.DataFrame()
        probs = self.model.predict_proba(X.values)
        return pd.DataFrame(probs, columns=[f"regime_{i}" for i in range(self.n_components)])

    def get_current_regime(self, X: pd.DataFrame) -> Tuple[int, float]:
        """Get current regime and confidence."""
        if not self._fitted or len(X) == 0:
            return 0, 0.0
        latest = X.iloc[-1:].values
        probs = self.model.predict_proba(latest)[0]
        regime = np.argmax(probs)
        confidence = probs[regime]
        return regime, confidence

    def diagnostics(self, regimes: np.ndarray) -> Dict:
        """Compute regime transition diagnostics."""
        unique, counts = np.unique(regimes, return_counts=True)
        transitions = np.sum(regimes[:-1] != regimes[1:])
        
        return {
            "num_regimes": len(unique),
            "regime_distribution": {str(k): int(v) for k, v in zip(unique, counts)},
            "num_transitions": int(transitions),
            "transition_rate": transitions / len(regimes) if len(regimes) > 0 else 0,
            "labels": {int(k): self.REGIME_LABELS.get(int(k), "UNKNOWN") for k in unique},
        }
