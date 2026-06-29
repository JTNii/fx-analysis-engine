"""
FX Analysis Engine — Risk Manager
Pre-trade checks, position sizing limits, drawdown circuit breaker.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    passed: bool
    level: str  # NORMAL, WARNING, BLOCKED
    message: str
    max_position: float


class RiskManager:
    """Comprehensive risk management."""

    def __init__(
        self,
        max_position_per_pair: float = 0.25,
        max_total_exposure: float = 1.0,
        max_daily_loss_pct: float = 0.02,
        max_drawdown_pct: float = 0.10,
    ):
        self.max_position = max_position_per_pair
        self.max_total_exposure = max_total_exposure
        self.max_daily_loss = max_daily_loss_pct
        self.max_drawdown = max_drawdown_pct
        self._daily_pnl = 0.0
        self._peak_nav = 1.0
        self._current_nav = 1.0

    def pre_trade_check(
        self,
        pair: str,
        proposed_size: float,
        current_positions: Optional[Dict[str, float]] = None,
        current_nav: float = 1.0,
    ) -> RiskCheckResult:
        """Check if a trade passes risk limits."""
        issues = []
        
        # Position size limit
        if abs(proposed_size) > self.max_position:
            issues.append(f"Position {proposed_size:.3f} exceeds max {self.max_position:.3f}")
            proposed_size = np.sign(proposed_size) * self.max_position
        
        # Total exposure
        total_exposure = abs(proposed_size)
        if current_positions:
            total_exposure += sum(abs(v) for v in current_positions.values())
        
        if total_exposure > self.max_total_exposure:
            issues.append(f"Total exposure {total_exposure:.3f} exceeds max {self.max_total_exposure:.3f}")
            proposed_size = max(0, self.max_total_exposure - sum(abs(v) for v in (current_positions or {}).values()))
        
        # Daily loss limit
        if self._daily_pnl < -self.max_daily_loss:
            issues.append(f"Daily loss limit reached: {self._daily_pnl*100:.1f}%")
            proposed_size = 0.0
        
        # Drawdown circuit breaker
        drawdown = (self._peak_nav - self._current_nav) / self._peak_nav
        if drawdown > self.max_drawdown:
            issues.append(f"Drawdown {drawdown*100:.1f}% exceeds limit {self.max_drawdown*100:.1f}%")
            proposed_size = 0.0
        
        passed = len(issues) == 0
        level = "BLOCKED" if not passed and proposed_size == 0 else "WARNING" if issues else "NORMAL"
        
        return RiskCheckResult(
            passed=passed,
            level=level,
            message="; ".join(issues) if issues else "All checks passed",
            max_position=float(proposed_size),
        )

    def update_pnl(self, pnl: float, nav: float):
        """Update P&L and NAV tracking."""
        self._daily_pnl += pnl
        self._current_nav = nav
        self._peak_nav = max(self._peak_nav, nav)
        
        if pnl < 0:
            self._daily_pnl = 0  # Reset on new day
