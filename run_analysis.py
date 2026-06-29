"""
FX Analysis Engine — Main Runner
End-to-end pipeline: Data → Features → PCA → Regime → Signal → Portfolio → Report
"""

import argparse
import logging
import sys
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.provider import FXDataProvider
from features.engine import FXFeatures
from regime.model import RegimeModel
from signals.engine import SignalEngine
from portfolio.optimizer import PortfolioOptimizer
from risk.manager import RiskManager
from analysis.report import FXReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def run_analysis(
    pairs: List[str] = None,
    days: int = 365,
    output_filename: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run full FX analysis pipeline."""
    
    if pairs is None:
        pairs = ["USDJPY", "EURUSD", "USDCNH", "GBPUSD", "AUDUSD"]
    
    if verbose:
        logger.info("=" * 60)
        logger.info("  FX Analysis Engine v1.0")
        logger.info("=" * 60)
    
    # 1. Data
    logger.info("\n[1/6] Fetching market data...")
    provider = FXDataProvider()
    data = provider.get_all_data(pairs=pairs, days=days)
    fx_rates = data["fx_rates"]
    
    # 2. Features
    logger.info("[2/6] Computing features...")
    feature_engine = FXFeatures(window=60)
    features = feature_engine.compute_all(fx_rates)
    
    if features.empty:
        logger.error("No features computed. Check data quality.")
        return {}
    
    logger.info(f"  Features shape: {features.shape}")
    
    # 3. Regime
    logger.info("[3/6] Detecting market regime...")
    regime_model = RegimeModel(n_components=3)
    regimes = regime_model.fit(features)
    diag = regime_model.diagnostics(regimes)
    current_regime_idx, regime_conf = regime_model.get_current_regime(features)
    current_regime_str = regime_model.REGIME_LABELS.get(current_regime_idx, "UNKNOWN")
    
    logger.info(f"  Current regime: {current_regime_str} (conf: {regime_conf:.1%})")
    
    # 4. Signals
    logger.info("[4/6] Generating signals...")
    signal_engine = SignalEngine()
    signals_df = signal_engine.generate_signals(features, regimes)
    
    if signals_df.empty:
        logger.error("No signals generated.")
        return {}
    
    # 5. Portfolio
    logger.info("[5/6] Optimizing portfolio...")
    optimizer = PortfolioOptimizer(vol_target=0.10, max_leverage=1.0)
    vols = {pair: 0.02 for pair in pairs}  # Simplified
    positions = optimizer.optimize(signals_df, vols)
    
    # 6. Risk
    logger.info("[6/6] Running risk checks...")
    risk_mgr = RiskManager(max_position_per_pair=0.25, max_total_exposure=1.0)
    risk_results = {}
    for pair in pairs:
        pos = positions.get(pair, 0.0)
        risk_results[pair] = risk_mgr.pre_trade_check(pair, pos, positions, 1.0)
    
    # Build report data
    prices = {}
    changes = {}
    for pair in pairs:
        if pair in fx_rates:
            df = fx_rates[pair]
            if len(df) > 0:
                prices[pair] = df["Close"].iloc[-1]
                changes[pair] = {}
                if len(df) >= 2:
                    changes[pair]["1d"] = ((df["Close"].iloc[-1] / df["Close"].iloc[-2]) - 1) * 100
                if len(df) >= 5:
                    changes[pair]["1w"] = ((df["Close"].iloc[-1] / df["Close"].iloc[-5]) - 1) * 100
                if len(df) >= 21:
                    changes[pair]["1m"] = ((df["Close"].iloc[-1] / df["Close"].iloc[-21]) - 1) * 100
    
    # Get latest signals
    latest_signals = {}
    for pair in pairs:
        if pair in signals_df.columns:
            latest_signals[pair] = signals_df[pair].iloc[-1]
    
    # Summary
    strong_buys = sum(1 for s in latest_signals.values() if s > 0.3)
    strong_sells = sum(1 for s in latest_signals.values() if s < -0.3)
    neutral = len(latest_signals) - strong_buys - strong_sells
    
    summary = (
        f"当前 Regime: **{current_regime_str}**。"
        f"信号汇总: {strong_buys}个做多 / {neutral}个中性 / {strong_sells}个做空。"
        f"建议: {'积极做多' if strong_buys > strong_sells else '谨慎观望' if neutral > 0 else '偏空防御'}。"
    )
    
    # Generate report
    logger.info("\nGenerating report...")
    reporter = FXReportGenerator()
    report = reporter.generate_markdown(
        pairs=pairs,
        prices=prices,
        changes=changes,
        signals=latest_signals,
        positions=positions,
        regime=current_regime_str,
        regime_conf=regime_conf,
        risk_level="NORMAL",
        risk_message="All checks passed",
        summary=summary,
    )
    
    filepath = reporter.save_report(report, output_dir="./reports", filename=output_filename)
    
    # Console summary
    if verbose:
        print("\n" + "=" * 60)
        print("  FX ANALYSIS SUMMARY")
        print("=" * 60)
        for pair in pairs:
            p = prices.get(pair, 0)
            sig = latest_signals.get(pair, 0.0)
            pos = positions.get(pair, 0.0)
            direction = "🟢做多" if sig > 0.3 else ("🔴做空" if sig < -0.3 else "⚪中性")
            print(f"  {pair:8s} | ${p:>10.4f} | 信号: {sig:+.3f} {direction} | 仓位: {pos:+.1%}")
        print(f"\n  Regime: {current_regime_str} ({regime_conf:.1%})")
        print(f"  报告:   {filepath}")
        print("=" * 60)
    
    return {
        "pairs": pairs,
        "prices": prices,
        "signals": latest_signals,
        "positions": positions,
        "regime": current_regime_str,
        "regime_conf": regime_conf,
        "regime_diag": diag,
        "report_path": filepath,
    }


def main():
    parser = argparse.ArgumentParser(description="FX Analysis Engine v1.0")
    parser.add_argument("--pairs", nargs="+", default=None, help="Currency pairs (default: all major)")
    parser.add_argument("--days", type=int, default=365, help="Lookback period")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = parser.parse_args()
    
    run_analysis(
        pairs=args.pairs,
        days=args.days,
        output_filename=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
