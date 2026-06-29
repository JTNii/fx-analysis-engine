# FX Analysis Engine，人民币汇率走势分析系统

💱 Production-grade multi-currency FX analysis system.

## Architecture

```
fx-analysis-engine/
├── data/provider.py         # FX rates, interest rates, COT data
├── features/engine.py       # PCA, GARCH vol, Carry, Momentum
├── regime/model.py          # HMM regime detection
├── signals/engine.py        # Multi-factor alpha signals
├── portfolio/optimizer.py   # Risk parity + vol targeting
├── risk/manager.py          # Pre-trade risk checks
├── analysis/report.py       # Markdown report generator
├── run_analysis.py          # CLI entry point
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis (default: 5 major pairs, 365 days)
python run_analysis.py

# Custom pairs and lookback
python run_analysis.py --pairs USDJPY EURUSD USDCNH --days 730

# Save with custom filename
python run_analysis.py --output fx_20260629.md
```

## Usage as Library

```python
from data.provider import FXDataProvider
from features.engine import FXFeatures
from regime.model import RegimeModel
from signals.engine import SignalEngine
from analysis.report import FXReportGenerator

# Fetch data
provider = FXDataProvider()
data = provider.get_all_data(pairs=["USDJPY", "EURUSD", "USDCNH"])

# Compute features
features = FXFeatures().compute_all(data["fx_rates"])

# Detect regime
regime = RegimeModel()
regimes = regime.fit(features)

# Generate signals
signals = SignalEngine().generate_signals(features, regimes)
```

## Signal Logic

- **PC1-3**: Principal components capture common FX movements
- **Volatility scaling**: Risk-adjusted signal strength
- **Regime adjustment**: Leverage varies by market state
- **TanH squashing**: Signals normalized to [-1, 1]

## Output

- Console: Summary table with prices, signals, positions
- `./reports/`: Full Markdown report

## License

MIT
