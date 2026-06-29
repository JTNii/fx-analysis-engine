"""
FX Analysis Engine — Report Generator
Standardized Markdown report output.
"""

from datetime import datetime
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class FXReportGenerator:
    """Generate standardized FX analysis reports."""
    
    def generate_markdown(
        self,
        pairs: List[str],
        prices: Dict[str, float],
        changes: Dict[str, Dict],
        signals: Dict[str, float],
        positions: Dict[str, float],
        regime: str,
        regime_conf: float,
        risk_level: str,
        risk_message: str,
        summary: str,
    ) -> str:
        now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        
        report = f"""# 💱 FX 多币种分析报告

**分析时间**: {now}  
**当前 Regime**: {regime} (置信度: {regime_conf:.1%})  
**风控状态**: {risk_level}

---

## 📊 核心汇率

| 货币对 | 价格 | 1日 | 1周 | 1月 | 信号 | 仓位 |
|--------|------|-----|-----|-----|------|------|
"""
        for pair in pairs:
            p = prices.get(pair, 0)
            ch = changes.get(pair, {})
            sig = signals.get(pair, 0.0)
            pos = positions.get(pair, 0.0)
            sig_label = "🟢 做多" if sig > 0.3 else ("🔴 做空" if sig < -0.3 else "⚪ 中性")
            pos_label = f"{pos:+.1%}" if pos != 0 else "-"
            
            report += f"| {pair} | {p:.4f} | {ch.get('1d', 0):+.2f}% | {ch.get('1w', 0):+.2f}% | {ch.get('1m', 0):+.2f}% | {sig_label} | {pos_label} |\n"
        
        report += f"""
---

## 🧠 Regime 检测

| 指标 | 值 |
|------|-----|
| 当前状态 | **{regime}** |
| 置信度 | {regime_conf:.1%} |

---

## 🎯 信号汇总

| 货币对 | 信号值 | 方向 |
|--------|--------|------|
"""
        for pair in pairs:
            sig = signals.get(pair, 0.0)
            direction = "做多" if sig > 0.3 else ("做空" if sig < -0.3 else "中性")
            arrow = "🟢" if sig > 0.3 else ("🔴" if sig < -0.3 else "⚪")
            report += f"| {pair} | {sig:+.3f} | {arrow} {direction} |\n"
        
        report += f"""
---

## 🛡 风控

| 指标 | 值 |
|------|-----|
| 状态 | {risk_level} |
| 说明 | {risk_message} |

---

## 📝 总结

{summary}

---

> ⚠️ **免责声明**: 本分析基于公开数据和量化模型，仅供参考，不构成投资建议。外汇交易具有高风险。
"""
        return report
    
    def save_report(self, report: str, output_dir: str = "./reports", filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"fx_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = f"{output_dir}/{filename}"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {filepath}")
        return filepath
