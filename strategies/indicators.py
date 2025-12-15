"""
Indicator Helper Module (Custom Implementation - Alternative to Built-in Indicators)

⚠️ IMPORTANT: Nautilus Trader has built-in indicators that are RECOMMENDED to use!

RECOMMENDED APPROACH - Use Nautilus Trader's Built-in Indicators:
    from nautilus_trader.indicators.momentum import RelativeStrengthIndex
    from nautilus_trader.indicators.trend import ExponentialMovingAverage
    from nautilus_trader.indicators.volatility import AverageTrueRange
    
    # Initialize in __init__:
    self.rsi_indicator = RelativeStrengthIndex(period=14)
    
    # Update in on_bar():
    self.rsi_indicator.handle_bar(bar)
    if self.rsi_indicator.initialized:
        current_rsi = self.rsi_indicator.value
    
    See: https://nautilustrader.io/docs/latest/api_reference/indicators/

ALTERNATIVE APPROACH - Custom Indicators (this file):
    This module provides custom technical indicator implementations using pandas and numpy.
    These are provided as an alternative if you prefer pandas-based calculations or need
    custom indicator logic. All indicators return pandas Series with the same index as the input data.
    
    You can use these custom helpers if:
    - You prefer pandas-based calculations
    - You need custom indicator logic not available in built-in indicators
    - You want to understand how indicators are calculated
    
    Usage:
        from .indicators import rsi, ema, atr
        prices_series = pd.Series(self.prices)
        rsi_values = rsi(prices_series, period=14)
"""

import pandas as pd
import numpy as np
from typing import Optional


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    Values range from 0 to 100:
    - RSI < 30: Oversold (potential buy signal)
    - RSI > 70: Overbought (potential sell signal)
    
    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss (over period)
    
    Args:
        prices: Series of closing prices
        period: Number of periods for RSI calculation (default: 14)
    
    Returns:
        Series of RSI values (0-100)
    
    References:
        - Technical Analysis: https://www.investopedia.com/terms/r/rsi.asp
        - Pine Script RSI: https://www.tradingview.com/pine-script-reference/v5/#fun_ta%7Brsi%7D
    """
    if len(prices) < period + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    EMA gives more weight to recent prices, making it more responsive to price changes
    than Simple Moving Average (SMA).
    
    Formula:
        EMA = Price(t) * k + EMA(y) * (1 - k)
        where k = 2 / (period + 1)
    
    Args:
        prices: Series of closing prices
        period: Number of periods for EMA calculation
    
    Returns:
        Series of EMA values
    
    References:
        - Technical Analysis: https://www.investopedia.com/terms/e/ema.asp
    """
    return prices.ewm(span=period, adjust=False).mean()


def sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    SMA is the average price over a specified number of periods.
    
    Args:
        prices: Series of closing prices
        period: Number of periods for SMA calculation
    
    Returns:
        Series of SMA values
    """
    return prices.rolling(window=period).mean()


def macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.
    
    Components:
        - MACD line: EMA(fast) - EMA(slow)
        - Signal line: EMA of MACD line
        - Histogram: MACD line - Signal line
    
    Args:
        prices: Series of closing prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    
    References:
        - Technical Analysis: https://www.investopedia.com/terms/m/macd.asp
    """
    ema_fast = ema(prices, fast_period)
    ema_slow = ema(prices, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    ATR measures market volatility by calculating the average of true ranges over a period.
    True Range is the maximum of:
        - High - Low
        - |High - Previous Close|
        - |Low - Previous Close|
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: Number of periods for ATR calculation (default: 14)
    
    Returns:
        Series of ATR values
    
    References:
        - Technical Analysis: https://www.investopedia.com/terms/a/atr.asp
    """
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_values = true_range.rolling(window=period).mean()
    
    return atr_values


def bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of:
        - Middle Band: SMA(period)
        - Upper Band: Middle + (num_std * standard deviation)
        - Lower Band: Middle - (num_std * standard deviation)
    
    Args:
        prices: Series of closing prices
        period: Number of periods for SMA calculation (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    
    References:
        - Technical Analysis: https://www.investopedia.com/terms/b/bollingerbands.asp
    """
    middle_band = sma(prices, period)
    std = prices.rolling(window=period).std()
    
    upper_band = middle_band + (num_std * std)
    lower_band = middle_band - (num_std * std)
    
    return upper_band, middle_band, lower_band


def validate_indicator_data(data: pd.Series, min_periods: int) -> bool:
    """
    Validate that there is enough data to calculate an indicator.
    
    Args:
        data: Series of price data
        min_periods: Minimum number of periods required
    
    Returns:
        True if enough data, False otherwise
    """
    return len(data) >= min_periods
