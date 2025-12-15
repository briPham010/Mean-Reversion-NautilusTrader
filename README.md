# RSI Algorithm Take-Home Assignment

## Overview

Your task is to implement an RSI-based trading strategy using Nautilus Trader. You will convert a Pine Script RSI algorithm into a working Nautilus Trader strategy, run backtests, optimize parameters, and deliver results.

## What You Need to Do

### Core Requirements

1. **Implement the RSI Strategy** (`strategies/rsi_algo_template.py`)
   - Complete all TODO sections in the strategy template
   - Implement indicator calculations (RSI)
   - Implement entry logic (when RSI crosses below threshold)
   - Implement exit logic (when RSI crosses above threshold)
   - Optionally implement pyramid logic for adding to positions

2. **Run Backtests**
   - Execute backtests with your implemented strategy
   - Analyze performance metrics (Sharpe ratio, total return, max drawdown, etc.)

3. **Optimize Parameters**
   - Find optimal parameter values (RSI period, entry/exit thresholds)
   - Implement your own optimization method (grid search, random search, Bayesian optimization, etc.)

4. **Deliver Results**
   - Completed strategy implementation
   - Backtest results with optimized parameters
   - Brief summary of your approach and findings

## What's Provided (Scaffolding)

The following scaffolding is provided to help you get started. **You may use it at your discretion, but be aware that some parts may be incomplete or have issues that you'll need to navigate around.**

### Files Provided

- **`strategies/rsi_algo_template.py`**: Strategy template with TODO sections
  - Contains skeleton code and helper methods
  - You must implement the core trading logic
  - Some scaffolding may need fixes or adjustments

- **`strategies/indicators.py`**: Custom indicator helper functions (ALTERNATIVE)
  - RSI, EMA, MACD, ATR implementations using pandas
  - **Note**: Nautilus Trader has built-in indicators that are RECOMMENDED to use
  - See `strategies/rsi_algo_template.py` for examples of both approaches
  - Built-in indicators: https://nautilustrader.io/docs/latest/api_reference/indicators/

- **`config/backtest_gc.yaml`**: Configuration file
  - Defines strategy parameters, venue, and data paths
  - You may need to adjust paths or parameters

- **`run_backtest.py`**: Backtest runner script
  - Loads config and runs backtests
  - **May have issues** - you may need to fix data loading or other components
  - Use as a reference or build your own runner

- **`optimize_params.py`**: Parameter optimization skeleton
  - **NOT IMPLEMENTED** - you must implement your own optimization method
  - Provides structure and helper functions
  - Choose your optimization approach (grid search, random search, Bayesian, etc.)

- **`data/`**: Data directory (data files available via Google Drive)
  - The `data/` directory structure is provided
  - **Download data files from**: [Google Drive Link](https://drive.google.com/drive/folders/1A2W_LSuyhJuu1Vc-8tm-KidG-kxaPLQI?usp=sharing)
  - Download `gc_1m.parquet` and `gc_1m.csv` from the link above
  - Place the data files in the `data/` directory after downloading
  - Gold futures (GC) 1-minute OHLCV data for backtesting

- **`requirements.txt`**: Python dependencies
  - Install with: `pip install -r requirements.txt`

## Important Notes About Scaffolding

⚠️ **The scaffolding is provided as a starting point, but:**

1. **Some code may be incomplete or broken** - You may encounter errors that you'll need to debug and fix
2. **You're not required to use all scaffolding** - Feel free to modify, rewrite, or ignore parts as needed
3. **Data loading may need adjustment** - The data format and loading approach may require fixes but it should be fine for the most part
4. **Configuration may need changes** - Paths, parameters, or structure may need adjustment for your setup
5. **Optimization is NOT implemented** - You must implement your own optimization method from scratch

**Your ability to navigate around issues and adapt the scaffolding is part of the assessment.**

## Getting Started

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Download and Place Data Files

**Important**: Download the data files from the Google Drive link:

1. **Download data files from**: [Google Drive Link](https://drive.google.com/drive/folders/1A2W_LSuyhJuu1Vc-8tm-KidG-kxaPLQI?usp=sharing)
2. Download both `gc_1m.parquet` and `gc_1m.csv` files
3. Place the data files in the `data/` directory
4. Ensure the files are named exactly: `gc_1m.parquet` and `gc_1m.csv`
5. Verify the files are in the correct location: `data/gc_1m.parquet`

### 3. Verify Setup (Optional)

Run integration tests to verify everything is set up correctly:

```bash
# Basic integration tests (tests scaffolding structure)
python tests/test_integration.py

# Working implementation tests (tests with minimal algo logic)
python tests/test_working_implementation.py
```

Or with pytest:
```bash
pytest tests/test_integration.py -v
```

**Test Files:**
- `test_integration.py`: Tests that all scaffolding components can be imported and initialized
- `test_working_implementation.py`: Tests with a working RSI implementation to verify end-to-end functionality

These tests verify:
- Data file exists and is readable
- Configuration loads correctly
- Instruments can be created
- Data can be loaded and converted to bars
- Strategy can be initialized
- Backtest engine can be set up
- **Working implementation**: RSI calculation, strategy execution, trade generation
- Components work together end-to-end

**Note**: 
- `test_working_implementation.py` includes a minimal working RSI strategy implementation to prove everything works
- The actual `rsi_algo_template.py` still has TODOs for you to implement
- These tests help verify your setup and may reveal issues with the scaffolding that you'll need to fix

### 3. Explore the Codebase

- Read `strategies/rsi_algo_template.py` to understand the structure
- **Indicators**: Choose between Nautilus built-in indicators (recommended) or custom pandas-based indicators
  - Built-in: See https://nautilustrader.io/docs/latest/api_reference/indicators/
  - Custom: Check `strategies/indicators.py` for available helper functions
- Review `config/backtest_gc.yaml` for configuration options
- Examine `run_backtest.py` to understand how backtests are run

### 4. Implement the Strategy

Start with `strategies/rsi_algo_template.py`:

1. **TODO 1**: Compute RSI indicator from price data
2. **TODO 2**: Implement long entry logic (RSI crosses below entry threshold)
3. **TODO 3**: Implement pyramid logic (optional - adding to positions)
4. **TODO 4**: Implement exit logic (RSI crosses above exit threshold)
5. **BONUS**: Implement RSI divergence detection

### 5. Test Your Implementation

Try running a backtest:

```bash
python run_backtest.py
```

**If you encounter errors**, debug and fix them. This is expected and part of the challenge.

### 6. Optimize Parameters

Implement your optimization method in `optimize_params.py`:

- Choose an approach (grid search, random search, Bayesian optimization, etc.)
- Define parameter ranges to explore
- Run optimization to find best parameters
- Document your approach

### 7. Analyze Results

Review backtest performance:
- Sharpe ratio
- Total return
- Maximum drawdown
- Win rate
- Number of trades

## Strategy Logic Reference

The strategy should implement the following logic (from Pine Script RSI Algo V4):

- **Entry**: Enter long when RSI crosses below `long_entry` threshold (oversold condition)
- **Exit**: Exit long when RSI crosses above `long_exit` threshold (overbought condition)
- **Pyramid**: Optionally add to positions when conditions are met (if enabled)
- **Position Sizing**: Use `base_qty` for initial positions, respect `max_position` limit

## Data Format

The data file (`data/gc_1m.parquet`) contains:
- **timestamp**: Datetime index
- **open, high, low, close**: Price data (float)
- **volume**: Volume data (int)

Data is 1-minute bars for Gold Futures (GC) continuous contract.

## Nautilus Trader Documentation

Refer to the official documentation for API details:

- **Strategies**: https://nautilustrader.io/docs/latest/strategies/
- **Backtesting**: https://nautilustrader.io/docs/latest/backtesting/
- **Data**: https://nautilustrader.io/docs/latest/data/

## Troubleshooting

### Common Issues

**Issue**: Import errors or missing modules
- Solution: Ensure virtual environment is activated and dependencies are installed
- Check `requirements.txt` for all required packages

**Issue**: Data file not found
- Solution: **Download data files from** [Google Drive Link](https://drive.google.com/drive/folders/1A2W_LSuyhJuu1Vc-8tm-KidG-kxaPLQI?usp=sharing) and place them in the `data/` directory
- Verify `data/gc_1m.parquet` exists after downloading the data files
- Check path in `config/backtest_gc.yaml` is correct

**Issue**: Strategy not executing or no trades
- Solution: Debug your implementation
- Check that indicators are calculated correctly
  - If using built-in indicators: Ensure `indicator.initialized` is True before accessing values
  - If using custom indicators: Ensure you have enough data (len(prices) >= period)
- Verify entry/exit logic is working
- Review logs for errors

**Issue**: Optimization script errors
- Solution: You must implement the optimization method yourself
- The skeleton is provided but not functional

**Issue**: Configuration errors
- Solution: Review `config/backtest_gc.yaml` structure
- Adjust paths, parameters, or structure as needed

## Deliverables

Upon completion, provide:

1. **Completed strategy** (`strategies/rsi_algo_template.py`)
   - All TODOs implemented
   - Strategy executes without errors
   - Generates trades and produces results

2. **Optimization implementation** (`optimize_params.py`)
   - Your chosen optimization method implemented
   - Documentation of approach
   - Results with optimal parameters

3. **Backtest results**
   - Performance metrics with optimized parameters
   - Summary of findings

4. **Brief report** (optional but recommended)
   - Implementation approach
   - Key decisions and trade-offs
   - Results and analysis
   - Any issues encountered and how you resolved them

## Tips

- **Start simple**: Get basic entry/exit working before adding complexity
- **Test incrementally**: Verify each component works before moving on
- **Read the docs**: Nautilus Trader documentation is comprehensive
- **Debug systematically**: Use logging and print statements to understand execution flow
- **Don't be afraid to modify scaffolding**: Adapt code to your needs
- **Focus on correctness**: A simple working strategy is better than a complex broken one

## Questions?

If you have questions about:
- **Nautilus Trader API**: Refer to official documentation
- **Strategy logic**: Review the template comments and Pine Script reference
- **Data format**: Check the parquet file structure
- **Implementation approach**: Use your best judgment and adapt as needed

Good luck!
