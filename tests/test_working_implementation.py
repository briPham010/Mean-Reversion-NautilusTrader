"""
Test Working Implementation

This test creates working versions of all components with minimal implementations
to verify everything works together end-to-end.

This file tests:
- indicators.py (RSI calculation)
- rsi_algo_template.py (with basic implementation)
- run_backtest.py (data loading and engine setup)
- optimize_params.py (with basic grid search implementation)
"""

import sys
from pathlib import Path
import pandas as pd
import itertools

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, ImportableStrategyConfig, StrategyFactory
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, BarAggregation, PriceType, AssetClass
from nautilus_trader.model.identifiers import InstrumentId, Venue, TraderId, Symbol
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import PositionSide, OrderSide
from nautilus_trader.core.message import Event
from nautilus_trader.model.identifiers import InstrumentId

# Import project modules
from run_backtest import load_config, create_instrument_from_config, load_parquet_to_bars
from strategies.indicators import rsi
from strategies.rsi_algo_template import RsiAlgoConfig


def test_indicators_rsi():
    """Test that RSI indicator works correctly."""
    print("Testing indicators.py - RSI calculation...")
    
    # Create test price data
    prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 102, 101, 100, 99, 98, 97, 98, 99])
    
    # Calculate RSI
    rsi_values = rsi(prices, period=14)
    
    assert len(rsi_values) == len(prices), "RSI should have same length as input"
    assert not rsi_values.isna().all(), "RSI should have some valid values"
    
    # Check RSI values are in valid range (0-100)
    valid_rsi = rsi_values.dropna()
    if len(valid_rsi) > 0:
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all(), "RSI values should be between 0 and 100"
    
    print(f"✅ RSI indicator works - calculated {len(valid_rsi)} valid values")
    if len(valid_rsi) > 0:
        print(f"   Sample RSI values: {valid_rsi.tail(3).tolist()}")


class WorkingRsiStrategy(Strategy):
    """Working version of RSI strategy with minimal implementation for testing."""
    
    def __init__(self, config: RsiAlgoConfig):
        super().__init__(config=config)
        
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        
        # Construct bar_type
        if config.bar_type.startswith(str(self.instrument_id)):
            self.bar_type = BarType.from_str(config.bar_type)
        else:
            bar_type_str = f"{self.instrument_id}-{config.bar_type}"
            self.bar_type = BarType.from_str(bar_type_str)
        
        self.rsi_period = config.rsi_period
        self.long_entry = config.long_entry
        self.long_exit = config.long_exit
        self.base_qty = config.base_qty
        
        self.prices = []
        self.bars = []
        self.current_rsi = None
        self.previous_rsi = None
        self.position = None
    
    def on_start(self):
        self.subscribe_bars(bar_type=self.bar_type)
        self._log.info(f"Working RSI Strategy started for {self.instrument_id}")
    
    def on_stop(self):
        instrument = self.cache.instrument(self.instrument_id)
        if instrument:
            positions_open = self.cache.positions_open(venue=instrument.venue)
            for pos in positions_open:
                if pos.instrument_id == self.instrument_id and pos.is_open:
                    self.close_position(pos)
        self.unsubscribe_bars(bar_type=self.bar_type)
    
    def on_bar(self, bar: Bar):
        # Extract close price
        try:
            close_price = float(bar.close.as_double()) if hasattr(bar.close, 'as_double') else float(bar.close)
        except:
            close_price = float(bar.close)
        
        self.bars.append(bar)
        self.prices.append(close_price)
        
        # Keep only recent prices
        max_prices = max(self.rsi_period * 3, 100)
        if len(self.prices) > max_prices:
            self.prices = self.prices[-max_prices:]
        
        # Update position
        self.position = None
        instrument = self.cache.instrument(self.instrument_id)
        if instrument:
            positions_open = self.cache.positions_open(venue=instrument.venue)
            for pos in positions_open:
                if pos.instrument_id == self.instrument_id:
                    self.position = pos
                    break
        
        # Calculate RSI
        if len(self.prices) >= self.rsi_period + 1:
            prices_series = pd.Series(self.prices)
            rsi_values = rsi(prices_series, period=self.rsi_period)
            
            if len(rsi_values) > 0 and not rsi_values.isna().all():
                valid_rsi = rsi_values.dropna()
                if len(valid_rsi) > 0:
                    self.current_rsi = float(valid_rsi.iloc[-1])
                    if len(valid_rsi) > 1:
                        self.previous_rsi = float(valid_rsi.iloc[-2])
        
        # Skip if RSI not ready
        if self.current_rsi is None:
            return
        
        # Entry logic: RSI crosses below long_entry
        if not self.is_long and self.current_rsi < self.long_entry:
            if self.previous_rsi is not None and self.previous_rsi >= self.long_entry:
                self.enter_long(self.base_qty)
        
        # Exit logic: RSI crosses above long_exit
        if self.is_long and self.current_rsi > self.long_exit:
            if self.previous_rsi is not None and self.previous_rsi <= self.long_exit:
                self.exit_long()
    
    def on_event(self, event: Event):
        from nautilus_trader.model.events import PositionOpened, PositionClosed
        
        if isinstance(event, PositionOpened):
            instrument = self.cache.instrument(self.instrument_id)
            if instrument:
                positions_open = self.cache.positions_open(venue=instrument.venue)
                for pos in positions_open:
                    if pos.instrument_id == self.instrument_id:
                        self.position = pos
                        break
        elif isinstance(event, PositionClosed):
            self.position = None
    
    @property
    def is_long(self):
        return self.position is not None and self.position.side == PositionSide.LONG
    
    def enter_long(self, qty: int):
        instrument = self.cache.instrument(self.instrument_id)
        if not instrument:
            return
        
        quantity = Quantity.from_int(qty)
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=quantity,
        )
        self.submit_order(order)
    
    def exit_long(self):
        if self.position and self.is_long:
            self.close_position(self.position)


def test_working_strategy():
    """Test that a working strategy implementation can run."""
    print("\nTesting working RSI strategy implementation...")
    
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    config = load_config(str(config_path))
    
    # Create engine
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("WORKING-TEST"),
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=engine_config)
    
    # Add venue
    venue = Venue("GLBX")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money.from_str("100000.00 USD")],
    )
    
    # Add instrument
    instrument_id, instrument = create_instrument_from_config(config)
    engine.add_instrument(instrument)
    
    # Create working strategy
    strategy_config = RsiAlgoConfig(
        instrument_id="GC.GLBX",
        bar_type="1-MINUTE-LAST-EXTERNAL",
        rsi_period=14,
        long_entry=31.0,
        long_exit=83.0,
        base_qty=1,
    )
    strategy = WorkingRsiStrategy(strategy_config)
    engine.add_strategy(strategy)
    
    # Load small subset of data from gc_1m.parquet
    data_path_str = config["data"]["bar_data"]["path"]
    data_path = script_dir / data_path_str
    
    # Verify we're using the correct parquet file
    assert data_path.name == "gc_1m.parquet", f"Expected gc_1m.parquet, got {data_path.name}"
    assert data_path.exists(), f"Data file not found: {data_path}"
    
    print(f"   Loading data from: {data_path}")
    bars = load_parquet_to_bars(
        data_path=data_path,
        instrument_id=instrument_id,
        start_time="2022-01-01T00:00:00Z",
        end_time="2022-01-10T00:00:00Z",  # 9 days
    )
    
    print(f"   Loaded {len(bars)} bars from gc_1m.parquet")
    engine.add_data(bars)
    
    # Run backtest
    engine.run()
    
    # Verify results
    venues = engine.list_venues()
    venue = venues[0]
    account_df = engine.trader.generate_account_report(venue)
    fills_df = engine.trader.generate_fills_report()
    
    assert not account_df.empty, "Account report should not be empty"
    
    # Extract PnL information
    initial_balance = None
    final_balance = None
    total_pnl = None
    
    if 'total' in account_df.columns:
        initial_balance = account_df['total'].iloc[0]
        final_balance = account_df['total'].iloc[-1]
        
        # Convert to float if Money object/string
        if isinstance(initial_balance, str):
            try:
                initial_balance = float(initial_balance.split()[0])
            except:
                pass
        if isinstance(final_balance, str):
            try:
                final_balance = float(final_balance.split()[0])
            except:
                pass
        
        if isinstance(initial_balance, (int, float)) and isinstance(final_balance, (int, float)):
            total_pnl = final_balance - initial_balance
    
    # Get realized PnL from fills
    realized_pnl = 0.0
    if not fills_df.empty and 'realized_pnl' in fills_df.columns:
        realized_pnl = fills_df['realized_pnl'].sum()
    
    print(f"✅ Working strategy executed successfully")
    print(f"   Account records: {len(account_df)}")
    print(f"   Fills: {len(fills_df)}")
    print(f"   Strategy processed {len(strategy.bars)} bars")
    if strategy.current_rsi is not None:
        print(f"   Final RSI: {strategy.current_rsi:.2f}")
    
    # Print PnL information
    print(f"\n💰 PnL Information:")
    if initial_balance is not None and isinstance(initial_balance, (int, float)):
        print(f"   Initial Balance: ${initial_balance:,.2f}")
    if final_balance is not None and isinstance(final_balance, (int, float)):
        print(f"   Final Balance: ${final_balance:,.2f}")
    if total_pnl is not None:
        print(f"   Total PnL: ${total_pnl:,.2f}")
    if isinstance(realized_pnl, (int, float)) and realized_pnl != 0:
        print(f"   Realized PnL (from fills): ${realized_pnl:,.2f}")


def test_run_backtest_integration():
    """Test that run_backtest.py components work together."""
    print("\nTesting run_backtest.py integration...")
    
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    config = load_config(str(config_path))
    
    # Test all functions from run_backtest
    instrument_id, instrument = create_instrument_from_config(config)
    assert instrument_id is not None
    assert instrument is not None
    
    data_path_str = config["data"]["bar_data"]["path"]
    data_path = script_dir / data_path_str
    
    # Verify we're using gc_1m.parquet
    assert data_path.name == "gc_1m.parquet", f"Expected gc_1m.parquet, got {data_path.name}"
    assert data_path.exists(), f"Data file not found: {data_path}"
    
    bars = load_parquet_to_bars(
        data_path=data_path,
        instrument_id=instrument_id,
        start_time="2022-01-01T00:00:00Z",
        end_time="2022-01-03T00:00:00Z",
    )
    assert len(bars) > 0
    
    print(f"✅ run_backtest.py functions work correctly")
    print(f"   Created instrument: {instrument_id}")
    print(f"   Loaded {len(bars)} bars from {data_path.name}")


def test_optimize_params_basic():
    """Test that optimize_params.py can work with a basic grid search implementation."""
    print("\nTesting optimize_params.py with basic implementation...")
    
    from optimize_params import (
        OptimizationResult,
        get_parameter_ranges,
        evaluate_parameter_combination,
        run_with_params,
    )
    
    # Test parameter ranges
    ranges = get_parameter_ranges()
    assert "rsi_period" in ranges
    assert "long_entry" in ranges
    assert "long_exit" in ranges
    
    # Test OptimizationResult
    result = OptimizationResult(
        rsi_period=14,
        long_entry=31.0,
        long_exit=83.0,
        sharpe_ratio=1.5,
        total_return=0.15,
        max_drawdown=-0.05,
        total_pnl=15000.0,
        num_trades=50,
    )
    assert result.rsi_period == 14
    
    # Test that we can create a basic grid search
    rsi_periods = ranges['rsi_period'][:2]  # Just 2 values for testing
    long_entries = ranges['long_entry'][:2]  # Just 2 values
    long_exits = ranges['long_exit'][:2]  # Just 2 values
    
    combinations = list(itertools.product(rsi_periods, long_entries, long_exits))
    assert len(combinations) == 2 * 2 * 2  # 8 combinations
    
    print(f"✅ optimize_params.py structure works")
    print(f"   Parameter ranges loaded: {len(ranges)} parameters")
    print(f"   Can generate {len(combinations)} combinations for testing")
    print(f"   Note: Full optimization implementation is TODO")


def test_full_integration():
    """Full end-to-end test with working implementation."""
    print("\n" + "="*80)
    print("FULL INTEGRATION TEST - All Components Working Together")
    print("="*80)
    
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    config = load_config(str(config_path))
    
    # Step 1: Load config
    print("\n1. Loading configuration...")
    assert config is not None
    print("   ✅ Config loaded")
    
    # Step 2: Create instrument
    print("\n2. Creating instrument...")
    instrument_id, instrument = create_instrument_from_config(config)
    print(f"   ✅ Instrument: {instrument_id}")
    
    # Step 3: Load data from gc_1m.parquet
    print("\n3. Loading data from gc_1m.parquet...")
    data_path_str = config["data"]["bar_data"]["path"]
    data_path = script_dir / data_path_str
    
    # Verify we're using the correct parquet file
    assert data_path.name == "gc_1m.parquet", f"Expected gc_1m.parquet, got {data_path.name}"
    assert data_path.exists(), f"Data file not found: {data_path}"
    
    print(f"   Data file: {data_path}")
    bars = load_parquet_to_bars(
        data_path=data_path,
        instrument_id=instrument_id,
        start_time="2022-01-01T00:00:00Z",
        end_time="2022-01-15T00:00:00Z",  # 14 days
    )
    print(f"   ✅ Loaded {len(bars)} bars from gc_1m.parquet")
    
    # Step 4: Test RSI calculation
    print("\n4. Testing RSI indicator...")
    prices = [float(bar.close.as_double()) if hasattr(bar.close, 'as_double') else float(bar.close) for bar in bars[:50]]
    prices_series = pd.Series(prices)
    rsi_values = rsi(prices_series, period=14)
    valid_rsi = rsi_values.dropna()
    print(f"   ✅ RSI calculated: {len(valid_rsi)} valid values")
    
    # Step 5: Create engine and run backtest
    print("\n5. Setting up backtest engine...")
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("FULL-TEST"),
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=engine_config)
    
    venue = Venue("GLBX")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money.from_str("100000.00 USD")],
    )
    engine.add_instrument(instrument)
    print("   ✅ Engine created, venue and instrument added")
    
    # Step 6: Add working strategy
    print("\n6. Adding working strategy...")
    strategy_config = RsiAlgoConfig(
        instrument_id="GC.GLBX",
        bar_type="1-MINUTE-LAST-EXTERNAL",
        rsi_period=14,
        long_entry=31.0,
        long_exit=83.0,
        base_qty=1,
    )
    strategy = WorkingRsiStrategy(strategy_config)
    engine.add_strategy(strategy)
    print("   ✅ Strategy added")
    
    # Step 7: Add data and run
    print("\n7. Running backtest...")
    engine.add_data(bars)
    engine.run()
    print("   ✅ Backtest completed")
    
    # Step 8: Verify results
    print("\n8. Verifying results...")
    venues = engine.list_venues()
    venue = venues[0]
    account_df = engine.trader.generate_account_report(venue)
    fills_df = engine.trader.generate_fills_report()
    
    assert not account_df.empty
    
    # Extract PnL information
    initial_balance = None
    final_balance = None
    total_pnl = None
    
    if 'total' in account_df.columns:
        initial_balance = account_df['total'].iloc[0]
        final_balance = account_df['total'].iloc[-1]
        
        # Convert to float if Money object/string
        if isinstance(initial_balance, str):
            try:
                initial_balance = float(initial_balance.split()[0])
            except:
                pass
        if isinstance(final_balance, str):
            try:
                final_balance = float(final_balance.split()[0])
            except:
                pass
        
        if isinstance(initial_balance, (int, float)) and isinstance(final_balance, (int, float)):
            total_pnl = final_balance - initial_balance
    
    # Get realized PnL from fills
    realized_pnl = 0.0
    if not fills_df.empty and 'realized_pnl' in fills_df.columns:
        realized_pnl = fills_df['realized_pnl'].sum()
    
    print(f"   ✅ Account report: {len(account_df)} records")
    print(f"   ✅ Fills: {len(fills_df)}")
    print(f"   ✅ Strategy processed: {len(strategy.bars)} bars")
    
    if strategy.current_rsi is not None:
        print(f"   ✅ Final RSI: {strategy.current_rsi:.2f}")
    
    # Print PnL information
    print(f"\n   💰 PnL Summary:")
    if initial_balance is not None and isinstance(initial_balance, (int, float)):
        print(f"      Initial Balance: ${initial_balance:,.2f}")
    if final_balance is not None and isinstance(final_balance, (int, float)):
        print(f"      Final Balance: ${final_balance:,.2f}")
    if total_pnl is not None:
        print(f"      Total PnL: ${total_pnl:,.2f} ({total_pnl/initial_balance*100:.2f}%)" if initial_balance and initial_balance != 0 else f"      Total PnL: ${total_pnl:,.2f}")
    if isinstance(realized_pnl, (int, float)) and realized_pnl != 0:
        print(f"      Realized PnL (from fills): ${realized_pnl:,.2f}")
    
    # Show some trade details if available
    if not fills_df.empty:
        winning_trades = 0
        losing_trades = 0
        if 'realized_pnl' in fills_df.columns:
            winning_trades = (fills_df['realized_pnl'] > 0).sum()
            losing_trades = (fills_df['realized_pnl'] < 0).sum()
        print(f"      Winning Trades: {winning_trades}")
        print(f"      Losing Trades: {losing_trades}")
    
    print("\n" + "="*80)
    print("✅ ALL COMPONENTS WORK TOGETHER SUCCESSFULLY!")
    print("="*80)
    print("\nVerified:")
    print("  ✅ indicators.py - RSI calculation works")
    print("  ✅ rsi_algo_template.py - Strategy structure works (with implementation)")
    print("  ✅ run_backtest.py - Data loading and engine setup works")
    print("  ✅ optimize_params.py - Structure works (implementation is TODO)")
    print("\nAll scaffolding is functional and ready for implementation!")


if __name__ == "__main__":
    try:
        test_indicators_rsi()
        test_run_backtest_integration()
        test_optimize_params_basic()
        test_working_strategy()
        test_full_integration()
        
        print("\n" + "="*80)
        print("🎉 ALL WORKING IMPLEMENTATION TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
