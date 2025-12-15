"""
Integration Tests for RSI Algorithm Take-Home

This test suite verifies that all components work together:
- Data loading from parquet
- Instrument creation
- Strategy initialization
- Backtest engine setup
- Basic strategy execution
- Configuration loading

Run with:
    pytest tests/test_integration.py -v
    or
    python -m pytest tests/test_integration.py -v
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, BarAggregation, PriceType, AssetClass
from nautilus_trader.model.identifiers import InstrumentId, Venue, TraderId, Symbol
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.data import Bar, BarType, BarSpecification

# Import project modules
from run_backtest import (
    load_config,
    create_instrument_from_config,
    load_parquet_to_bars,
    setup_backtest_engine,
)
from strategies.rsi_algo_template import RsiAlgoStrategy, RsiAlgoConfig


def test_data_file_exists():
    """Test that the data file exists and is readable."""
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / "data" / "gc_1m.parquet"
    
    assert data_path.exists(), f"Data file not found at {data_path}"
    
    # Try to read the file
    df = pd.read_parquet(data_path)
    assert len(df) > 0, "Data file is empty"
    assert "timestamp" in df.columns, "Missing timestamp column"
    assert "open" in df.columns, "Missing open column"
    assert "high" in df.columns, "Missing high column"
    assert "low" in df.columns, "Missing low column"
    assert "close" in df.columns, "Missing close column"
    assert "volume" in df.columns, "Missing volume column"
    
    print(f"✅ Data file exists and has {len(df):,} rows")


def test_config_loading():
    """Test that configuration file can be loaded."""
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    
    assert config_path.exists(), f"Config file not found at {config_path}"
    
    config = load_config(str(config_path))
    
    # Verify required sections exist
    assert "engine" in config, "Missing 'engine' section in config"
    assert "venue" in config, "Missing 'venue' section in config"
    assert "data" in config, "Missing 'data' section in config"
    assert "strategies" in config["engine"], "Missing 'strategies' in engine config"
    assert len(config["engine"]["strategies"]) > 0, "No strategies defined"
    
    print("✅ Configuration file loaded successfully")


def test_instrument_creation():
    """Test that instrument can be created from config."""
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    config = load_config(str(config_path))
    
    instrument_id, instrument = create_instrument_from_config(config)
    
    assert instrument_id is not None, "Instrument ID is None"
    assert instrument is not None, "Instrument is None"
    assert isinstance(instrument, FuturesContract), "Instrument is not a FuturesContract"
    assert str(instrument_id) == "GC.GLBX", f"Unexpected instrument ID: {instrument_id}"
    
    print(f"✅ Instrument created: {instrument_id}")


def test_data_loading():
    """Test that data can be loaded from parquet and converted to bars."""
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    config = load_config(str(config_path))
    
    instrument_id, _ = create_instrument_from_config(config)
    data_path = script_dir / config["data"]["bar_data"]["path"]

    # Choose a window that is guaranteed to exist in the dataset
    df = pd.read_parquet(data_path)
    start_ts = df["timestamp"].min()
    end_ts = start_ts + pd.Timedelta(days=1)

    bars = load_parquet_to_bars(
        data_path=data_path,
        instrument_id=instrument_id,
        start_time=start_ts.isoformat(),
        end_time=end_ts.isoformat(),
    )
    
    assert len(bars) > 0, "No bars loaded"
    assert all(isinstance(bar, Bar) for bar in bars), "Not all items are Bar objects"
    
    # Verify bar structure
    first_bar = bars[0]
    assert first_bar.bar_type is not None, "Bar missing bar_type"
    assert first_bar.open is not None, "Bar missing open price"
    assert first_bar.close is not None, "Bar missing close price"
    assert first_bar.high is not None, "Bar missing high price"
    assert first_bar.low is not None, "Bar missing low price"
    assert first_bar.volume is not None, "Bar missing volume"
    
    print(f"✅ Loaded {len(bars)} bars from parquet file")


def test_strategy_initialization():
    """Test that strategy can be initialized with config."""
    config = RsiAlgoConfig(
        instrument_id="GC.GLBX",
        bar_type="1-MINUTE-LAST-EXTERNAL",
        rsi_period=14,
        long_entry=31.0,
        long_exit=83.0,
        base_qty=2,
        enable_pyramid=True,
        max_position=6,
    )
    
    strategy = RsiAlgoStrategy(config=config)
    
    assert strategy is not None, "Strategy is None"
    assert strategy.instrument_id is not None, "Strategy missing instrument_id"
    assert strategy.bar_type is not None, "Strategy missing bar_type"
    assert strategy.rsi_period == 14, "RSI period not set correctly"
    assert strategy.long_entry == 31.0, "Long entry not set correctly"
    assert strategy.long_exit == 83.0, "Long exit not set correctly"
    
    print("✅ Strategy initialized successfully")


def test_backtest_engine_setup():
    """Test that backtest engine can be set up with all components."""
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    config = load_config(str(config_path))
    
    # Create engine with minimal data for testing
    engine = setup_backtest_engine(config)
    
    assert engine is not None, "Engine is None"
    
    # Verify venue was added
    venues = engine.list_venues()
    assert len(venues) > 0, "No venues added to engine"
    
    # Verify instrument was added
    instrument_id, _ = create_instrument_from_config(config)
    instrument = engine.cache.instrument(instrument_id)
    assert instrument is not None, "Instrument not found in cache"
    
    # Verify strategy was added
    strategies = engine.trader.strategies()
    assert len(strategies) > 0, "No strategies added to engine"
    
    print("✅ Backtest engine set up successfully")


def test_simple_backtest_execution():
    """Test that a simple backtest can run end-to-end."""
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    config = load_config(str(config_path))
    
    # Modify config to use a small date range for faster testing
    config["data"]["bar_data"]["start_time"] = "2022-01-01T00:00:00Z"
    config["data"]["bar_data"]["end_time"] = "2022-01-05T00:00:00Z"  # Just 4 days
    
    # Set up engine
    engine = setup_backtest_engine(config)
    
    # Run backtest
    engine.run()
    
    # Verify backtest completed
    venues = engine.list_venues()
    assert len(venues) > 0, "No venues after backtest"
    
    # Check that we can generate reports
    venue = venues[0]
    account_df = engine.trader.generate_account_report(venue)
    assert not account_df.empty, "Account report is empty"
    
    fills_df = engine.trader.generate_fills_report()
    # Fills may be empty if strategy doesn't trade, which is OK
    
    print("✅ Backtest executed successfully")
    print(f"   Account report: {len(account_df)} rows")
    print(f"   Fills report: {len(fills_df)} fills")


def test_strategy_bar_processing():
    """Test that strategy can process bars correctly."""
    # Create a minimal strategy config
    config = RsiAlgoConfig(
        instrument_id="GC.GLBX",
        bar_type="1-MINUTE-LAST-EXTERNAL",
        rsi_period=14,
        long_entry=31.0,
        long_exit=83.0,
        base_qty=1,
    )
    
    strategy = RsiAlgoStrategy(config=config)
    
    # Create a simple backtest engine
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("TEST-001"),
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
    instrument_id = InstrumentId.from_str("GC.GLBX")
    expiration_ns = int(pd.Timestamp("2099-12-31T23:59:59Z", tz="UTC").timestamp() * 1_000_000_000)
    instrument = FuturesContract(
        instrument_id=instrument_id,
        raw_symbol=Symbol("GC"),
        asset_class=AssetClass.COMMODITY,
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.10"),
        multiplier=Quantity.from_int(100),
        lot_size=Quantity.from_int(1),
        underlying="GC",
        activation_ns=0,
        expiration_ns=expiration_ns,
        ts_event=0,
        ts_init=0,
    )
    engine.add_instrument(instrument)
    
    # Add strategy
    engine.add_strategy(strategy)
    
    # Create a few test bars
    bar_spec = BarSpecification(step=1, aggregation=BarAggregation.MINUTE, price_type=PriceType.LAST)
    bar_type = BarType(instrument_id, bar_spec)
    
    test_bars = []
    base_price = 1800.0
    for i in range(30):  # 30 bars to have enough for RSI calculation
        ts_ns = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1_000_000_000) + (i * 60 * 1_000_000_000)
        price = base_price + (i * 0.1)  # Slight upward trend
        
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{price:.2f}"),
            high=Price.from_str(f"{price + 0.5:.2f}"),
            low=Price.from_str(f"{price - 0.5:.2f}"),
            close=Price.from_str(f"{price:.2f}"),
            volume=Quantity.from_int(100),
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        test_bars.append(bar)
    
    engine.add_data(test_bars)
    
    # Run backtest
    engine.run()
    
    # Verify strategy processed bars
    # Strategy should have received bars (even if it didn't trade)
    assert len(strategy.prices) > 0, "Strategy did not process any bars"
    assert len(strategy.bars) > 0, "Strategy did not store any bars"
    
    print(f"✅ Strategy processed {len(strategy.bars)} bars")


def test_optimize_params_imports():
    """Test that optimize_params.py can be imported and basic functions work."""
    from optimize_params import (
        OptimizationResult,
        get_parameter_ranges,
        evaluate_parameter_combination,
        objective_function,
    )
    
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
    assert result.rsi_period == 14, "OptimizationResult not working"
    
    # Test get_parameter_ranges
    ranges = get_parameter_ranges()
    assert "rsi_period" in ranges, "Missing rsi_period in ranges"
    assert "long_entry" in ranges, "Missing long_entry in ranges"
    assert "long_exit" in ranges, "Missing long_exit in ranges"
    assert len(ranges["rsi_period"]) > 0, "rsi_period range is empty"
    
    # Test objective_function
    obj_value = objective_function(result)
    assert isinstance(obj_value, (int, float)), "Objective function should return number"
    
    print("✅ Optimization module imports and basic functions work")


def test_end_to_end_integration():
    """Comprehensive end-to-end test of all components working together."""
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    
    # Load config
    config = load_config(str(config_path))
    print("✅ Step 1: Config loaded")
    
    # Create instrument
    instrument_id, instrument = create_instrument_from_config(config)
    print(f"✅ Step 2: Instrument created: {instrument_id}")
    
    # Load data (small subset)
    data_path = script_dir / config["data"]["bar_data"]["path"]
    bars = load_parquet_to_bars(
        data_path=data_path,
        instrument_id=instrument_id,
        start_time="2022-01-01T00:00:00Z",
        end_time="2022-01-10T00:00:00Z",  # 9 days for testing
    )
    print(f"✅ Step 3: Loaded {len(bars)} bars")
    
    # Create engine
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("INTEGRATION-TEST"),
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=engine_config)
    
    # Add venue
    venue = Venue(config["venue"]["name"])
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money.from_str(bal) for bal in config["venue"]["starting_balances"]],
    )
    print("✅ Step 4: Venue added")
    
    # Add instrument
    engine.add_instrument(instrument)
    print("✅ Step 5: Instrument added")
    
    # Create and add strategy
    strategy_config_dict = config["engine"]["strategies"][0]
    from nautilus_trader.config import ImportableStrategyConfig, StrategyFactory
    
    strategy_config = ImportableStrategyConfig(
        strategy_path=f"{strategy_config_dict['module']}:{strategy_config_dict['class']}",
        config_path=f"{strategy_config_dict['module']}:RsiAlgoConfig",
        config=strategy_config_dict["config"],
    )
    strategy = StrategyFactory.create(strategy_config)
    engine.add_strategy(strategy)
    print("✅ Step 6: Strategy added")
    
    # Add data
    engine.add_data(bars)
    print(f"✅ Step 7: {len(bars)} bars added to engine")
    
    # Run backtest
    engine.run()
    print("✅ Step 8: Backtest executed")
    
    # Verify results
    venues = engine.list_venues()
    venue = venues[0]
    account_df = engine.trader.generate_account_report(venue)
    assert not account_df.empty, "Account report should not be empty"
    
    fills_df = engine.trader.generate_fills_report()
    
    print(f"✅ Step 9: Results generated")
    print(f"   Account records: {len(account_df)}")
    print(f"   Fills: {len(fills_df)}")
    
    print("\n🎉 All integration tests passed! All components work together.")


def test_integration_with_working_strategy():
    """Test integration using WorkingRsiStrategy from test_working_implementation.py."""
    from tests.test_working_implementation import WorkingRsiStrategy
    
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "backtest_gc.yaml"
    config = load_config(str(config_path))
    
    # Modify config to use a small date range for faster testing
    config["data"]["bar_data"]["start_time"] = "2022-01-01T00:00:00Z"
    config["data"]["bar_data"]["end_time"] = "2022-01-10T00:00:00Z"  # 9 days
    
    print("Testing integration with WorkingRsiStrategy...")
    
    # Create instrument
    instrument_id, instrument = create_instrument_from_config(config)
    print(f"✅ Instrument created: {instrument_id}")
    
    # Load data
    data_path = script_dir / config["data"]["bar_data"]["path"]
    bars = load_parquet_to_bars(
        data_path=data_path,
        instrument_id=instrument_id,
        start_time=config["data"]["bar_data"]["start_time"],
        end_time=config["data"]["bar_data"]["end_time"],
    )
    print(f"✅ Loaded {len(bars)} bars")
    
    # Create engine
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("WORKING-INTEGRATION-TEST"),
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=engine_config)
    
    # Add venue
    venue = Venue(config["venue"]["name"])
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money.from_str(bal) for bal in config["venue"]["starting_balances"]],
    )
    print("✅ Venue added")
    
    # Add instrument
    engine.add_instrument(instrument)
    print("✅ Instrument added")
    
    # Create and add WorkingRsiStrategy
    strategy_config = RsiAlgoConfig(
        instrument_id=str(instrument_id),
        bar_type="1-MINUTE-LAST-EXTERNAL",
        rsi_period=14,
        long_entry=31.0,
        long_exit=83.0,
        base_qty=2,
    )
    strategy = WorkingRsiStrategy(config=strategy_config)
    engine.add_strategy(strategy)
    print("✅ WorkingRsiStrategy added")
    
    # Add data
    engine.add_data(bars)
    print(f"✅ {len(bars)} bars added to engine")
    
    # Run backtest
    engine.run()
    print("✅ Backtest executed")
    
    # Verify results
    venues = engine.list_venues()
    venue = venues[0]
    account_df = engine.trader.generate_account_report(venue)
    assert not account_df.empty, "Account report should not be empty"
    
    fills_df = engine.trader.generate_fills_report()
    
    # Extract PnL information
    if not account_df.empty:
        # Account report uses 'total' column, not 'balance'
        initial_balance = account_df.iloc[0]['total']
        final_balance = account_df.iloc[-1]['total']
        
        # Convert to float if needed
        if hasattr(initial_balance, 'as_double'):
            initial_balance = float(initial_balance.as_double())
        elif isinstance(initial_balance, str):
            initial_balance = float(initial_balance.replace(' USD', '').replace(',', ''))
        else:
            initial_balance = float(initial_balance)
            
        if hasattr(final_balance, 'as_double'):
            final_balance = float(final_balance.as_double())
        elif isinstance(final_balance, str):
            final_balance = float(final_balance.replace(' USD', '').replace(',', ''))
        else:
            final_balance = float(final_balance)
        
        total_pnl = final_balance - initial_balance
        pnl_pct = (total_pnl / initial_balance) * 100
        
        print(f"\n💰 PnL Summary:")
        print(f"   Initial Balance: ${initial_balance:,.2f}")
        print(f"   Final Balance: ${final_balance:,.2f}")
        print(f"   Total PnL: ${total_pnl:,.2f} ({pnl_pct:.2f}%)")
    
    print(f"✅ Results generated")
    print(f"   Account records: {len(account_df)}")
    print(f"   Fills: {len(fills_df)}")
    
    # Verify strategy processed bars
    assert len(strategy.bars) > 0, "Strategy should have processed bars"
    assert len(strategy.prices) > 0, "Strategy should have stored prices"
    
    # Verify RSI was calculated
    assert strategy.current_rsi is not None, "Strategy should have calculated RSI"
    assert 0 <= strategy.current_rsi <= 100, "RSI should be between 0 and 100"
    
    print(f"\n✅ Strategy processed {len(strategy.bars)} bars")
    print(f"   Current RSI: {strategy.current_rsi:.2f}")
    prev_rsi_str = f"{strategy.previous_rsi:.2f}" if strategy.previous_rsi is not None else "N/A"
    print(f"   Previous RSI: {prev_rsi_str}")
    
    print("\n🎉 Integration test with WorkingRsiStrategy passed!")


if __name__ == "__main__":
    print("=" * 80)
    print("Running Integration Tests")
    print("=" * 80)
    print()
    
    try:
        test_data_file_exists()
        print()
        
        test_config_loading()
        print()
        
        test_instrument_creation()
        print()
        
        test_data_loading()
        print()
        
        test_strategy_initialization()
        print()
        
        test_backtest_engine_setup()
        print()
        
        test_strategy_bar_processing()
        print()
        
        test_optimize_params_imports()
        print()
        
        print("=" * 80)
        print("Running Comprehensive End-to-End Test")
        print("=" * 80)
        print()
        test_end_to_end_integration()
        print()
        
        print("=" * 80)
        print("Testing Integration with WorkingRsiStrategy")
        print("=" * 80)
        print()
        test_integration_with_working_strategy()
        print()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
