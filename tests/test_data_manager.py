import pytest
from algorithmic_trading.data_manager import DataManager
import polars as pl
import datetime

@pytest.mark.parametrize(
    "ticker_types,include_delisted,date,raises,expected",
    [
        # Valid cases
        (['WARRANT'], False, '2025-05-02', None, pl.read_parquet("tests/Test-Data/get_historical_tickers.parquet")), # Valid ticker type without delisted tickers
        (['RIGHT'], True, '', None, None), # Without date arg, just testing for a successful response
        # Exception cases
        (123, True, '2025-05-02', TypeError, None), # Invalid ticker type
        (['WARRANT'], 'yes', '2025-05-02', TypeError, None), # Invalid include_delisted type
        (['WARRANT'], True, 'invalid-date', ValueError, None) # Invalid date format         
    ]
)
def test_get_historical_tickers(ticker_types, include_delisted, date, raises, expected):
    if raises:
        with pytest.raises(raises):
            DataManager.get_historical_tickers(ticker_types=ticker_types, include_delisted=include_delisted, date=date)
    else:
        result = DataManager.get_historical_tickers(ticker_types=ticker_types, include_delisted=include_delisted, date=date)
        if expected is not None:
            assert result.equals(expected)
        else:
            assert isinstance(result, pl.DataFrame)
            assert result.height > 0

@pytest.mark.parametrize(
    "start_date, end_date, tickers, timeframe, multiplier, raises, expected",
    [
        # Valid cases
        ('2024-08-01', '2024-08-22', ['AAPL', 'GOOG'], 'day', 1, None, pl.read_parquet("tests/Test-Data/get_historical_ohlc_day.parquet")), # Multiple tickers, daily data
        ('2025-08-01', '2025-08-02', 'TSLA', 'minute', 5, None, pl.read_parquet("tests/Test-Data/get_historical_ohlc_5_min.parquet")), #Single ticker with minute timeframe and multiplier
        #Invalid cases
        ('2025-08-01', '2025-07-31', ['AAPL', 'GOOG'], 'day', 1, ValueError, None), #Start date after end date
        ('2025-08-01', '2025-08-22', ['AAPL', 'GOOG'], 'invalid-timeframe', 1, ValueError, None), #Invalid timeframe
        ('2025-08-01', '2025-08-22', ['AAPL', 'GOOG'], 'day', -1, ValueError, None), #Invalid multiplier
        ('2025-08-01', '2025-08-22', 123, 'day', 1, TypeError, None) #Invalid tickers type 
    ]   
)       
def test_get_historical_ohlc(start_date, end_date, tickers, timeframe, multiplier, raises, expected):
    if raises:
        with pytest.raises(raises):
            DataManager.get_historical_ohlc(start_date=start_date, end_date=end_date, tickers=tickers, timeframe=timeframe, multiplier=multiplier)
    else:
        result = DataManager.get_historical_ohlc(start_date=start_date, end_date=end_date, tickers=tickers, timeframe=timeframe, multiplier=multiplier)
        assert isinstance(result, pl.DataFrame)
        assert result.height > 0
        if expected is not None:
            try:
                assert result.equals(expected)
            except AssertionError:
                print(result.head(result.height))
                assert result.equals(expected)
