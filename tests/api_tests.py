import os

import pytest
import requests
import polars as pl

from algorithmic_trading.helper import Helper   
from algorithmic_trading.data_manager import DataManager
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

@pytest.mark.parametrize(
    "request_url,df_schema,raises,expected",
    [
        # Valid cases
        (f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-01-09/2024-01-12?adjusted=true&sort=asc&limit=120&apiKey={POLYGON_API_KEY}", 
         None, None, pl.read_parquet("tests/Test-Data/get_paginated_request.parquet")), # Valid request URL with no schema
        (f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-01-09/2024-01-12?adjusted=true&sort=asc&limit=120&apiKey={POLYGON_API_KEY}", 
         ["v", "vw", "o", "c", "h", "l", "t", "n"], 
         None, pl.read_parquet("tests/Test-Data/get_paginated_request.parquet")), # Valid request URL with schema
        # Exception cases
        (f"https://api.polygon.io/v1/last/stocks/INVALID?apiKey={POLYGON_API_KEY}", 
         None, ValueError, None), # Invalid ticker
        (f"https://api.polygon.io/v1/last/stocks/AAPL?apiKey={POLYGON_API_KEY}", 
         {"ticker": str, "last": float},
         ValueError, None) # Invalid schema for the response
    ]
)
def test_get_paginated_request(request_url, df_schema, raises, expected):
    if raises:
        with pytest.raises(raises):
            Helper.get_paginated_request(request_url=request_url, df_schema=df_schema)
    else:
        result = Helper.get_paginated_request(request_url=request_url, df_schema=df_schema)
        assert result.equals(expected) 


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
# if __name__ == '__main__':
#     data = Helper.get_paginated_request(f"https://api.polygon.io/v3/reference/tickers?type=WARRANT&market=stocks&date=2025-05-02&active=true&order=asc&limit=1000&sort=ticker&apiKey=QEjL21NHSpWuFK4EC4_Bej76tZSwrU69", df_schema={'ticker': str, 'name': str, 'market': str, 'locale': str, 'primary_exchange': str, 'type': str, 'active': bool, 'currency_name': str, 'cik': str, 'last_updated_utc': str, 'delisted_utc': str, 'composite_figi': str, 'share_class_figi': str})
#     data.write_parquet("tests/Test-Data/get_historical_tickers.parquet")