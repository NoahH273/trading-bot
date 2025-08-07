import datetime
import pytest
from algorithmic_trading.helper import Helper
import numpy as np
import os
import polars as pl
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
    "date,fmt,tz,expected,raises",
    [
        # Valid cases
        (datetime.date(2020, 5, 1), 'isoformat', datetime.timezone.utc, '2020-05-01T00:00:00+00:00', None), #Date to ISO format
        ('2021-12-25T05:12:06+00:00', 'datetime', datetime.timezone.utc, datetime.datetime(2021, 12, 25, 5, 12, 6, tzinfo=datetime.timezone.utc), None), #ISO format to datetime
        (datetime.datetime(2021, 10, 10), 'datetime', datetime.timezone.utc, datetime.datetime(2021, 10, 10, tzinfo=datetime.timezone.utc), None), #Datetime to datetime 
        (1658105431.54, 'isoformat', datetime.timezone(datetime.timedelta(hours=-7)), '2022-07-17T17:50:31.540000-07:00', None), #Float unix timestamp to ISO format
        (1620814598, 'datetime', datetime.timezone.utc, datetime.datetime(2021, 5, 12, 3, 16, 38, tzinfo=datetime.timezone.utc), None), #Integer unix timestamp to ISO format
        (datetime.datetime(2024, 7, 29, 1, 14, 57, tzinfo=datetime.timezone.utc), "timestamp", datetime.timezone.utc, 1722215697000, None), #Current datetime to timestamp
        # Exception cases
        (datetime.date(2025, 2, 18), 'wrong', datetime.timezone.utc, None, ValueError), #Invalid format
        ([], 'datetime', datetime.timezone.utc, None, TypeError), #Invalid type
        ('20345-2345-23', 'datetime', datetime.timezone.utc, None, ValueError), #Wrongly formatted date string
        (23521352152125123521351, 'isoformat', datetime.timezone.utc, None, OverflowError), #Overflow error for large unix timestamp
        (datetime.datetime(2024, 7, 29, 1, 14, 57, tzinfo=datetime.timezone.utc), "timestamp", "est", None, TypeError) #Invalid type for timestamp
    ]
)
def test_set_date(date, fmt, tz, expected, raises):
    if raises:
        with pytest.raises(raises):
            Helper.set_date(date=date, fmt=fmt, tz=tz)
    else:
        result = Helper.set_date(date=date, fmt=fmt, tz=tz)
        assert result == expected


@pytest.mark.parametrize(
    "lst,expected,raises",
    [
        # Valid cases
        (['AAPL', 'GOOGL'], np.array(['AAPL', 'GOOGL'], dtype=str), None), #string list to Numpy array
        ('MSFT', np.array(['MSFT'], dtype=np.str_), None), #Single string to Numpy array
        (np.array(['TSLA', 'AMZN'], dtype=np.str_), np.array(['TSLA', 'AMZN'], dtype=np.str_), None), #Numpy string array to Numpy array
        (['AAPL', 123], np.array(['AAPL', '123']), None), #List with mixed types to Numpy array
        # Exception cases                                                           
        (123, None, TypeError), #Invalid type
        (None, None, TypeError), #None type
    ]
)
def test_set_str_list(lst, expected, raises):
    if raises:
        with pytest.raises(raises):
            Helper.set_str_list(lst=lst)
    else:
        result = Helper.set_str_list(lst=lst)
        assert np.array_equal(result, expected)
        assert result.dtype.type == np.str_
        assert isinstance(result, np.ndarray)           

if __name__ == '__main__':
    ticker = 'AAPL'
    historical_ohlc_5_min = Helper.get_paginated_request(
        request_url="https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-08-01/2024-08-22?adjusted=true&sort=asc&limit=50000&apiKey=QEjL21NHSpWuFK4EC4_Bej76tZSwrU69",
        df_schema={'ticker': str, 'timestamp': str, 'o': float, 'h': float, 'l': float, 'c': float, 'v': float, 'vw': float, 'n': int, 'otc': bool, 't': int}
    )
    historical_ohlc_5_min = historical_ohlc_5_min.with_columns(pl.col("otc").fill_null(value=False))
    historical_ohlc_5_min = historical_ohlc_5_min.with_columns(pl.col("ticker").fill_null(value=ticker))
    historical_ohlc_5_min = historical_ohlc_5_min.with_columns(
        pl.col("t")
        .cast(pl.Datetime("ms", time_zone="UTC"))  
        .dt.strftime("%+")
        .alias("timestamp")
    )
    ticker = 'GOOG'
    df2 = Helper.get_paginated_request(
        request_url="https://api.polygon.io/v2/aggs/ticker/GOOG/range/1/day/2024-08-01/2024-08-22?adjusted=true&sort=asc&limit=50000&apiKey=QEjL21NHSpWuFK4EC4_Bej76tZSwrU69",
        df_schema={'ticker': str, 'timestamp': str, 'o': float, 'h': float, 'l': float, 'c': float, 'v': float, 'vw': float, 'n': int, 'otc': bool, 't': int}
    )
    df2 = df2.with_columns(pl.col("otc").fill_null(value=False))
    df2 = df2.with_columns(pl.col("ticker").fill_null(value=ticker))
    df2 = df2.with_columns(
        pl.col("t")
        .cast(pl.Datetime("ms", time_zone="UTC"))  
        .dt.strftime("%+")
        .alias("timestamp")
    )
    historical_ohlc_5_min.vstack(df2, in_place=True)


    historical_ohlc_5_min.write_parquet("tests/Test-Data/get_historical_ohlc_day.parquet")
