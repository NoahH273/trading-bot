import datetime
import os
import warnings
import numpy as np
from tqdm import tqdm

import polars as pl

from algorithmic_trading.helper import Helper

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

class DataManager:
    @staticmethod
    def get_historical_tickers(ticker_types: Helper.str_list_types = None, include_delisted: bool = True, date: str = '') -> pl.DataFrame:
        """Retrieves data from the Polygon tickers endpoint.

        Args:
            ticker_types (str, list, numpy str array, polars dataframe with , optional): The ticker types to retrieve tickers for. Defaults to all, for a full list of valid types go to https://polygon.io/docs/rest/stocks/tickers/all-tickers.
            include_delisted (bool, optional): Whether or not to include delisted tickers. Defaults to True.
            date (str, optional): The date to retrieve tickers for. Defaults to today.
        Returns:
            results_df: A Polars Dataframe with a row for each ticker with the schema: {'ticker': str, 'name': str, 'market': str, 'locale': str, 'primary_exchange': str, 'type': str, 'active': bool, 'currency_name': str, 'cik': str, 'last_updated_utc': str, 'delisted_utc': str, 'composite_figi': str, 'share_class_figi': str}
        """
        ticker_types = Helper.set_str_list(ticker_types)
        if date != '':
            date = Helper.set_date(date, 'isoformat').split('T')[0]
            date = f"date={date}&"
        if not isinstance(include_delisted, bool):
            raise TypeError('include_delisted must be a boolean')
        df_schema = {'ticker': str, 'name': str, 'market': str, 'locale': str, 'primary_exchange': str, 'type': str, 'active': bool, 'currency_name': str, 'cik': str, 'last_updated_utc': str, 'delisted_utc': str, 'composite_figi': str, 'share_class_figi': str}
        results_df = pl.DataFrame(schema = df_schema)
        
        for ticker_type in ticker_types:
            request_url = f'https://api.polygon.io/v3/reference/tickers?type={ticker_type}&market=stocks&{date}active=true&order=asc&limit=1000&sort=ticker&apiKey={POLYGON_API_KEY}'

            listed_df = Helper.get_paginated_request(request_url=request_url, df_schema=df_schema)
            results_df.vstack(listed_df, in_place=True)
            if not include_delisted:
                continue
            request_url = f'https://api.polygon.io/v3/reference/tickers?type={ticker_type}&market=stocks&{date}active=false&order=asc&limit=1000&sort=ticker&apiKey={POLYGON_API_KEY}'
            unlisted_df = Helper.get_paginated_request(request_url=request_url, df_schema=df_schema)
            results_df.vstack(unlisted_df, in_place=True)
        return results_df
    
    @staticmethod
    def __ohlc_input_validator(timeframe: str, multiplier: int) -> None:
        """Helper function for functions dealing with ohlc data, validates timeframe and multiplier inputs used for polygon requests and file naming.

        Args:
            timeframe (str): The timeframe for each ohlc bar to be collected.
            multiplier (int): A multiplier for the timeframe. Ex: A multiplier of 5 and timeframe of "minute" makes 5 minute bars.

        Raises:
            ValueError: If start_date comes after end_date
            ValueError: If timeframe arg is not a valid string
            ValueError: If multiplier is not an integer greater than 0
        """
        valid_timeframes = {'hour', 'day', 'minute', 'week', 'month', 'quarter', 'year', 'second'}
        if timeframe not in valid_timeframes:
            raise ValueError(f'invalid timeframe argument, must be one of {valid_timeframes}.')
        if not isinstance(multiplier, int) or multiplier < 1:
            raise ValueError('multiplier must be an integer greater than 0')
        
    @staticmethod 
    def get_historical_ohlc(start_date: Helper.date_types = '2000-01-01', end_date: Helper.date_types = datetime.date.today(), tickers: Helper.str_list_types = None, use_ticker_types: bool = False, timeframe: str = 'day', multiplier: int = 1) -> pl.DataFrame:
        """Gets historical ohlc bars.

        Args:
            start_date (str, date or datetime object, unix timestamp, optional): The day to start collecting ohlc bars from. Defaults to '2000-01-01'. With timeframes >= 1 day, start date must be hour 4 utc of that day, or the previous day/week/month bar will be returned as well.
            end_date (str, date or datetime object, unix timestamp, optional): The day to stop collecting ohlc bars. Defaults to today's date. With timeframes >= 1 day, if end date has an hour later than or at 4 utc, the next day/week/month bar will be returned as well.
            tickers (str, str list, numpy str array, optional): A list of stock tickers to collect ohlc bars for. Defaults to all tickers in history.
            use_ticker_types (bool, optional): If True, uses the tickers arg as ticker types instead of tickers. Defaults to False.
            timeframe (str, optional): The timeframe for each ohlc bar. Valid values are ['second', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year']. Defaults to 'day'.
            multiplier (int, optional): The multiplier for the timeframe. Ex: A multiplier of 5 and timeframe of "minute" makes 5 minute bars. Defaults to 1.

        Returns:
            data_df: A Polars Dataframe with timestamped ohlc bar data with a schema of {ticker: str, timestamp: str, o: float, h: float, l: float, c: float, v: float, vw: float, n: int, otc: bool, t: int}
        """
        start_date = Helper.set_date(start_date, 'timestamp', tz=datetime.timezone.utc)
        end_date = Helper.set_date(end_date, 'timestamp', tz=datetime.timezone.utc)
        if(end_date < start_date):
            raise ValueError('start_date must be earlier than end_date')
        if tickers is None:
            tickers = pl.read_parquet('algorithmic_trading/Data/tickers.parquet').select(pl.col('ticker')).to_series().to_numpy() #Gets all tickers in history
        elif use_ticker_types:
            tickers = set(Helper.set_str_list(tickers))
            ticker_df = pl.read_parquet('algorithmic_trading/Data/tickers.parquet')
            ticker_df = ticker_df.filter(pl.col('type').is_in(tickers))
            tickers = ticker_df.select(pl.col('ticker')).to_series().unique()
            tickers = tickers.to_numpy().astype(np.str_)
        tickers = Helper.set_str_list(tickers)
        DataManager.__ohlc_input_validator(timeframe, multiplier)

        df_schema = {'ticker': str, 'timestamp': str, 'o': float, 'h': float, 'l': float, 'c': float, 'v': float, 'vw': float, 'n': int, 'otc': bool, 't': int}
        data_df = pl.DataFrame(schema=df_schema)
        for ticker in tqdm(tickers):
            if "/" in ticker:
                warnings.warn(f'Ticker {ticker} contains a slash, which is not supported by the Polygon API. Skipping this ticker.')
                continue
            request_url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timeframe}/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}'
            ticker_df = Helper.get_paginated_request(request_url=request_url, df_schema=df_schema, safe=False)
            if ticker_df is None or ticker_df.height == 0:
                warnings.warn(f'No data found for ticker {ticker} in the specified date range.')
                continue
            ticker_df = ticker_df.with_columns(pl.col("otc").fill_null(value=False))
            ticker_df = ticker_df.with_columns(pl.col("ticker").fill_null(value=ticker))
            ticker_df = ticker_df.with_columns(
                  pl.col("t")
                 .cast(pl.Datetime("ms", time_zone=("UTC")))  
                 .dt.strftime("%+")
                 .alias("timestamp")
            )
            data_df.vstack(ticker_df, in_place=True)
            print(f"Collected data for ticker: {ticker}")
        if data_df.height == 0:
            warnings.warn('No data found for the specified date range and tickers.')
    
        return data_df
    

    def post_historical_ohlc(data_df: pl.DataFrame, ohlc_path: str, timeframe: str, multiplier: int) -> None:
        """Posts historical ohlc data to parquet files.

        Args:
            data_df (pl.DataFrame): A Polars DataFrame with ohlc data to post. Schemas are not enforced to provide support for technical indicators.
            ohlc_path (str): The path to the folder where the parquet files will be saved. This can be a relative or absolute path, but the folder must exist before posting data, and the actual data will be contained in a subdirectory based on the timeframe and multiplier.
            timeframe (str): The timeframe for the ohlc data. Valid values are ['second', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year']. Unexpected behavior may occur if the timeframe passed is not the same as the timeframe used to collect the data in data_df.
            multiplier (int): The multiplier for the timeframe. Ex: A multiplier of 5 and timeframe of "minute" makes 5 minute bars. Unexpected behavior may occur if the multiplier passed is not the same as the multiplier used to collect the data in data_df.

        Raises:
            ValueError: If timeframe is not a valid string.
            ValueError: If multiplier is not an integer greater than 0.
        """
        if not os.path.exists(ohlc_path):
            raise ValueError(f'Path {ohlc_path} does not exist. Please create the directory before posting data.')
        DataManager.__ohlc_input_validator(timeframe, multiplier)
        timestamps = data_df.unique('timestamp').get_column('timestamp').to_list()
        folder_path = f'{ohlc_path}/{multiplier}_{timeframe}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        for timestamp in timestamps:
            timestamp_df = data_df.filter(pl.col('timestamp') == timestamp)
            if timestamp_df.height == 0:
                continue
            date = Helper.set_date(timestamp, 'datetime', tz=datetime.timezone.utc).date().isoformat()
            file_path = f'{folder_path}/{date}.parquet'
            if os.path.exists(file_path):
                existing_df = pl.read_parquet(file_path)
                combined_df = pl.concat([existing_df, timestamp_df], how="diagonal_relaxed").unique(subset=['timestamp', 'ticker'], keep='last')
                combined_df.write_parquet(file_path)
            else:
                timestamp_df.write_parquet(file_path)

if __name__ == "__main__":
    ohlc_df = DataManager.get_historical_ohlc(start_date='2020-05-02', end_date='2025-08-06', tickers='CS', use_ticker_types=True, timeframe='hour', multiplier=1)
    DataManager.post_historical_ohlc(data_df=ohlc_df, ohlc_path = 'algorithmic_trading/Data/OHLC', timeframe = 'hour', multiplier = 1)
