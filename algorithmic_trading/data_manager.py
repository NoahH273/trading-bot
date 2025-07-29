import datetime
import os

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
    def __get_data_input_validator(start_date: str, end_date: str, timeframe: str, multiplier: int) -> None:
        """Helper function for get_data, validates input types and values

        Args:
            start_date (str): The start date to collect data for each stock
            end_date (str): The end date to collect data for each stock.
            timeframe (str): The timeframe for each ohlc bar to be collected.
            multiplier (int): A multiplier for the timeframe. Ex: A multiplier of 5 and timeframe of "minute" makes 5 minute bars.

        Raises:
            ValueError: If start_date comes after end_date
            ValueError: If timeframe arg is not a valid string
            ValueError: If multiplier is not an integer greater than 0
        """
        if(end_date < start_date):
            raise ValueError('start_date must be earlier than end_date')
        valid_timeframes = {'hour', 'day', 'minute', 'week', 'month', 'quarter', 'year', 'second'}
        if timeframe not in valid_timeframes:
            raise ValueError(f'invalid timeframe argument, must be one of {valid_timeframes}.')
        if not isinstance(multiplier, int) or multiplier < 1:
            raise ValueError('multiplier must be an integer greater than 0')
        
    @staticmethod 
    def get_data(start_date: Helper.date_types = '2000-01-01', end_date: Helper.date_types = datetime.date.today(), tickers: Helper.str_list_types = None, timeframe: str = 'day', multiplier: int = 1) -> pl.DataFrame:
        """Gets historical ohlc bars.

        Args:
            start_date (str, date or datetime object, unix timestamp, optional): The day to start collecting ohlc bars from. Defaults to '2000-01-01'.
            end_date (str, date or datetime object, unix timestamp, optional): The day to stop collecting ohlc bars. Defaults to today's date.
            tickers (str, str list, numpy str array, optional): A list of stock tickers to collect ohlc bars for. Defaults to all tickers in history.
            timeframe (str, optional): The timeframe for each ohlc bar. Valid values are ['second', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year']. Defaults to 'day'.
            multiplier (int, optional): The multiplier for the timeframe. Ex: A multiplier of 5 and timeframe of "minute" makes 5 minute bars. Defaults to 1.

        Returns:
            data_df: A Polars Dataframe with timestamped ohlc bar data with a schema of {ticker: str, timestamp: str, o: float, h: float, l: float, c: float, v: float, vw: float, n: int, otc: bool, t: int}
        """
        start_date = Helper.set_date(start_date, 'isoformat')
        end_date = Helper.set_date(end_date, 'isoformat')
        if tickers is None:
            tickers = pl.read_parquet('Data/ticker_list.parquet').select(pl.col('ticker')).to_numpy()
        else: tickers = Helper.set_str_list(tickers)
        DataManager.__get_data_input_validator(start_date, end_date, timeframe, multiplier)       
        print(tickers)

if __name__ == '__main__':
    print('' is True)