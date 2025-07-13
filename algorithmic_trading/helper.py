import datetime
import logging
import time

import numpy as np
import numpy.typing as npt
import polars as pl
import requests

from secret import POLYGON_API_KEY
class Helper:
    
    date_types = str|datetime.date|datetime.datetime|int|float
    ticker_list_types = str|list|npt.NDArray[np.str_]|pl.DataFrame

    @staticmethod
    def get_paginated_request(request_url: str, df_schema: dict | None = None) ->  pl.DataFrame:
        """Returns all data from a Polygon API request in one Polars Dataframe.

        Args:
            request_url (str): URL for the HTTP request.
            df_schema ({str:Datatype} dict | None, optional): Defines the schema of the resulting Dataframe. No schema could cause errors in non-uniform data.

        Raises:
            ValueError: If the HTTP request is unsuccessful.

        Returns:
            results_df: A Polars Dataframe with data from all pages of the HTTP request.
        """
        request_response = requests.get(request_url)
        if request_response.status_code != 200:
            raise ValueError(f'http request failed: {request_url}')
        request_json = request_response.json()
        if request_json['status'] == 'ERROR':
            time.sleep(70)
        results_df = pl.DataFrame(schema=df_schema, data=request_json['results'])

        while 'next_url' in request_json:
            request_url = f'{request_json['next_url']}&apiKey={POLYGON_API_KEY}'
            request_json = requests.get(request_url).json()
            if request_json['status'] == 'ERROR':
                time.sleep(70)
                request_json= requests.get(request_url).json()
            page_df = pl.DataFrame(schema = df_schema, data = request_json['results'])
            results_df.vstack(page_df, in_place=True)
        return results_df

    @staticmethod
    def set_date(date: date_types, format: str) -> date_types:
        """Converts dates between different formats.

        Args:
            date (str, datetime.date, datetime.datetime, int, float): The date to change the format of.
            format (str): The format to change the date to. Accepted strings are "isoformat", "date", and "timestamp".

        Raises:
            TypeError: _description_

        Returns:
            converted_date: New date as str, datetime date object, or unix millisecond timestamp (int) depending on format arg.
        """
        if isinstance(date, str):
            try:
                datetime.date.fromisoformat(date)
                return date
            except (ValueError, TypeError):
                logging.exception('date string must be in isoformat: YYYY-MM-DD')
        elif isinstance(date, int):
            try:
                date = datetime.date.fromtimestamp(date)
                return date.isoformat()
            except ValueError:
                logging.exception('unix timestamp out of range')
        elif isinstance(date, datetime.date):
            return date.isoformat()
        elif isinstance(date, datetime.datetime):
            return date.date().isoformat()
        else:
            raise TypeError('date must be isoformat date string, unix millisecond timestamp, date object, or datetime object')
    
    @staticmethod
    def set_ticker_list(ticker_list: ticker_list_types = None) -> np.ndarray:
        """Converts ticker list inputs of various formats to a Numpy string array.

        Args:
            ticker_list (str, list, Numpy array, polars dataframe): The input to be mutated to a Numpy array.

        Raises:
            TypeError: _description_
            TypeError: _description_
            TypeError: _description_

        Returns:
            formatted_tickers: A Numpy array of ticker strings.
        """
        if tickers == None:
            return pl.read_parquet('Data/tickers.parquet').select(pl.col('ticker')).to_numpy()
        elif isinstance(tickers, np.ndarray):
            if tickers.dtype.type == str:
                return tickers
            else: raise TypeError('arg tickers must be an array of strings')
        elif isinstance(tickers, str):
            return np.array(tickers)
        elif isinstance(tickers, list):
            tickers = np.array(tickers)
            if tickers.dtype.type == str:
                return tickers
            else: raise TypeError('arg tickers must be an array of strings')
        elif isinstance(tickers, pl.dataframe.frame.DataFrame):
            try:
                arr = tickers.select(pl.col('ticker')).to_numpy()
                return arr
            except pl.dataframe.frame.ColumnNotFoundError:
                logging.exception('the dataframe passed for tickers arg does not contain a column named "ticker"')
        else:
            raise TypeError('invalid tickers argument: must be of type string, string list, or polars dataframe with a "ticker" column. leave empty for all tickers')

# if __name__ == '__main__':
#     Helper.set_date()
#     Helper.set_ticker_list()
#     Helper.get_paginated_request()