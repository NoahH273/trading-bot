import datetime
import time
import os 

import numpy as np
import numpy.typing as npt
import polars as pl
import requests

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
class Helper:
    
    date_types = str|datetime.date|datetime.datetime|int|float
    str_list_types = str|list|npt.NDArray[np.str_]

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
            TypeError: If date is not a valid type
            ValueError: If format is not a valid string

        Returns:
            converted_date: New date as str, datetime date object, or unix millisecond timestamp (int) depending on format arg.
        """
        if not isinstance(date, datetime.date):
            if isinstance(date, str):
                date = datetime.date.fromisoformat(date)       
            elif isinstance(date, int):
                date = datetime.date.fromtimestamp(date)
            elif isinstance(date, datetime.datetime):
                date = date.date()
            else:
                raise TypeError('date must be isoformat date string, unix millisecond timestamp, date object, or datetime object')
            
        if format == 'date':
            return date
        elif format == 'isoformat':
            return date.isoformat()
        elif format == 'timestamp':
            return date.timestamp()
        else:
            raise ValueError(f'{format} invalid format argument. format must be "isoformat", "timestamp", or "date")')
    
    @staticmethod
    def set_str_list(lst: str_list_types = None) -> npt.NDArray[np.str_]:
        """Converts strings and string lists to a Numpy string array.

        Args:
            lst (str, list, numpy array): The input to be mutated to a Numpy array.

        Raises:
            TypeError: If lst is not a string, list, or numpy array with dtype string

        Returns:
            formatted_tickers: A Numpy array of ticker strings.
        """
        if isinstance(lst, np.ndarray):
            if lst.dtype.type == str:
                return lst
            else: raise TypeError('arg lst must be an array of strings')
        elif isinstance(lst, str) or isinstance(lst, list):
            return np.array(lst)
        else:
            raise TypeError('invalid lst argument: must be of type string, list, or numpy array')


if __name__ == '__main__':
    print(Helper.set_date(251235123523452345234523452345234523542345, 'isoformat'))