import datetime
import time
import os
import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import requests
from requests.adapters import HTTPAdapter, Retry

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
class Helper:
    
    date_types = str|datetime.date|datetime.datetime|int|float
    str_list_types = str|list|npt.NDArray[np.str_]

    @staticmethod
    def get_paginated_request(request_url: str, df_schema: dict | None = None, safe: bool = True, session: requests.Session = None) ->  pl.DataFrame:
        """Returns all data from a Polygon API request in one Polars Dataframe.

        Args:
            request_url (str): URL for the HTTP request.
            df_schema ({str:Datatype} dict | None, optional): Defines the schema of the resulting Dataframe. No schema could cause errors in non-uniform data.
            safe (bool, optional): If True, raises an error if the request fails or times out. If False, returns an empty DataFrame with the specified schema, and gives a warning. Defaults to True.
            session (requests.Session, optional): A requests Session object to use for the HTTP requests. If None, a new Session will be created. Defaults to None.

        Raises:
            ValueError: If the HTTP request is unsuccessful, or if the request returns no results.

        Returns:
            results_df: A Polars Dataframe with data from all pages of the HTTP request.
        """
        if session is None:
            session = requests.Session()
            retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], raise_on_redirect=True)
            adapter = HTTPAdapter(max_retries=retries)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

        request_response = session.get(request_url, timeout=15)
        if request_response.status_code != 200:
            if safe:
                raise ValueError(f'http request failed: {request_url}')
            else:
                warnings.warn(f'http request failed: {request_url}')
                return pl.DataFrame(schema=df_schema)
        request_json = request_response.json()
        if request_json['status'] == 'ERROR':
            time.sleep(70)
            request_response = requests.get(request_url, timeout=15)
            request_json = request_response.json()
        if 'resultsCount' in request_json and request_json['resultsCount'] == 0:
            warnings.warn(f'No results found for request: {request_url}')
            return pl.DataFrame(schema=df_schema)
        results_df = pl.DataFrame(schema=df_schema, data=request_json['results'])

        while 'next_url' in request_json:
            request_url = f'{request_json['next_url']}&apiKey={POLYGON_API_KEY}'
            request_json = session.get(request_url, timeout=15).json()
            if request_json['status'] == 'ERROR':
                time.sleep(70)
                request_json= requests.get(request_url, timeout=15).json()
            page_df = pl.DataFrame(schema = df_schema, data = request_json['results'])
            results_df.vstack(page_df, in_place=True)
        return results_df

    @staticmethod
    def set_date(date: date_types, fmt: str, tz: datetime.tzinfo = datetime.timezone.utc) -> date_types:
        """Converts dates between different formats.

        Args:
            date (str, datetime.date, datetime.datetime, int, float): The date to change the format of.
            fmt (str): The format to change the date to. Accepted strings are "isoformat", "timestamp", and "datetime".
            tz (datetime.tzinfo, optional): The timezone in which to return the new date. Defaults to UTC. If the input date does not have a timezone and tz is not the same as loca timezone, the date's naive value will be changed.

        Raises:
            TypeError: If date is not a valid type.
            ValueError: If format is not a valid string, or isoformat date string is not in the correct format.
            OverflowError: If the unix timestamp is too large to convert to a date.

        Returns:
            converted_date: New date as str or datetime date object depending on fmt.
        """
        if not isinstance(date, datetime.datetime):
            if isinstance(date, str):
                date = datetime.datetime.fromisoformat(date)       
            elif isinstance(date, int) or isinstance(date, float):
                date = datetime.datetime.fromtimestamp(date)
            elif isinstance(date, datetime.date):
                date = datetime.datetime.combine(date, datetime.time(0, 0, 0, 0))
            else:
                raise TypeError('date must be isoformat date string, unix millisecond timestamp, date object, or datetime object')
        
        if date.tzinfo is None or date.tzinfo != tz:
            date = date.replace(tzinfo=tz)
        elif date.tzinfo != tz:
            date = date.astimezone(tz)

        if fmt == 'datetime':
            return date
        elif fmt == 'isoformat':
            return date.isoformat()
        elif fmt == 'timestamp':
            return int(date.timestamp() * 1000)
        else:
            raise ValueError(f'{format} invalid fmt argument. format must be "isoformat" or "datetime")')
    
    @staticmethod
    def get_time_delta(timestamps: str_list_types) -> datetime.timedelta:
        """Calculates the most common time delta between a list of timestamps.

        Args:
            timestamps (str, list, numpy str array): A list of timestamps in isoformat string format.
        
        Returns:
            most_common_delta (datetime.timedelta): The most common time delta between the timestamps.
        """
        timestamps = Helper.set_str_list(timestamps)
        if len(timestamps) < 2:
            raise ValueError('At least 2 timestamps are required to calculate a time delta')
        prev_timestamp = datetime.datetime.fromisoformat(timestamps[0])
        time_deltas = {}
        for timestamp in timestamps:
            current_timestamp = datetime.datetime.fromisoformat(timestamp)
            time_delta = current_timestamp - prev_timestamp
            if time_delta in time_deltas:
                time_deltas[time_delta] += 1
            else:
                time_deltas[time_delta] = 1
            prev_timestamp = current_timestamp
        
        highest_count = 0
        most_common_delta = None
        for time_delta, count in time_deltas.items():
            if count > highest_count:
                highest_count = count
                most_common_delta = time_delta
        return most_common_delta
    
    @staticmethod
    def set_str_list(lst: str_list_types) -> npt.NDArray[np.str_]:
        """Converts strings and string lists to a Numpy string array.

        Args:
            lst (str, list, numpy array): The input to be mutated to a Numpy array.

        Raises:
            TypeError: If lst is not a string, list, or numpy array with dtype string

        Returns:
            formatted_tickers: A Numpy array of ticker strings.
        """
        if isinstance(lst, np.ndarray):
            if lst.dtype.type == np.str_:
                return lst
            else: raise TypeError('arg lst must be an array of strings')
        elif isinstance(lst, str):
            return np.array([lst], dtype=np.str_)
        elif isinstance(lst, list):
            return np.array(lst, dtype=np.str_)
        else:
            raise TypeError('invalid lst argument: must be of type string, list, or numpy array')

