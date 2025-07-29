import datetime
import pytest
from algorithmic_trading.helper import Helper
import numpy as np
import numpy.typing as npt

@pytest.mark.parametrize(
    "date,fmt,datetime,expected,raises",
    [
        # Valid cases
        (datetime.date(2020, 5, 1), 'isoformat', '2020-05-01T00:00:00', None), #Date to ISO format
        ('2021-12-25T05:12:06', 'datetime', datetime.datetime(2021, 12, 25, 5, 12, 6), None), #ISO format to datetime
        (datetime.datetime(2021, 10, 10), 'datetime', datetime.datetime(2021, 10, 10), None), #Datetime to datetime 
        (1658105431.54, 'isoformat', '2022-07-17T17:50:31.540000', None), #Float unix timestamp to ISO format
        (1620814598, 'datetime', datetime.datetime(2021, 5, 12, 3, 16, 38), None), #Integer unix timestamp to ISO format
        (datetime.datetime.today(), )
        # Exception cases
        (datetime.date(2025, 2, 18), 'wrong', None, ValueError), #Invalid format
        ([], 'datetime', None, TypeError), #Invalid type
        ('20345-2345-23', 'datetime', None, ValueError), #Wrongly formatted date string
        (23521352152125123521351, 'isoformat', None, OverflowError) #Overflow error for large unix timestamp
    ]
)
def test_set_date(date, fmt, datetime, expected, raises):
    if raises:
        with pytest.raises(raises):
            Helper.set_date(date=date, fmt=fmt, datetime=datetime)
    else:
        result = Helper.set_date(date=date, fmt=fmt, datetime=datetime)
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

