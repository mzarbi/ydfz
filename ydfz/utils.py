import json
import pickle
from struct import pack, calcsize, unpack
from datetime import datetime

import numpy as np
import pandas as pd
from bitarray import bitarray
from construct import Adapter
from pybloom_live import BloomFilter


class SerializableBloomFilter(BloomFilter):
    """
    Serializable version of BloomFilter, allowing serialization and deserialization of the filter.
    """

    FILE_FMT = '>diiii'  # Float (double) and four integers, all big-endian

    def to_bytes(self):
        """
        Serialize the bloom filter to bytes.
        This method packs the primitive attributes into bytes and appends the bit array as bytes.
        """
        header = pack(
            self.FILE_FMT,
            self.error_rate,
            self.num_slices,
            self.bits_per_slice,
            self.capacity,
            self.count
        )
        return header + self.bitarray.tobytes()

    @classmethod
    def from_bytes(cls, data):
        """
        Deserialize the bloom filter from bytes.
        """
        headerlen = calcsize(cls.FILE_FMT)
        if len(data) < headerlen:
            raise ValueError('Data too small!')

        # Unpack and debug print to check the values
        unpacked_data = unpack(cls.FILE_FMT, data[:headerlen])

        error_rate, num_slices, bits_per_slice, capacity, count = unpacked_data

        # Create an instance with unpacked values
        instance = cls(capacity, error_rate)  # Adjust initial_capacity as needed
        instance.num_slices = num_slices
        instance.bits_per_slice = bits_per_slice
        instance.capacity = capacity
        instance.count = count
        instance.bitarray = bitarray(endian='little')
        instance.bitarray.frombytes(data[headerlen:])

        return instance


class BloomFilterAdapter(Adapter):
    """
    Adapter for encoding and decoding bloom filter objects.
    """

    def _decode(self, obj, context, path):
        """
        Decode a serialized bloom filter back into a bloom filter object or None.
        """
        if obj:
            return SerializableBloomFilter.from_bytes(obj)
        return None

    def _encode(self, obj, context, path):
        """
        Encode a bloom filter object into bytes, or handle None.
        """
        if obj is None:
            return b''
        return obj


class PickleAdapter(Adapter):
    """
    Adapter for encoding and decoding objects using pickle serialization.
    """

    def _decode(self, obj, context, path):
        """
        Decode a pickled object.
        """
        return pickle.dumps(obj)

    def _encode(self, obj, context, path):
        """
        Encode an object using pickle serialization.
        """
        return pickle.loads(obj)


def extract_null_mask(series):
    """
    Extract null mask from a pandas Series.

    Parameters:
        series (pandas.Series): The input Series.

    Returns:
        tuple: A tuple containing:
            - non_null_data (pandas.Series): The non-null data.
            - mask_flag (int): Flag indicating null or non-null mask.
            - packed_mask (numpy.ndarray): Packed null mask.
    """
    non_null_mask = series.notna()
    non_null_count = non_null_mask.sum()

    mask = non_null_mask if non_null_count <= len(series) / 2 else ~non_null_mask
    mask_flag = 0 if non_null_count <= len(series) / 2 else 1

    packed_mask = np.packbits(mask)
    non_null_data = series[non_null_mask]

    return non_null_data, mask_flag, packed_mask


def datetime_to_yyyymmdd(dt):
    """
    Convert datetime object to YYYYMMDD integer.

    Parameters:
        dt (datetime.datetime or pandas.Timestamp): The input datetime object.

    Returns:
        int: YYYYMMDD integer representation of the input date.
    """
    # Check if the input is a pandas Timestamp
    if isinstance(dt, pd.Timestamp):
        day = dt.day
        month = dt.month
        year = dt.year
    # Check if the input is a Python datetime object
    elif isinstance(dt, datetime):
        day = dt.day
        month = dt.month
        year = dt.year
    else:
        raise TypeError("Input must be either a pandas Timestamp or a Python datetime object.")

    return int(f"{year:04d}{month:02d}{day:02d}")


def convert_to_categorical(series):
    """
    Convert Series to categorical type and return category codes with a mapping.

    Parameters:
        series (pandas.Series): The input Series.

    Returns:
        tuple: A tuple containing:
            - codes (pandas.Series): Categorical codes of the input Series.
            - categories_map (dict): Mapping of category codes to categories.
    """
    if series.dtype == object or series.dtype.name == 'category':
        series = series.astype('category')
        categories_map = dict(enumerate(series.cat.categories))
        return series.cat.codes.replace(-1, len(series.cat.categories)), categories_map
    raise ValueError("Series must be of object type for categorization.")


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # Convert pd.NA and NaN to None
        if pd.isnull(obj):
            return None
        if pd.isna(obj):
            return None
        elif isinstance(obj, float) and np.isnan(obj):
            return None

        # Convert timestamp to string if it's a datetime object
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        #
        return json.JSONEncoder.default(self, obj)