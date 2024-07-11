import datetime
import sys
import pandas as pd
import numpy as np
import os
from ydfz.compressors import Compressor
from ydfz.serializers import PickleSerializer, MsgpackSerializer, MetadataSerializer
from ydfz.utils import SerializableBloomFilter, extract_null_mask, datetime_to_yyyymmdd, convert_to_categorical, \
    CustomEncoder, EncodeFromNumpy


class MappedFile:
    def __init__(self, filename, mode, file_metadata=None, footer_serializer=None):
        self.filename = filename
        self.mode = mode
        self.file = None
        self.partitions = []  # In-memory cache for partitions data
        self.file_metadata = file_metadata if file_metadata else {}
        self.footer_serializer = footer_serializer() if footer_serializer else PickleSerializer()
        self.metadata_serializer = MsgpackSerializer()

        self.file_metadata['footer_serializer'] = self.footer_serializer.__class__.__name__

        # Open file based on the mode
        if mode == 'write':
            self.file = open(filename, 'wb+')
        elif mode == 'read' or mode == 'append':
            self.file = open(filename, 'rb+' if mode == 'append' else 'rb')
            self._read_footer()
            if mode == 'append' and self.partitions:
                self.file.seek(self.partitions[0]['offset'])  # Assume first partition points to footer start
                self.file.truncate()

    def _read_footer(self):
        self.file.seek(-8, os.SEEK_END)
        footer_length = int.from_bytes(self.file.read(4), 'little')
        metadata_length = int.from_bytes(self.file.read(4), 'little')

        self.file.seek(-(footer_length + metadata_length + 8), os.SEEK_END)

        metadata_data = self.file.read(metadata_length)
        metadata_data = Compressor.decompress_data(metadata_data, 'snappy')
        self.file_metadata = self.metadata_serializer.deserialize(metadata_data)

        footer_data = self.file.read(footer_length)
        self.footer_serializer = globals()[self.file_metadata['footer_serializer']]()
        footer_data = Compressor.decompress_data(footer_data, 'snappy')
        self.partitions = self.footer_serializer.deserialize(footer_data)

    def write_partition(self, serialized_data, partition_metadata={}):
        if self.mode in ['write', 'append']:
            pos = self.file.tell()
            partition_info = {
                'offset': pos,
                'length': len(serialized_data),
            }
            partition_info.update(partition_metadata)
            self.partitions.append(partition_info)
            self.file.write(serialized_data)
        else:
            raise PermissionError("File not opened in write or append mode.")

    def read_partition(self, index):
        if index < 0 or index >= len(self.partitions):
            raise IndexError("Partition index out of range.")
        partition = self.partitions[index]
        self.file.seek(partition['offset'])
        serialized_data = self.file.read(partition['length'])
        return serialized_data

    def close(self):
        if self.mode in ['write', 'append']:
            footer_data = self.footer_serializer.serialize(self.partitions)
            footer_data = Compressor.compress_data(footer_data, 'snappy')
            footer_length = len(footer_data)
            metadata_data = self.metadata_serializer.serialize(self.file_metadata)
            metadata_data = Compressor.compress_data(metadata_data, 'snappy')
            metadata_length = len(metadata_data)

            self.file.write(metadata_data)
            self.file.write(footer_data)
            self.file.write(footer_length.to_bytes(4, 'little'))
            self.file.write(metadata_length.to_bytes(4, 'little'))
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # Handle exceptions if necessary
        return False


class MappedDataFrame(MappedFile):
    def __init__(self, filename, mode, file_metadata=None, footer_serializer=None):
        super().__init__(filename, mode, file_metadata, footer_serializer)

    def close(self):
        self.file_metadata.update({'file_type': 'MappedDataFrame'})
        super().close()

    def compute_stats(self, non_null_data, dtype, series_metadata):
        if non_null_data.size > 0:
            if dtype == "datetime64[ns]":
                series_metadata['min'] = datetime_to_yyyymmdd(non_null_data.min())
                series_metadata['max'] = datetime_to_yyyymmdd(non_null_data.max())
            elif "int" in dtype:
                series_metadata['min'] = int(non_null_data.min())
                series_metadata['max'] = int(non_null_data.max())
            elif "float" in dtype:
                series_metadata['min'] = float(non_null_data.min())
                series_metadata['max'] = float(non_null_data.max())
            else:
                series_metadata['min'] = non_null_data.min()
                series_metadata['max'] = non_null_data.max()

        # Initialize a Bloom filter
        bloom_filter = SerializableBloomFilter(capacity=non_null_data.size, error_rate=0.001)
        for item in non_null_data:
            if dtype == "datetime64[ns]":
                bloom_filter.add(datetime_to_yyyymmdd(item))
            elif "int" in dtype:
                bloom_filter.add(int(item))
            elif "float" in dtype:
                bloom_filter.add(float(item))
            else:
                bloom_filter.add(item)

        # Store the Bloom filter in a byte format
        bloom_filter_byte_data = bloom_filter.to_bytes()
        series_metadata['bloom_filter'] = bloom_filter_byte_data
        series_metadata['bloom_filter_length'] = len(bloom_filter_byte_data)

    def write_series(self, series, dtype, compression=None, compute_stats=True, series_metadata={}):
        non_null_data, mask_flag, null_mask = extract_null_mask(series)
        null_mask = null_mask.tobytes()
        series_metadata.update({
            'series_len': len(series),
            'dtype': dtype,
            'mask_flag': mask_flag,
            'mask_bytes': null_mask,
            'mask_bytes_length': len(null_mask),
            'categories': None,
            'compression': compression,
            'compute_stats': compute_stats,
            'min': None,
            'max': None,
            'bloom_filter_length': None,
            'bloom_filter': None
        })

        if dtype in ["int64", "int32", "int16", "int8", "float64", "float32", "float16", "float8"]:
            non_null_data_ = non_null_data.astype(dtype).values
            data_bytes = non_null_data_.tobytes()
        elif dtype in ["int"]:
            non_null_data_ = non_null_data.astype("int").values
            data_bytes = non_null_data_.tobytes()

        elif dtype == "datetime64[ns]":
            non_null_data_ = non_null_data.view('int64').values
            data_bytes = non_null_data_.tobytes()

        elif dtype == "object":
            non_null_data_, categories = convert_to_categorical(non_null_data)
            non_null_data_ = non_null_data_.astype('int32')
            data_bytes = non_null_data_.values.tobytes()
            series_metadata['categories'] = categories
        elif dtype == "bool":
            packed_data = np.packbits(non_null_data)
            data_bytes = packed_data.tobytes()
        else:
            raise ValueError("Unsupported dtype.")

        compressed_data = Compressor.compress_data(data_bytes, compression)

        if compute_stats:
            self.compute_stats(non_null_data, dtype, series_metadata)

        # Write the possibly compressed data to the partition
        self.write_partition(compressed_data, series_metadata)

    def read_series(self, index):
        data_bytes, series_metadata = self.read_partition(index), self.partitions[index]

        compression = series_metadata.get('compression', None)
        # Decompress the data bytes according to the compression method stored in the metadata
        decompressed_data = Compressor.decompress_data(data_bytes, compression)

        dtype = series_metadata['dtype']
        mask_bytes = series_metadata['mask_bytes']
        series_len = series_metadata['series_len']
        mask_flag = series_metadata['mask_flag']
        categories = series_metadata.get('categories', None)

        # Convert mask bytes back to boolean array
        unpacked_mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8))
        valid_indices = unpacked_mask[:series_len] == (1 if mask_flag == 0 else 0)

        # Initialize a full series with appropriate NaN equivalent for the dtype
        if dtype in ["float64", "float32", "float16"]:
            full_data = np.full(series_len, np.nan, dtype=dtype)
        elif dtype == "object":
            full_data = np.full(series_len, None, dtype='object')  # Use None for object type
        elif dtype == "bool":
            full_data = np.full(series_len, False, dtype=bool)  # Default to False for boolean
        elif dtype == "datetime64[ns]":
            full_data = np.full(series_len, np.nan, dtype='datetime64[ns]')  # Use pd.NaT for datetime
        elif dtype == "int":
            full_data = np.full(series_len, np.nan, dtype='float')
        else:
            full_data = np.full(series_len, np.nan, dtype=dtype)  # For other types that support np.nan

        # Convert data bytes back to the appropriate dtype and fill in non-null positions
        if dtype == "bool":
            non_null_data = np.unpackbits(np.frombuffer(decompressed_data, dtype=np.uint8))
            full_data[valid_indices] = non_null_data[:np.count_nonzero(valid_indices)]
        elif dtype == "datetime64[ns]":
            non_null_data = np.frombuffer(decompressed_data, dtype='int64').astype('datetime64[ns]')
            full_data[valid_indices] = non_null_data
        elif categories is not None:
            non_null_data = np.frombuffer(decompressed_data, dtype='int32')  # Matching dtype as int32
            category_list = [categories[i] for i in sorted(categories.keys())]
            category_array = pd.Categorical.from_codes(non_null_data, categories=category_list, ordered=True)
            full_data[valid_indices] = category_array
        elif dtype == "int":
            non_null_data = np.frombuffer(decompressed_data, dtype="int")
            full_data[valid_indices] = non_null_data
        else:
            non_null_data = np.frombuffer(decompressed_data, dtype=dtype)
            full_data[valid_indices] = non_null_data

        return pd.Series(full_data)

    def read_shard(self, shard_index):
        shard_data = []
        column_names = []
        for index, partition in enumerate(self.partitions):
            if partition.get('shard') == shard_index:
                series = self.read_series(index)
                #series = series.astype(partition.get('dtype'))
                shard_data.append(series)
                column_names.append(partition.get('name'))
        df = pd.concat(shard_data, axis=1)
        df.columns = column_names
        return df

    def query(self, output_file, query_conditions):
        relevant_shards = self.predicate_pushdown(query_conditions)
        with PagedFile(output_file, 'write', page_size=100, format='json') as pg:
            for shard_index in relevant_shards:
                shard = self.read_shard(shard_index)
                filtered_shard = self.filter_dataframe(shard, query_conditions)
                pg.write_data(filtered_shard)

    def filter_dataframe(self, df, filter_config):
        filtered_df = df.copy()

        # Iterate over the rules
        for rule in filter_config.get('rules', []):
            field = rule['field']
            operator = rule['operator']
            value = rule['value']

            if operator == '>':
                filtered_df = filtered_df[filtered_df[field] > value]
            elif operator == '<':
                filtered_df = filtered_df[filtered_df[field] < value]
            elif operator == '>=':
                filtered_df = filtered_df[filtered_df[field] >= value]
            elif operator == '<=':
                filtered_df = filtered_df[filtered_df[field] <= value]
            elif operator == '==':
                filtered_df = filtered_df[filtered_df[field] == value]
            elif operator == '!=':
                filtered_df = filtered_df[filtered_df[field] != value]

        # Recursively apply filters for groups
        for group in filter_config.get('groups', []):
            group_filter = self.filter_dataframe(df, group)
            if filter_config['operation'] == 'AND':
                filtered_df = pd.merge(filtered_df, group_filter, how='inner')
            elif filter_config['operation'] == 'OR':
                filtered_df = pd.concat([filtered_df, group_filter]).drop_duplicates()

        return filtered_df

    def predicate_pushdown(self, query_conditions):
        # Start the recursive evaluation
        return self.evaluate_conditions(self.partitions, query_conditions)

    def evaluate_conditions(self, partitions, conditions):
        operation = conditions.get('operation').upper()  # AND or OR
        relevant_partitions = set()

        # Process each rule at this level against all partitions
        for rule in conditions.get('rules', []):
            matching_partitions = set(self.apply_rule_to_partitions(partitions, rule))
            if operation == 'AND':
                if not relevant_partitions:
                    relevant_partitions = matching_partitions
                else:
                    relevant_partitions &= matching_partitions
            elif operation == 'OR':
                relevant_partitions |= matching_partitions

        # Recursively process groups
        for group in conditions.get('groups', []):
            group_partitions = self.evaluate_conditions(partitions, group)
            if operation == 'AND':
                relevant_partitions &= group_partitions
            elif operation == 'OR':
                relevant_partitions |= group_partitions

        return relevant_partitions

    def apply_rule_to_partitions(self, partitions, rule):
        results = []
        for index, entry in enumerate(partitions):
            if self.check_relevance(entry, rule):
                results.append(entry.shard)
        return results

    def check_relevance(self, entry, rule):
        column_name, operator, value = rule['field'], rule['operator'], rule['value']
        if entry.name == column_name:
            if operator == 'equal' and value in entry.bloom_filter():
                return True
            if self.evaluate_range(entry.dtype, entry.min, entry.max, operator, value):
                return True
        return False

    def evaluate_range(self, dtype, min_value, max_value, operator, value):
        if dtype == "datetime64[ns]":
            value = datetime_to_yyyymmdd(value)
        if max_value is None and min_value is None:
            return False
        # Check against max_value
        if max_value is not None:
            if operator == '>=' and max_value < value:
                return False
            if operator == '>' and max_value <= value:
                return False

        # Check against min_value
        if min_value is not None:
            if operator == '<=' and min_value > value:
                return False
            if operator == '<' and min_value >= value:
                return False
        return True

    def describe_file(self):
        # Display file metadata and initialize variables for statistics
        print("File Metadata:")
        for key, value in self.file_metadata.items():
            print(f"  {key}: {value}")

        total_data_size = 0
        total_metadata_size = 0
        print("\nPartitions:")
        for idx, partition in enumerate(self.partitions):
            print(f"  Partition {idx}:")
            print(f"    Offset: {partition['offset']}")
            print(f"    Length: {partition['length']} bytes")
            print(f"    Data Type: {partition['dtype']}")
            print(f"    Series Length: {partition['series_len']}")
            print(f"    Compression: {partition.get('compression', 'None')}")
            print(f"    Categories: {partition.get('categories', [])}")
            if 'bloom_filter' in partition:
                print(f"    Bloom filter size: {sys.getsizeof(partition['min'])} bytes")
            if 'min' in partition:
                print(f"    Min Value: {partition['min']}")
            if 'max' in partition:
                print(f"    Max Value: {partition['max']}")

            # Estimating metadata size (rough estimation)
            metadata_size = sys.getsizeof(partition)
            total_metadata_size += metadata_size
            total_data_size += partition['length']

            print(f"    Estimated Metadata Size: {metadata_size} bytes")

        # Calculating overhead
        file_size = os.path.getsize(self.filename)
        overhead = file_size - total_data_size - total_metadata_size
        print("\nSummary:")
        print(f"  Total Data Size: {total_data_size} bytes")
        print(f"  Total Metadata Size: {total_metadata_size} bytes")
        print(f"  Overhead (file system, etc.): {overhead} bytes")
        print(f"  Total File Size: {file_size} bytes")


class PagedFile(MappedFile):
    def __init__(self, filename, mode, file_metadata=None, footer_serializer=None, page_size=100, format='json'):
        super().__init__(filename, mode, file_metadata, footer_serializer)
        self.page_size = page_size
        self.format = format
        self.buffer = pd.DataFrame()
        self.page_count = 0
        self.schema = []
        if 'schema' not in self.file_metadata:
            self.file_metadata['schema'] = None
        else:
            self.schema = self.file_metadata['schema']

        if 'page_count' not in self.file_metadata:
            self.file_metadata['page_count'] = 0
        else:
            self.page_count = self.file_metadata['page_count']

    def write_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")

        if self.file_metadata['schema'] is None:
            self.file_metadata['schema'] = data.dtypes.apply(lambda x: str(x)).to_dict()

        self.buffer = pd.concat([self.buffer, data], ignore_index=True)

        while len(self.buffer) >= self.page_size:
            self.write_page()

    def write_page(self):
        page_data = self.buffer.iloc[:self.page_size]
        serialized_data = self.serialize_page(page_data)
        self.write_partition(serialized_data)
        self.buffer = self.buffer.iloc[self.page_size:]
        self.page_count += 1
        self.file_metadata['page_count'] = self.page_count

    def serialize_page(self, page_data):
        if self.format == 'json':
            import json
            page_data = page_data.fillna("")
            page_data["Transaction_ID"] = page_data["Transaction_ID"].astype("int")
            with open("sdsdqs.json", "w") as f:

                json.dump(page_data.to_dict(orient='records'), f, cls=EncodeFromNumpy, indent=4, allow_nan=None)
            return json.dumps(page_data.to_dict(orient='records'), cls=EncodeFromNumpy, allow_nan=None).encode('utf-8')
        elif self.format == 'msgpack':
            import msgpack
            return msgpack.packb(page_data.to_dict(orient='records'))
        elif self.format == 'parquet':
            import pyarrow.parquet as pq
            from pyarrow import Table
            table = Table.from_pandas(page_data)
            buf = pq.serialize_table(table)
            return buf.to_pybytes()
        else:
            raise ValueError("Unsupported format specified.")

    def read_page(self, page_index):
        if page_index < 0 or page_index >= self.page_count:
            raise IndexError("Page index out of range.")
        serialized_data = self.read_partition(page_index)
        return self.deserialize_page(serialized_data)

    def deserialize_page(self, serialized_data):
        if self.format == 'json':
            import json
            data_dict = json.loads(serialized_data)
            return data_dict
        elif self.format == 'msgpack':
            import msgpack
            data_dict = msgpack.unpackb(serialized_data, raw=False)
            return data_dict
        elif self.format == 'parquet':
            import pyarrow.parquet as pq
            from pyarrow import Table
            buf = pq.BufferReader(serialized_data)
            table = pq.read_table(buf)
            return table.to_pandas().to_dict(orient='records')
        else:
            raise ValueError("Unsupported format specified.")

    def close(self):
        if not self.buffer.empty:
            serialized_data = self.serialize_page(self.buffer)
            self.write_partition(serialized_data)
            self.buffer = pd.DataFrame()  # Clear the buffer after writing
            self.page_count += 1
            self.file_metadata['page_count'] = self.page_count
        super().close()
