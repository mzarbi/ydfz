import gzip
import lz4
import snappy


class Compressor:
    """
    Utility class for data compression and decompression.
    """

    @staticmethod
    def compress_data(data_bytes, compression):
        """
        Compress data using the specified compression algorithm.

        Parameters:
            data_bytes (bytes): Data to compress.
            compression (str): Compression algorithm ('gzip', 'snappy', 'lz4').

        Returns:
            bytes: Compressed data.
        """
        if compression == 'gzip':
            return gzip.compress(data_bytes)
        elif compression == 'snappy':
            return snappy.compress(data_bytes)
        elif compression == 'lz4':
            return lz4.frame.compress(data_bytes)
        return data_bytes

    @staticmethod
    def decompress_data(compressed_data, compression):
        """
        Decompress data using the specified compression algorithm.

        Parameters:
            compressed_data (bytes): Compressed data.
            compression (str): Compression algorithm ('gzip', 'snappy', 'lz4').

        Returns:
            bytes: Decompressed data.
        """
        if compression == 'gzip':
            return gzip.decompress(compressed_data)
        elif compression == 'snappy':
            return snappy.decompress(compressed_data)
        elif compression == 'lz4':
            return lz4.frame.decompress(compressed_data)
        return compressed_data
