from construct import Struct, Int32ul, Flag, Prefixed, GreedyString, If, Lazy, GreedyRange, Pickled, Bytes
import pickle
import msgpack
import json

from ydfz.utils import BloomFilterAdapter


class Serializer:
    """
    Base class for serialization and deserialization.
    """

    def serialize(self, obj):
        """
        Serialize an object.

        Parameters:
            obj: The object to serialize.

        Returns:
            bytes: Serialized data.
        """
        raise NotImplementedError

    def deserialize(self, data):
        """
        Deserialize data.

        Parameters:
            data (bytes): The data to deserialize.

        Returns:
            object: The deserialized object.
        """
        raise NotImplementedError


class PickleSerializer(Serializer):
    """
    Serializer using pickle for serialization and deserialization.
    """

    def serialize(self, obj):
        """
        Serialize an object using pickle.

        Parameters:
            obj: The object to serialize.

        Returns:
            bytes: Serialized data.
        """
        return pickle.dumps(obj)

    def deserialize(self, data):
        """
        Deserialize data using pickle.

        Parameters:
            data (bytes): The data to deserialize.

        Returns:
            object: The deserialized object.
        """
        return pickle.loads(data)


class MsgpackSerializer(Serializer):
    """
    Serializer using MessagePack for serialization and deserialization.
    """

    def serialize(self, obj):
        """
        Serialize an object using MessagePack.

        Parameters:
            obj: The object to serialize.

        Returns:
            bytes: Serialized data.
        """
        return msgpack.packb(obj, use_bin_type=True)

    def deserialize(self, data):
        """
        Deserialize data using MessagePack.

        Parameters:
            data (bytes): The data to deserialize.

        Returns:
            object: The deserialized object.
        """
        return msgpack.unpackb(data, raw=False, strict_map_key=False)


class MetadataSerializer(Serializer):
    """
    Serializer for metadata using Construct library.
    """

    def __init__(self):
        self.data_structure = GreedyRange(Struct(
            "offset" / Int32ul,
            "shard" / Int32ul,
            "length" / Int32ul,
            "name" / Prefixed(Int32ul, GreedyString("utf8")),
            "series_len" / Int32ul,
            "dtype" / Prefixed(Int32ul, GreedyString("utf8")),
            "mask_flag" / Int32ul,
            "mask_bytes_length" / Int32ul,
            "mask_bytes" / Prefixed(Int32ul, Bytes(lambda this: this.mask_bytes_length)),
            "compression" / Prefixed(Int32ul, GreedyString("utf8")),
            "categories" / Pickled,
            "compute_stats" / Flag,
            "min" / If(lambda this: this.compute_stats, Pickled),
            "max" / If(lambda this: this.compute_stats, Pickled),
            "bloom_filter_length" / If(lambda this: this.compute_stats, Int32ul),
            "bloom_filter" / If(lambda this: this.compute_stats,
                                Lazy(BloomFilterAdapter(Bytes(lambda this: this.bloom_filter_length))))
        ))

    def serialize(self, obj):
        """
        Serialize metadata.

        Parameters:
            obj: The metadata object.

        Returns:
            bytes: Serialized metadata.
        """
        return self.data_structure.build(obj)

    def deserialize(self, data):
        """
        Deserialize metadata.

        Parameters:
            data (bytes): Serialized metadata.

        Returns:
            object: Deserialized metadata.
        """
        return self.data_structure.parse(data)


class JsonSerializer(Serializer):
    """
    Serializer using JSON for serialization and deserialization.
    """

    def serialize(self, obj):
        """
        Serialize an object using JSON.

        Parameters:
            obj: The object to serialize.

        Returns:
            bytes: Serialized data.
        """
        return json.dumps(obj).encode('utf-8')

    def deserialize(self, data):
        """
        Deserialize data using JSON.

        Parameters:
            data (bytes): The data to deserialize.

        Returns:
            object: The deserialized object.
        """
        return json.loads(data.decode('utf-8'))
