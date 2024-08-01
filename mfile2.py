import os
import zipfile
import pickle
import msgpack
import snappy


class PickleSerializer:
    @staticmethod
    def serialize(data):
        return pickle.dumps(data)

    @staticmethod
    def deserialize(data):
        return pickle.loads(data)


class MsgpackSerializer:
    @staticmethod
    def serialize(data):
        return msgpack.packb(data)

    @staticmethod
    def deserialize(data):
        return msgpack.unpackb(data)


class Compressor:
    @staticmethod
    def compress_data(data, algorithm):
        if algorithm == 'snappy':
            return snappy.compress(data)
        return data

    @staticmethod
    def decompress_data(data, algorithm):
        if algorithm == 'snappy':
            return snappy.uncompress(data)
        return data


class MappedFile:
    def __init__(self, filename, mode, file_metadata=None, footer_serializer=None):
        self.filename = filename
        self.mode = mode
        self.file = None
        self.metadata_file = None
        self.partitions = []  # In-memory cache for partitions data
        self.file_metadata = file_metadata if file_metadata else {}
        self.footer_serializer = footer_serializer() if footer_serializer else PickleSerializer()
        self.metadata_serializer = MsgpackSerializer()

        self.file_metadata['footer_serializer'] = self.footer_serializer.__class__.__name__

        # Open file based on the mode
        if mode == 'write':
            self.file = open(filename + '.data', 'wb+')
            self.metadata_file = open(filename + '.meta', 'wb+')
        elif mode == 'read' or mode == 'append':
            with zipfile.ZipFile(filename, 'r') as zipf:
                zipf.extractall()
            self.file = open(filename + '.data', 'rb+' if mode == 'append' else 'rb')
            self.metadata_file = open(filename + '.meta', 'rb')
            self._read_footer()
            if mode == 'append' and self.partitions:
                self.file.seek(self.partitions[0]['offset'])  # Assume first partition points to footer start
                self.file.truncate()

    def _read_footer(self):
        self.metadata_file.seek(-8, os.SEEK_END)
        footer_length = int.from_bytes(self.metadata_file.read(4), 'little')
        metadata_length = int.from_bytes(self.metadata_file.read(4), 'little')

        self.metadata_file.seek(-(footer_length + metadata_length + 8), os.SEEK_END)

        metadata_data = self.metadata_file.read(metadata_length)
        metadata_data = Compressor.decompress_data(metadata_data, 'snappy')
        self.file_metadata = self.metadata_serializer.deserialize(metadata_data)

        footer_data = self.metadata_file.read(footer_length)
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

            self.metadata_file.write(metadata_data)
            self.metadata_file.write(footer_data)
            self.metadata_file.write(footer_length.to_bytes(4, 'little'))
            self.metadata_file.write(metadata_length.to_bytes(4, 'little'))

            self.metadata_file.close()
            self.file.close()

            with zipfile.ZipFile(self.filename, 'w') as zipf:
                zipf.write(self.filename + '.data', arcname=os.path.basename(self.filename + '.data'))
                zipf.write(self.filename + '.meta', arcname=os.path.basename(self.filename + '.meta'))

            os.remove(self.filename + '.data')
            os.remove(self.filename + '.meta')
        else:
            self.metadata_file.close()
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # Handle exceptions if necessary
        return False


# Writing example
with MappedFile('example.zip', 'write') as mf:
    mf.write_partition(b'some data', {'name': 'partition1'})
    mf.write_partition(b'some more data', {'name': 'partition2'})

# Reading example
with MappedFile('example.zip', 'read') as mf:
    data = mf.read_partition(0)
    print(data)
