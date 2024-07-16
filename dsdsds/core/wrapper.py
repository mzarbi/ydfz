import hashlib
import json
from inspect import getmembers, signature
from random import shuffle, randint
from Pyro5.client import Proxy
from storage import AbstractKVStore, SecretsKVStore
from utils.logging_utils import logger


class KVProxy:
    def __init__(self, exposed_class=AbstractKVStore, host='localhost', port=6666, namespace="key_value_store"):
        self.host = host
        self.port = port
        self.namespace = namespace
        self.exposed_class = exposed_class
        for name, method in getmembers(self.exposed_class):
            if getattr(method, '_pyroExposed', False):
                self.create_wrapped_method(name, method)

    def create_wrapped_method(self, method_name, method):
        def wrapper(*args, **kwargs):
            with Proxy(f"PYRO:{self.namespace}@{self.host}:{self.port}") as proxy:
                method_to_call = getattr(proxy, method_name)
                return method_to_call(*args, **kwargs)

        # Set the dynamically created wrapper as an instance method
        setattr(self, method_name, wrapper)


class BKVProxy:
    def __init__(self, config, exposed_class=AbstractKVStore, namespace="key_value_store"):
        self.hosts = config
        self.namespace = namespace
        self.exposed_class = exposed_class
        for name, method in getmembers(self.exposed_class):
            if getattr(method, '_pyroExposed', False):
                self.create_wrapped_method(name, method)

    def create_wrapped_method(self, method_name, method):
        def wrapper(*args, **kwargs):
            hosts_list = list(self.hosts.values())
            shuffle(hosts_list)
            for host in hosts_list:
                try:
                    with Proxy(f"PYRO:{self.namespace}@{host['host']}:{host['port']}") as proxy:
                        method_to_call = getattr(proxy, method_name)
                        if method_name == 'add_key':
                            proxy.add_key(*args, value="vvv", ttl=3600)
                        else:
                            return method_to_call(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to connect to {host['host']}:{host['port']} with error: {type(e)}: {e}")
                    continue
            logger.warning("All hosts have been exhausted and the method call failed.")

        # Set the dynamically created wrapper as an instance method
        setattr(self, method_name, wrapper)

    def cache_result(self, store_name, ttl=3600*10, readonly=False):
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Create a unique key based on the function name and arguments
                key = self._create_cache_key(func.__name__, *args, **kwargs)

                # Try to get the value from the KV store
                try:
                    result = self.get_key(store_name, key)
                    if result is not None:
                        logger.info(f"Cache hit for key: {key}")
                        return result
                except Exception as e:
                    logger.error(f"Failed to fetch key {key} from store {store_name}: {e}")

                # Compute the result as it was not found in the cache
                result = func(*args, **kwargs)

                # Store the result in the KV store
                try:
                    self.add_key(store_name, key, value=result, ttl=ttl, readonly=readonly)
                    logger.info(f"Stored result for key: {key}")
                except Exception as e:
                    logger.warning(f"Failed to store key {key} in store {store_name}: {e}")

                return result

            return wrapper

        return decorator

    def _create_cache_key(self, func_name, *args, **kwargs):
        # Create a unique key based on the function name and arguments
        key_string = json.dumps({'func_name': func_name, 'args': args, 'kwargs': kwargs}, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode('utf-8')).hexdigest()
        return key_hash

    def list_methods(self):
        logger.info("Available commands")
        for name, method in getmembers(self.exposed_class):
            if getattr(method, '_pyroExposed', False):
                logger.info(f"\t - {name}: {signature(method)}")


if __name__ == '__main__':
    # Example usage
    config = {
        #'host1': {'host': 'MSI', 'port': 6666},
        'host2': {'host': 'MSI', 'port': 6661},
    }

    k = BKVProxy(config, exposed_class=SecretsKVStore)

    @k.cache_result(store_name='secrets')
    def my_function(a, b, dd=""):
        return a + b
    print(my_function(1, 1))
    print(my_function(1, 1))
    """for tmp in range(1000):
        print(my_function(1, randint(0,100)))"""
