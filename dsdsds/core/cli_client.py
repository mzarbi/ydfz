import click
from Pyro5.client import Proxy
from inspect import getmembers, ismethod, signature
from utils.logging_utils import logger


def create_click_command(method_name, method):
    params = signature(method).parameters

    @click.command(name=method_name.replace('_', '-'))
    @click.option('--host', default="localhost", help='Host for the KV Store server.')
    @click.option('--port', default=6666, help='Port for the KV Store server.')
    @click.option('--namespace', default="key_value_store", help='Namespace for the KV Store server.')
    def command(host, port, namespace, **kwargs):
        """Dynamically generated command."""
        with Proxy(f"PYRO:{namespace}@{host}:{port}") as proxy:
            method_to_call = getattr(proxy, method.__name__)
            result = method_to_call(**kwargs)
            logger.info(result)

    # Dynamically add options based on method parameters
    for param in params.values():
        if param.name not in ('self', 'args', 'kwargs'):
            default = param.default if param.default is not param.empty else None
            command = click.option(f'--{param.name}', default=default, help=f'{param.name} for kv store')(command)

    return command


def register_dynamic_commands(cli_group, kv_store_class):
    for name, method in getmembers(kv_store_class):
        if getattr(method, '_pyroExposed', False):
            cli_group.add_command(create_click_command(name, method))
