import json
import hashlib
from random import shuffle, randint
from flask import Flask, Blueprint, request, jsonify
from Pyro5.client import Proxy
from inspect import getmembers, signature
from flasgger import Swagger, swag_from

from core.storage import SecretsKVStore
from core.wrapper import BKVProxy
from utils.logging_utils import logger


def create_flask_route(proxy_instance, method_name, method):
    params = signature(method).parameters

    def route_function():
        """Dynamically generated route."""
        kwargs = {param: request.args.get(param) for param in params if param not in ('self', 'args', 'kwargs')}
        result = getattr(proxy_instance, method_name)(**kwargs)
        logger.info(result)
        return jsonify(result)

    route_function.__name__ = method_name
    route_function.__doc__ = method.__doc__  # Include the docstring for documentation
    return route_function


def register_dynamic_routes(app, proxy_instance, url_prefix='/kv_store'):
    blueprint = Blueprint('kv_store', __name__)
    for name, method in getmembers(proxy_instance.exposed_class):
        if getattr(method, '_pyroExposed', False):
            route = create_flask_route(proxy_instance, name, method)
            endpoint = f"/{name.replace('_', '-')}"
            swag = {
                'tags': [url_prefix.strip('/')],
                'summary': method.__doc__.split('\n')[0] if method.__doc__ else name,
                'parameters': [
                    {
                        'name': param,
                        'in': 'query',
                        'type': 'string',
                        'required': False,
                        'description': f'{param} for kv store'
                    } for param in signature(method).parameters if param not in ('self', 'args', 'kwargs')
                ],
                'responses': {
                    '200': {
                        'description': 'Success'
                    }
                }
            }
            blueprint.add_url_rule(endpoint, view_func=swag_from(swag)(route), methods=['GET'])
    app.register_blueprint(blueprint, url_prefix=url_prefix)


if __name__ == '__main__':
    # Example usage:
    app = Flask(__name__)
    Swagger(app)

    # Configuration for BKVProxy
    config = {
        'host1': {'host': 'MSI', 'port': 6666},
        'host2': {'host': 'MSI', 'port': 6661},
    }

    # Initialize the BKVProxy with the SecretsKVStore class
    proxy_instance = BKVProxy(config, exposed_class=SecretsKVStore)

    # Register dynamic routes with the Flask app
    register_dynamic_routes(app, proxy_instance)

    app.run(debug=True)
