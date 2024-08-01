import os
import signal
import socket
import click
from Pyro5.server import Daemon

from core.cli_client import register_dynamic_commands
from core.storage import SecretsKVStore
from utils.logging_utils import logger


@click.group()
def cli():
    """Command line interface for managing the KV Store."""
    pass


@cli.command()  # Marks the function as a command within the CLI group
@click.option('--host', default=os.environ.get("KV_DEFAULT_HOST", socket.gethostname()), help='Host for the KV Store server.')
@click.option('--port', default=os.environ.get("KV_DEFAULT_PORT", 6666), type=int, help='Port for the KV Store server.')
@click.option('--status-ttl', default=os.environ.get("KV_DEFAULT_TTL", 3600 * 10), is_flag=False, help='The default ttl value for keys')
@click.option('--cleanup_frequency', default=os.environ.get("KV_DEFAULT_CLEANUP_FREQUENCY", 10), is_flag=False, help='The cleanup frequency in seconds')
@click.option('--backup_dir', default=os.environ.get("KV_BACKUP_DIR", r""), is_flag=False, help='The backup directory')
@click.option('--max_backups', default=os.environ.get("KV_DEFAULT_MAX_BACKUPS", 10), is_flag=False, help='The maximum number of rotating ackup files')
@click.option('--use_backup', default=os.environ.get("KV_DEFAULT_USE_BACKUP", False), is_flag=True, help='Use the latest backup file')
def start_secrets_server(host, port, status_ttl, cleanup_frequency, backup_dir, max_backups, use_backup):
    """Starts the KV Store server."""
    daemon = Daemon(host=host, port=port)
    store = SecretsKVStore(status_ttl=status_ttl, cleanup_frequency=cleanup_frequency, backup_dir=backup_dir, max_backups=max_backups, use_backup=use_backup)
    uri = daemon.register(store, objectId="key_value_store")
    logger.info(f"Service started. Object uri = {uri}")

    def signal_handler(sig, frame):
        logger.info('Signal received, shutting down...')
        store.shutdown()
        daemon.shutdown()
        logger.info("Server has been shut down.")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        daemon.requestLoop()
    finally:
        daemon.close()

register_dynamic_commands(cli, SecretsKVStore)

if __name__ == "__main__":
    cli()
