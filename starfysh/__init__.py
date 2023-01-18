import logging

# Configure global logging format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

LOGGER = logging.getLogger('Starfysh')
