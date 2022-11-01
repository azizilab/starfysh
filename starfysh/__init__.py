import logging

# Configure global logging format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

LOGGER = logging.getLogger('Starfysh')
