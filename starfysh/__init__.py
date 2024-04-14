import os
import multiprocessing
import logging

n_cores = multiprocessing.cpu_count()
os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)

# Configure global logging format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

LOGGER = logging.getLogger('Starfysh')
