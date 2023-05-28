import logging
import sys

def get_logger(name: str):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)  # TODO: set via env

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)
    return log
