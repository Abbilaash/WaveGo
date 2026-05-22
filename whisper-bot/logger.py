import os
import logging

LOG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "logs")

logger = logging.getLogger("WAVEGO")
logger.setLevel(logging.INFO)

# Setup file handler if not already present
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def log_action(category, action, details=None):
    msg = f"[{category.upper()}] {action}"
    if details:
        msg += f" - Details: {details}"
    logger.info(msg)
