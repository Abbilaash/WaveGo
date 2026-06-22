import os
import logging
from datetime import datetime, timedelta

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "logs"))
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

LAST_PRUNE_CHECK = None

def prune_logs_if_needed():
    if not os.path.exists(LOG_FILE):
        return
    
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            first_line = f.readline()
        if not first_line:
            return
        
        # Try to parse the date from the start of the first line (e.g. '2026-05-23 01:28:48')
        try:
            first_date = datetime.strptime(first_line[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return
            
        now = datetime.now()
        if now - first_date >= timedelta(days=2):
            cutoff_date = now - timedelta(days=2)
            
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                try:
                    line_date = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                    if line_date >= cutoff_date:
                        new_lines.append(line)
                except Exception:
                    new_lines.append(line)
            
            # Temporarily release file handlers to avoid locking issues on some operating systems
            handlers_to_restore = []
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
                handlers_to_restore.append(handler)
                
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
                
            # Restore handlers
            for handler in handlers_to_restore:
                logger.addHandler(handler)
                
    except Exception as e:
        print(f"Error during log pruning: {e}")

def log_action(category, action, details=None):
    global LAST_PRUNE_CHECK
    now = datetime.now()
    if LAST_PRUNE_CHECK is None or (now - LAST_PRUNE_CHECK).total_seconds() > 300:
        LAST_PRUNE_CHECK = now
        prune_logs_if_needed()
        
    msg = f"[{category.upper()}] {action}"
    if details:
        msg += f" - Details: {details}"
    logger.info(msg)
