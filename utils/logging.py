import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_str: Optional[str] = None,
    log_file: Optional[str] = None,
):
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
    )
