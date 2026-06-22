import re
import logging
import json
import time

def clean_base64_image(base64_str):
    """Strip base64 data-uri header and validate format"""
    if not base64_str:
        return None
        
    # Strip whitespace
    base64_str = base64_str.strip()
    
    # Strip standard data URL prefixes
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]
        
    # Ensure it only contains valid base64 characters
    if not re.match(r'^[A-Za-z0-9+/=]+$', base64_str):
        return None
        
    return base64_str

class JSONFormatter(logging.Formatter):
    """Format logs as single-line JSON for structured cloud logging"""
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        # Add custom extra parameters if present
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
            
        return json.dumps(log_entry)

def setup_logger(name="object_detection"):
    """Setup structured logger sending JSON to stdout"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()
