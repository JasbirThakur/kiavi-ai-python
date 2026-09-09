import logging
import re
import sys

class SensitiveDataFilter(logging.Filter):
    """Filters out passwords, tokens, and API keys from logs."""
    SENSITIVE_PATTERNS = [
        (re.compile(r'(password[\'\":\s=]+)([^\s,\'"}]+)', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(api[_-]?key[\'\":\s=]+)([^\s,\'"}]+)', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(secret[\'\":\s=]+)([^\s,\'"}]+)', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(bearer\s+)([A-Za-z0-9\-_.]+)', re.IGNORECASE), r'\1[REDACTED]'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern, repl in self.SENSITIVE_PATTERNS:
                record.msg = pattern.sub(repl, record.msg)
        return True

def setup_logger(name: str = "kiavi") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = setup_logger()

