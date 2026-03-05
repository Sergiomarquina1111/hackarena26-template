import logging, sys
from pathlib import Path

def setup_logging(log_file: str, level: str = "INFO") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    for h in (logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        root.addHandler(h)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
