import logging
import pandas as pd


def setup_logger(logger_name) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] - [%(levelname)s] - %(message)s")
    )
    logger.addHandler(console_handler)

    # copy the log to a file
    file_handler = logging.FileHandler(
        f"tmp_result_{pd.Timestamp.now():%Y%m%d%H%M%S}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] - [%(levelname)s] - %(message)s")
    )
    logger.addHandler(file_handler)
    return logger


logger = setup_logger(__name__)


def date2timestamp(date_str: str, ms=True) -> str:
    try:
        ts = pd.Timestamp(date_str).timestamp()  # float
        if ms:
            return str(int(ts * 1000))
        else:
            return str(int(ts))
    except ValueError:
        logger.error(f"Invalid date string: {date_str}")
        raise ValueError(f"Invalid date string: {date_str}")


if __name__ == "__main__":
    print(date2timestamp("2020-01-01 00:00:00"))
    print(date2timestamp("2020-01-01"))
    print(date2timestamp("2020-01-01 01:01:23"))
