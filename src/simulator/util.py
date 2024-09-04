import logging
from datetime import datetime


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
        f"tmp_result_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
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
        ts = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()  # float
        if ms:
            return str(int(ts * 1000))
        else:
            return str(int(ts))
    except ValueError:
        try:
            ts = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp()  # float
            if ms:
                return str(int(ts * 1000))
            else:
                return str(int(ts))
        except ValueError:
            try:
                ts = datetime.strptime(date_str, "%Y-%m-%d").timestamp()  # float
                if ms:
                    return str(int(ts * 1000))
                else:
                    return str(int(ts))
            except ValueError:
                logger.error(f"Invalid date string: {date_str}")
                raise ValueError(f"Invalid date string: {date_str}")

def timestamp2date(timestamp_ms: str) -> str:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000).strftime("%Y-%m-%d %H:%M:%S.%f")


if __name__ == "__main__":
    logger.info(date2timestamp("2020-01-01 00:00:00.231"))
    logger.info(timestamp2date(date2timestamp("2020-01-01 00:00:00.231")))
    logger.info(date2timestamp("2020-01-01"))
    logger.info(timestamp2date(date2timestamp("2020-01-01")))
    logger.info(date2timestamp("2020-01-01 01:01:23"))
    logger.info(timestamp2date(date2timestamp("2020-01-01 01:01:23")))
