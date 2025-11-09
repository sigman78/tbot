from datetime import datetime, timedelta


def next_timestamp_generator(
    start_time: datetime | None = None, interval: timedelta = timedelta(microseconds=1)
):
    """Create a timestamp generator function for cleaner test code.

    Args:
        start_time: Starting timestamp (defaults to current time)
        interval: Time increment between timestamps

    Returns:
        Function that returns the next sequential timestamp
    """
    current = start_time or datetime.utcnow()

    def next_timestamp():
        nonlocal current
        result = current
        current += interval
        return result

    return next_timestamp
