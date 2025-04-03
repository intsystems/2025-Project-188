from enum import IntEnum


class TimeRange(IntEnum):
    """
    Enum representing common time ranges.
    """
    WEEK = 7
    MONTH = 30
    YEAR = 365

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value