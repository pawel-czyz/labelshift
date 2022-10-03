"""Creates a Timer class, a convenient thing to measure the elapsed time."""
import time


class Timer:
    """Class which can be used to measure elapsed experiment time.

    Example:
        >>> timer = Timer()
        >>> timer.reset()
        >>> elapsed_time = timer.check()

    Note:
        It is *not* safe to use it together inside parallel operations.
    """

    def __init__(self) -> None:
        """Initializes and resets the timer."""
        self._t0: float = time.time()

    def reset(self) -> None:
        """Resets the timer."""
        self._t0 = time.time()

    def check(self) -> float:
        """Returns the elapsed time since the last reset."""
        return time.time() - self._t0
