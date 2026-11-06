"""Wake window. A small local model hears the phrase; this only keeps the clock."""

import time


class WakeGate:
    """Stays open for a few seconds after the wake phrase, then goes back to sleep."""

    def __init__(self, window_s: float = 8, clock=time.monotonic) -> None:
        self._window_s = window_s
        self._clock = clock
        self.awake = False
        self._deadline: float | None = None

    def wake(self) -> None:
        self.awake = True
        self._deadline = self._clock() + self._window_s

    def touch(self) -> None:
        if self.awake:
            self._deadline = self._clock() + self._window_s

    def expire(self, busy: bool) -> bool:
        if not self.awake or busy or self._deadline is None:
            return False
        if self._clock() < self._deadline:
            return False
        self.sleep()
        return True

    def sleep(self) -> None:
        self.awake = False
        self._deadline = None
