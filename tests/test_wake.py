from hermes_voice.wake import WakeGate


class _Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_gate_starts_asleep():
    gate = WakeGate()
    assert gate.awake is False
    assert gate.expire(False) is False


def test_wake_opens_the_window():
    clock = _Clock()
    gate = WakeGate(window_s=8, clock=clock)
    gate.wake()
    assert gate.awake is True
    clock.now = 7.9
    assert gate.expire(False) is False
    assert gate.awake is True
    clock.now = 8
    assert gate.expire(False) is True
    assert gate.awake is False


def test_touch_restarts_the_window():
    clock = _Clock()
    gate = WakeGate(window_s=8, clock=clock)
    gate.wake()
    clock.now = 7
    gate.touch()
    clock.now = 14
    assert gate.expire(False) is False
    assert gate.awake is True
    clock.now = 15
    assert gate.expire(False) is True
    assert gate.awake is False


def test_expire_is_blocked_while_busy():
    clock = _Clock()
    gate = WakeGate(window_s=8, clock=clock)
    gate.wake()
    clock.now = 8
    assert gate.expire(True) is False
    assert gate.awake is True
    assert gate.expire(False) is True
    assert gate.awake is False


def test_sleep_closes_the_gate():
    clock = _Clock()
    gate = WakeGate(window_s=8, clock=clock)
    gate.wake()
    gate.sleep()
    assert gate.awake is False
    clock.now = 1
    assert gate.expire(False) is False
    gate.wake()
    clock.now = 9
    assert gate.expire(False) is True


def test_touch_while_asleep_does_not_open_the_gate():
    clock = _Clock()
    gate = WakeGate(window_s=8, clock=clock)
    gate.touch()
    clock.now = 100
    assert gate.awake is False
    assert gate.expire(False) is False
