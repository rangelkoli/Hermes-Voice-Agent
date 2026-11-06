import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from hermes_voice.wake import MODEL_DIR, MODEL_URL, WakeError, WakeGate, ensure_model


class _Spotter:
    def __init__(self, hit=False):
        self.hit = hit
        self.fed: list[bytes] = []
        self.resets = 0

    def feed(self, frame: bytes) -> bool:
        self.fed.append(frame)
        return self.hit

    def reset(self) -> None:
        self.resets += 1


class _Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _archive(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_asleep_frames_go_nowhere():
    gate = WakeGate(_Spotter())
    assert gate.feed(b"a") == []
    assert gate.feed(b"b") == []
    assert gate.awake is False


def test_hit_returns_the_ring_ending_with_this_frame():
    spotter = _Spotter()
    gate = WakeGate(spotter)
    assert gate.feed(b"a") == []
    assert gate.feed(b"b") == []
    spotter.hit = True
    assert gate.feed(b"c") == [b"a", b"b", b"c"]
    assert gate.awake is True


def test_ring_is_limited_to_ring_ms():
    # 100 frames, ring_ms=300 -> 10 returned (30 ms frames).
    spotter = _Spotter()
    gate = WakeGate(spotter, ring_ms=300)
    frames = [bytes([index]) for index in range(100)]
    for frame in frames[:-1]:
        assert gate.feed(frame) == []
    spotter.hit = True
    assert gate.feed(frames[-1]) == frames[-10:]


def test_awake_frames_pass_through_and_skip_the_spotter():
    spotter = _Spotter(hit=True)
    gate = WakeGate(spotter)
    assert gate.feed(b"a") == [b"a"]
    seen = len(spotter.fed)
    assert gate.feed(b"b") == [b"b"]
    assert gate.feed(b"c") == [b"c"]
    assert len(spotter.fed) == seen
    assert gate.awake is True


def test_expire_closes_after_the_window_unless_busy():
    clock = _Clock()
    spotter = _Spotter(hit=True)
    gate = WakeGate(spotter, window_s=8, clock=clock)
    assert gate.expire(False) is False
    gate.feed(b"a")
    clock.now = 7.9
    assert gate.expire(False) is False
    assert gate.awake is True
    clock.now = 8
    assert gate.expire(True) is False
    assert gate.awake is True
    assert gate.expire(False) is True
    assert gate.awake is False


def test_touch_restarts_the_window():
    clock = _Clock()
    gate = WakeGate(_Spotter(hit=True), window_s=8, clock=clock)
    gate.feed(b"a")
    clock.now = 7
    gate.touch()
    clock.now = 14
    assert gate.expire(False) is False
    assert gate.awake is True
    clock.now = 15
    assert gate.expire(False) is True
    assert gate.awake is False


def test_sleep_clears_ring_and_resets_spotter():
    spotter = _Spotter()
    gate = WakeGate(spotter, ring_ms=3000)
    for index in range(20):
        assert gate.feed(bytes([index])) == []
    gate.sleep()
    assert gate.awake is False
    assert spotter.resets == 1
    spotter.hit = True
    assert gate.feed(b"new") == [b"new"]


def test_model_is_downloaded_verified_and_unpacked(tmp_path, monkeypatch):
    payload = _archive({
        f"{MODEL_DIR}/tokens.txt": b"tok\n",
        f"{MODEL_DIR}/bpe.model": b"bpe",
    })
    monkeypatch.setattr(
        "hermes_voice.wake.MODEL_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    seen = {}

    def download(url, dest):
        seen["url"] = url
        Path(dest).write_bytes(payload)

    result = ensure_model(tmp_path, download)
    assert seen["url"] == MODEL_URL
    assert result == tmp_path / MODEL_DIR
    assert (result / "tokens.txt").read_bytes() == b"tok\n"
    assert (result / "bpe.model").read_bytes() == b"bpe"
    assert [path.name for path in tmp_path.iterdir()] == [MODEL_DIR]


def test_bad_hash_raises_and_leaves_nothing(tmp_path):
    def download(url, dest):
        del url
        Path(dest).write_bytes(b"not the model")

    with pytest.raises(WakeError) as exc:
        ensure_model(tmp_path, download)
    assert "--open-mic" in str(exc.value)
    assert list(tmp_path.iterdir()) == []


def test_failed_download_raises_wake_error(tmp_path):
    def download(url, dest):
        del url
        Path(dest).write_bytes(b"partial")
        raise OSError("network down")

    with pytest.raises(WakeError) as exc:
        ensure_model(tmp_path, download)
    assert "--open-mic" in str(exc.value)
    assert list(tmp_path.iterdir()) == []


def test_existing_model_is_not_downloaded_again(tmp_path):
    model = tmp_path / MODEL_DIR
    model.mkdir()
    (model / "tokens.txt").write_text("keep", encoding="utf-8")

    def download(url, dest):
        del url, dest
        raise AssertionError("downloaded")

    assert ensure_model(tmp_path, download) == model
    assert (model / "tokens.txt").read_text(encoding="utf-8") == "keep"


def test_archive_with_unsafe_path_is_rejected(tmp_path, monkeypatch):
    payload = _archive({
        f"{MODEL_DIR}/tokens.txt": b"tok\n",
        "../escaped-wake-word.txt": b"nope",
    })
    monkeypatch.setattr(
        "hermes_voice.wake.MODEL_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )

    def download(url, dest):
        del url
        Path(dest).write_bytes(payload)

    with pytest.raises(WakeError) as exc:
        ensure_model(tmp_path, download)
    assert "--open-mic" in str(exc.value)
    assert not (tmp_path.parent / "escaped-wake-word.txt").exists()
    assert list(tmp_path.iterdir()) == []


def test_failed_unpack_after_tokens_txt_leaves_no_final_model_directory(tmp_path, monkeypatch):
    # ignore_errors can leave a partial tree; deleting the final directory is a no-op here.
    payload = _archive({f"{MODEL_DIR}/tokens.txt": b"tok\n"})
    monkeypatch.setattr(
        "hermes_voice.wake.MODEL_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )

    def extractall(self, path, *args, **kwargs):
        del self, args, kwargs
        partial = Path(path) / MODEL_DIR
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "tokens.txt").write_text("partial", encoding="utf-8")
        raise OSError("unpack stopped after tokens.txt")

    monkeypatch.setattr(tarfile.TarFile, "extractall", extractall)
    monkeypatch.setattr("hermes_voice.wake.shutil.rmtree", lambda *_args, **_kwargs: None)

    def download(url, dest):
        del url
        Path(dest).write_bytes(payload)

    with pytest.raises(WakeError) as exc:
        ensure_model(tmp_path, download)
    assert "--open-mic" in str(exc.value)
    assert not (tmp_path / MODEL_DIR).exists()
