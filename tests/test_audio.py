"""AudioPlayer harus jalan terus walau tidak ada sound device, tidak ada
pygame, atau file suaranya hilang. Test ini memakai stub, bukan device asli."""

from app.audio import AudioPlayer


class FakeChannel:
    def __init__(self):
        self.busy = True
        self.stopped = False

    def get_busy(self):
        return self.busy

    def stop(self):
        self.stopped = True
        self.busy = False


class FakeSound:
    def __init__(self):
        self.plays = 0
        self.channel = FakeChannel()

    def play(self):
        self.plays += 1
        return self.channel


def test_player_is_disabled_when_mixer_fails():
    def boom(*_a, **_k):
        raise RuntimeError("tidak ada audio device")

    p = AudioPlayer({}, mixer_init=boom)
    assert p.enabled is False
    p.play("blur")   # tidak boleh melempar
    p.stop("blur")
    p.close()


def test_missing_sound_file_is_skipped_not_fatal(tmp_path):
    p = AudioPlayer({"blur": str(tmp_path / "tidak-ada.wav")},
                    mixer_init=lambda: None, sound_factory=lambda _p: FakeSound())
    assert p.enabled is True
    assert "blur" not in p.sounds
    p.play("blur")   # tidak boleh melempar


def test_play_does_not_stack_while_still_sounding(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"x")
    sound = FakeSound()
    p = AudioPlayer({"blur": str(path)}, mixer_init=lambda: None,
                    sound_factory=lambda _p: sound)
    p.play("blur")
    p.play("blur")
    assert sound.plays == 1


def test_stop_stops_the_active_channel(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"x")
    sound = FakeSound()
    p = AudioPlayer({"blur": str(path)}, mixer_init=lambda: None,
                    sound_factory=lambda _p: sound)
    p.play("blur")
    p.stop("blur")
    assert sound.channel.stopped is True


def test_mute_silences_playback_and_stops_what_is_sounding(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"x")
    sound = FakeSound()
    p = AudioPlayer({"blur": str(path)}, mixer_init=lambda: None,
                    sound_factory=lambda _p: sound)
    p.play("blur")
    assert p.toggle_mute() is True
    assert sound.channel.stopped is True
    p.play("blur")
    assert sound.plays == 1        # dibisukan -> tidak main lagi
    assert p.toggle_mute() is False
