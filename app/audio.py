"""Pemutar suara.

Setiap kegagalan menurunkan kemampuan, tidak menghentikan aplikasi: tanpa
sound device, tanpa pygame, atau tanpa file, program tetap jalan dengan efek
visual saja.
"""

import os


def _default_mixer_init():
    import pygame

    pygame.mixer.init()


def _default_sound_factory(path):
    import pygame

    return pygame.mixer.Sound(path)


class AudioPlayer:
    """Memutar satu suara per kunci, tanpa menumpuk kalau masih berbunyi.

    `mixer_init` dan `sound_factory` bisa disuntik supaya test tidak butuh
    sound device sungguhan.
    """

    def __init__(self, paths, mixer_init=None, sound_factory=None):
        self.enabled = False
        self.muted = False
        self.sounds = {}
        self._channels = {}
        mixer_init = mixer_init or _default_mixer_init
        sound_factory = sound_factory or _default_sound_factory
        try:
            mixer_init()
            self.enabled = True
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] audio dimatikan (mixer gagal init): {exc}")
            return
        for key, path in paths.items():
            if not os.path.exists(path):
                print(f"[warn] sound tidak ditemukan: {path}")
                continue
            try:
                self.sounds[key] = sound_factory(path)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] gagal load sound {path}: {exc}")

    def play(self, key):
        """Mainkan sekali. Tidak mengulang kalau masih berbunyi (anti-spam)."""
        if not self.enabled or self.muted or key not in self.sounds:
            return
        ch = self._channels.get(key)
        if ch is not None and ch.get_busy():
            return
        try:
            self._channels[key] = self.sounds[key].play()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] gagal play {key}: {exc}")

    def stop(self, key):
        """Hentikan saat gestur dilepas, biar tidak terus berbunyi."""
        ch = self._channels.get(key)
        if ch is not None and ch.get_busy():
            ch.stop()

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            for key in list(self._channels):
                self.stop(key)
        return self.muted

    def close(self):
        if not self.enabled:
            return
        try:
            import pygame

            pygame.mixer.quit()
        except Exception:  # noqa: BLE001
            pass
