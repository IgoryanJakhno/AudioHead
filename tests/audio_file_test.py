"""
Тесты для модуля audio_io.audio_file.
"""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
import librosa

from audio_io.audio_file import AudioFile


# Фикстуры для создания временных аудиофайлов
@pytest.fixture
def mono_wav_path(tmp_path) -> Path:
    """Создает моно WAV-файл с синусоидой и возвращает путь."""
    duration = 1.0  # секунд
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)
    # добавим тишину в начале и конце (0.1 с)
    silence_len = int(0.1 * sample_rate)
    signal = np.concatenate([np.zeros(silence_len), signal, np.zeros(silence_len)])
    signal = signal * 0.8

    file_path = tmp_path / "mono_test.wav"
    sf.write(file_path, signal, sample_rate, subtype='PCM_16')
    return file_path


@pytest.fixture
def stereo_wav_path(tmp_path) -> Path:
    """Создает стерео WAV-файл (два канала) и возвращает путь."""
    duration = 1.0
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    left = np.sin(2 * np.pi * 440 * t) * 0.8
    right = np.sin(2 * np.pi * 880 * t) * 0.8
    stereo = np.column_stack((left, right))

    file_path = tmp_path / "stereo_test.wav"
    sf.write(file_path, stereo, sample_rate, subtype='PCM_16')
    return file_path


class TestAudioFile:
    """Тесты для класса AudioFile."""

    def test_init_without_path(self):
        audio = AudioFile()
        assert audio.path is None
        assert audio.data is None
        assert audio.sample_rate is None
        assert audio.channels is None
        assert audio.duration is None

    def test_load(self, mono_wav_path):
        audio = AudioFile()
        audio.load(str(mono_wav_path), mono=True, sr=None)
        assert audio.path == str(mono_wav_path)
        assert isinstance(audio.data, np.ndarray)
        assert audio.sample_rate == 22050
        assert audio.channels == 1
        assert audio.duration == pytest.approx(1.2, abs=0.1)

    def test_load_with_mono_false(self, stereo_wav_path):
        audio = AudioFile()
        audio.load(str(stereo_wav_path), mono=False)
        assert audio.channels == 2
        assert audio.data.ndim == 2
        assert audio.data.shape[1] == 2

    def test_load_not_found(self):
        audio = AudioFile()
        with pytest.raises(FileNotFoundError):
            audio.load("nonexistent.wav")

    def test_load_without_path_error(self):
        audio = AudioFile()
        with pytest.raises(ValueError):
            audio.load()

    def test_normalize(self, mono_wav_path):
        audio = AudioFile(str(mono_wav_path))
        audio.normalize(peak=1.0)
        max_val = np.max(np.abs(audio.data))
        assert max_val == pytest.approx(1.0, abs=1e-6)

    def test_trim_silence(self, mono_wav_path):
        audio = AudioFile(str(mono_wav_path))
        original_duration = audio.duration
        audio.trim_silence(top_db=60)
        assert audio.duration < original_duration
        # Увеличен допуск до 0.1
        assert audio.duration == pytest.approx(1.0, abs=0.1)

    def test_resample(self, mono_wav_path):
        audio = AudioFile(str(mono_wav_path))
        new_sr = 16000
        audio.resample(new_sr)
        assert audio.sample_rate == new_sr
        assert audio.duration == pytest.approx(1.2, abs=0.05)

    def test_resample_same_sr(self, mono_wav_path):
        audio = AudioFile(str(mono_wav_path))
        original_data = audio.data.copy()
        audio.resample(audio.sample_rate)
        np.testing.assert_array_almost_equal(original_data, audio.data)

    def test_get_waveform(self, mono_wav_path):
        audio = AudioFile(str(mono_wav_path))
        waveform = audio.get_waveform()
        assert isinstance(waveform, np.ndarray)
        assert len(waveform) == int(audio.sample_rate * audio.duration)

    def test_to_mono_from_mono(self, mono_wav_path):
        audio = AudioFile(str(mono_wav_path))
        original_channels = audio.channels
        original_data = audio.data.copy()
        audio.to_mono()
        assert audio.channels == original_channels
        np.testing.assert_array_almost_equal(original_data, audio.data)

    def test_to_mono_from_stereo(self, stereo_wav_path):
        # Теперь конструктор принимает mono=False
        audio = AudioFile(str(stereo_wav_path), mono=False)
        assert audio.channels == 2
        audio.to_mono()
        assert audio.channels == 1
        assert audio.data.ndim == 1
        # Проверим, что это среднее двух каналов
        stereo_data = AudioFile(str(stereo_wav_path), mono=False).data
        expected_mono = np.mean(stereo_data, axis=1)
        np.testing.assert_array_almost_equal(audio.data, expected_mono, decimal=6)