"""
Тесты для модуля audio_io.audio_file.
"""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

from audio_io.audio_file import AudioFile

# Фикстуры для создания временных аудиофайлов
@pytest.fixture
def mono_wav_path(tmp_path) -> Path:
    """Создает моно WAV-файл с синусоидой и возвращает путь."""
    duration = 1.0  # секунд
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # синусоида 440 Гц с небольшой тишиной в начале/конце (для теста trim)
    signal = np.sin(2 * np.pi * 440 * t)
    # добавим тишину в начале и конце (0.1 с)
    silence_len = int(0.1 * sample_rate)
    signal = np.concatenate([np.zeros(silence_len), signal, np.zeros(silence_len)])
    # нормализуем до +/-0.8
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
        """Проверка инициализации без пути."""
        audio = AudioFile()
        assert audio.path is None
        assert audio.data is None
        assert audio.sample_rate is None
        assert audio.channels is None
        assert audio.duration is None

    def test_load(self, mono_wav_path):
        """Проверка загрузки файла."""
        audio = AudioFile()
        audio.load(str(mono_wav_path), mono=True, sr=None)
        assert audio.path == str(mono_wav_path)
        assert isinstance(audio.data, np.ndarray)
        assert audio.sample_rate == 22050
        assert audio.channels == 1
        assert audio.duration == pytest.approx(1.2, abs=0.1)  # 1.0 + 0.2 тишины

    def test_load_with_mono_false(self, stereo_wav_path):
        """Загрузка стерео с сохранением каналов."""
        audio = AudioFile()
        audio.load(str(stereo_wav_path), mono=False)
        assert audio.channels == 2
        assert audio.data.ndim == 2
        assert audio.data.shape[1] == 2

    def test_load_not_found(self):
        """Ошибка при отсутствии файла."""
        audio = AudioFile()
        with pytest.raises(FileNotFoundError):
            audio.load("nonexistent.wav")

    def test_load_without_path_error(self):
        """Ошибка, если путь не указан."""
        audio = AudioFile()
        with pytest.raises(ValueError):
            audio.load()

    def test_normalize(self, mono_wav_path):
        """Проверка нормализации сигнала."""
        audio = AudioFile(str(mono_wav_path))
        # исходный пик ~0.8, нормализуем до 1.0
        audio.normalize(peak=1.0)
        max_val = np.max(np.abs(audio.data))
        assert max_val == pytest.approx(1.0, abs=1e-6)

    def test_trim_silence(self, mono_wav_path):
        """Проверка обрезки тишины."""
        audio = AudioFile(str(mono_wav_path))
        original_duration = audio.duration
        audio.trim_silence(top_db=60)
        # После обрезки длительность должна уменьшиться (было 1.2 с, станет ~1.0)
        assert audio.duration < original_duration
        # Проверим, что длительность близка к 1.0 с (сигнал без тишины)
        assert audio.duration == pytest.approx(1.0, abs=0.05)

    def test_resample(self, mono_wav_path):
        """Проверка передискретизации."""
        audio = AudioFile(str(mono_wav_path))
        new_sr = 16000
        audio.resample(new_sr)
        assert audio.sample_rate == new_sr
        # Длительность должна остаться примерно той же (с учётом точности)
        assert audio.duration == pytest.approx(1.2, abs=0.05)

    def test_resample_same_sr(self, mono_wav_path):
        """Передискретизация с той же частотой не меняет данные."""
        audio = AudioFile(str(mono_wav_path))
        original_data = audio.data.copy()
        audio.resample(audio.sample_rate)
        np.testing.assert_array_almost_equal(original_data, audio.data)

    def test_get_waveform(self, mono_wav_path):
        """Проверка возврата аудиоданных."""
        audio = AudioFile(str(mono_wav_path))
        waveform = audio.get_waveform()
        assert isinstance(waveform, np.ndarray)
        # Длина должна соответствовать sample_rate * duration
        assert len(waveform) == int(audio.sample_rate * audio.duration)

    def test_to_mono_from_mono(self, mono_wav_path):
        """Преобразование моно в моно ничего не меняет."""
        audio = AudioFile(str(mono_wav_path))
        original_channels = audio.channels
        original_data = audio.data.copy()
        audio.to_mono()
        assert audio.channels == original_channels
        np.testing.assert_array_almost_equal(original_data, audio.data)

    def test_to_mono_from_stereo(self, stereo_wav_path):
        """Преобразование стерео в моно (усреднение)."""
        audio = AudioFile(str(stereo_wav_path), mono=False)
        assert audio.channels == 2
        audio.to_mono()
        assert audio.channels == 1
        # Проверим, что данные стали одномерными
        assert audio.data.ndim == 1
        # Проверим, что это среднее двух каналов (приблизительно)
        # reload stereo to compare
        stereo_data = AudioFile(str(stereo_wav_path), mono=False).data
        expected_mono = np.mean(stereo_data, axis=1)
        np.testing.assert_array_almost_equal(audio.data, expected_mono, decimal=6)