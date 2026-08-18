"""
Тесты для модуля analysis.features.
"""

import pytest
import numpy as np
import librosa
from analysis.features import FeatureExtractor


@pytest.fixture
def sine_audio():
    """Генерирует моно-синусоиду 440 Гц длительностью 1 сек с частотой 22050 Гц."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)
    return audio, sr


class TestFeatureExtractor:
    """Тесты для FeatureExtractor."""

    def test_stft_shape(self, sine_audio):
        audio, sr = sine_audio
        n_fft = 2048
        hop_length = 512
        stft = FeatureExtractor.stft(audio, n_fft=n_fft, hop_length=hop_length)
        expected_freq_bins = n_fft // 2 + 1
        expected_time_frames = 1 + len(audio) // hop_length
        assert stft.shape == (expected_freq_bins, expected_time_frames)
        assert np.iscomplexobj(stft)

    def test_spectrogram_shape(self, sine_audio):
        audio, sr = sine_audio
        spec = FeatureExtractor.spectrogram(audio)
        # Проверка, что значения в дБ
        assert spec.ndim == 2
        assert spec.dtype == np.float64
        # Максимум должен быть около 0 дБ (так как ref=np.max)
        assert np.max(spec) <= 0.0

    def test_chromagram_shape(self, sine_audio):
        audio, sr = sine_audio
        chroma = FeatureExtractor.chromagram(audio, sr)
        assert chroma.shape[0] == 12  # 12 тонов
        assert chroma.ndim == 2

    def test_mfcc_shape(self, sine_audio):
        audio, sr = sine_audio
        n_mfcc = 13
        mfcc = FeatureExtractor.mfcc(audio, sr, n_mfcc=n_mfcc)
        assert mfcc.shape[0] == n_mfcc
        assert mfcc.ndim == 2

    def test_onset_envelope_shape(self, sine_audio):
        audio, sr = sine_audio
        onset_env = FeatureExtractor.onset_envelope(audio, sr)
        assert onset_env.ndim == 1
        # Должно быть больше нуля (хоть какие-то колебания)
        assert len(onset_env) > 0

    def test_tempogram_shape(self, sine_audio):
        audio, sr = sine_audio
        tempo = FeatureExtractor.tempogram(audio, sr)
        assert tempo.ndim == 2
        # Проверяем, что размерность по времени больше 0
        assert tempo.shape[1] > 0

    def test_stft_empty_audio(self):
        with pytest.raises(ValueError):
            FeatureExtractor.stft(np.array([]))

    def test_chromagram_invalid_sr(self, sine_audio):
        audio, _ = sine_audio
        with pytest.raises(ValueError):
            FeatureExtractor.chromagram(audio, sr=0)

    def test_mfcc_invalid_sr(self, sine_audio):
        audio, _ = sine_audio
        with pytest.raises(ValueError):
            FeatureExtractor.mfcc(audio, sr=-1)