"""
Модуль для извлечения аудио-признаков: STFT, спектрограмма, хромаграмма, MFCC,
огибающая онсетов, темпограмма.
"""

import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Статический класс с методами для вычисления различных аудио-признаков.
    Все методы принимают одномерный массив аудиоданных (моно) и частоту дискретизации.
    """

    @staticmethod
    def stft(audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Вычисляет STFT (Short-Time Fourier Transform) аудиосигнала.

        Args:
            audio (np.ndarray): одномерный массив аудиоданных.
            n_fft (int): размер окна БПФ.
            hop_length (int): шаг между окнами.

        Returns:
            np.ndarray: комплексная STFT-матрица формы (n_fft//2 + 1, time_frames).
        """
        if audio is None or len(audio) == 0:
            raise ValueError("Аудиоданные пустые или не заданы")
        return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    @staticmethod
    def spectrogram(audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Вычисляет амплитудную спектрограмму в децибелах.

        Args:
            audio (np.ndarray): одномерный массив аудиоданных.
            n_fft (int): размер окна БПФ.
            hop_length (int): шаг между окнами.

        Returns:
            np.ndarray: спектрограмма в дБ, форма (n_fft//2 + 1, time_frames).
        """
        stft_matrix = FeatureExtractor.stft(audio, n_fft=n_fft, hop_length=hop_length)
        amplitude = np.abs(stft_matrix)
        # Преобразование в дБ
        return librosa.amplitude_to_db(amplitude, ref=np.max)

    @staticmethod
    def chromagram(audio: np.ndarray, sr: int, n_fft: int = 4096, hop_length: int = 2048) -> np.ndarray:
        """
        Вычисляет хромаграмму (12 тонов) для аудиосигнала.

        Args:
            audio (np.ndarray): одномерный массив аудиоданных.
            sr (int): частота дискретизации.
            n_fft (int): размер окна БПФ.
            hop_length (int): шаг между окнами.

        Returns:
            np.ndarray: хромаграмма формы (12, time_frames).
        """
        if sr is None or sr <= 0:
            raise ValueError("Неверная частота дискретизации")
        return librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)

    @staticmethod
    def mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13,
             n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Вычисляет MFCC (Mel-frequency cepstral coefficients).

        Args:
            audio (np.ndarray): одномерный массив аудиоданных.
            sr (int): частота дискретизации.
            n_mfcc (int): количество коэффициентов.
            n_fft (int): размер окна БПФ.
            hop_length (int): шаг между окнами.

        Returns:
            np.ndarray: MFCC матрица формы (n_mfcc, time_frames).
        """
        if sr is None or sr <= 0:
            raise ValueError("Неверная частота дискретизации")
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                    n_fft=n_fft, hop_length=hop_length)

    @staticmethod
    def onset_envelope(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
        """
        Вычисляет огибающую для обнаружения онсетов (onset strength).

        Args:
            audio (np.ndarray): одномерный массив аудиоданных.
            sr (int): частота дискретизации.
            hop_length (int): шаг между окнами.

        Returns:
            np.ndarray: массив огибающей длины time_frames.
        """
        if sr is None or sr <= 0:
            raise ValueError("Неверная частота дискретизации")
        return librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)

    @staticmethod
    def tempogram(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
        """
        Вычисляет темпограмму (периодичность) для анализа темпа.

        Args:
            audio (np.ndarray): одномерный массив аудиоданных.
            sr (int): частота дискретизации.
            hop_length (int): шаг между окнами.

        Returns:
            np.ndarray: темпограмма формы (темпы, time_frames).
        """
        if sr is None or sr <= 0:
            raise ValueError("Неверная частота дискретизации")
        # Используем Fourier tempogram для простоты
        return librosa.feature.tempogram(y=audio, sr=sr, hop_length=hop_length)