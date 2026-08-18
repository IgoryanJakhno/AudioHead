"""
Модуль для работы с аудиофайлами: загрузка, предобработка и метаданные.
"""

import logging
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf  # опционально, для будущего экспорта

logger = logging.getLogger(__name__)


class AudioFile:
    """
    Представляет аудиофайл с сигналом и метаданными.

    Атрибуты:
        path (str): путь к исходному файлу.
        data (np.ndarray): одномерный (моно) или двумерный (стерео) массив сэмплов.
        sample_rate (int): частота дискретизации в Гц.
        channels (int): количество каналов (1 или 2).
        duration (float): длительность в секундах.
    """

    def __init__(self, path: str = None):
        """
        Инициализирует объект AudioFile.

        Args:
            path (str, optional): путь к аудиофайлу. Если указан, файл загружается сразу.
        """
        self.path = path
        self.data = None
        self.sample_rate = None
        self.channels = None
        self.duration = None

        if path is not None:
            self.load(path)

    def load(self, path: str = None, mono: bool = True, sr: int = None) -> "AudioFile":
        """
        Загружает аудиофайл с диска.

        Args:
            path (str, optional): путь к файлу. Если не указан, используется self.path.
            mono (bool): если True, преобразовать в моно; иначе сохранить стерео.
            sr (int, optional): целевая частота дискретизации. Если None, сохраняется исходная.

        Returns:
            AudioFile: сам объект для цепочки вызовов.

        Raises:
            FileNotFoundError: если файл не найден.
            ValueError: если путь не указан.
            Exception: при других ошибках загрузки.
        """
        if path is None:
            path = self.path
        if path is None:
            raise ValueError("Не указан путь к аудиофайлу")

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        try:
            # librosa.load поддерживает большинство форматов (wav, mp3, flac, ...)
            # При mono=True возвращает (n_samples,), при mono=False — (n_samples, n_channels)
            data, sr_loaded = librosa.load(path, sr=sr, mono=mono)
            self.data = data
            self.sample_rate = sr_loaded
            self.channels = 1 if mono else (data.shape[1] if data.ndim > 1 else 1)
            self.duration = librosa.get_duration(y=data, sr=sr_loaded)
            self.path = str(path_obj)

            logger.info(
                f"Загружен файл '{path}': длит. {self.duration:.2f} с, "
                f"частота {self.sample_rate} Гц, каналов: {self.channels}"
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки файла '{path}': {e}")
            raise

        return self

    def normalize(self, peak: float = 1.0) -> "AudioFile":
        """
        Нормализует сигнал по пиковому значению.

        Args:
            peak (float): желаемый пиковый уровень (по умолчанию 1.0).

        Returns:
            AudioFile: сам объект.
        """
        if self.data is None:
            raise ValueError("Аудиоданные не загружены")
        max_val = np.max(np.abs(self.data))
        if max_val > 0:
            self.data = self.data * (peak / max_val)
        logger.debug(f"Нормализация выполнена, пик = {peak}")
        return self

    def trim_silence(self, top_db: float = 60) -> "AudioFile":
        """
        Обрезает тишину в начале и конце аудио.

        Args:
            top_db (float): порог в децибелах для определения тишины.

        Returns:
            AudioFile: сам объект.
        """
        if self.data is None:
            raise ValueError("Аудиоданные не загружены")
        trimmed, _ = librosa.effects.trim(self.data, top_db=top_db)
        self.data = trimmed
        self.duration = librosa.get_duration(y=self.data, sr=self.sample_rate)
        logger.debug(f"Обрезка тишины выполнена, новая длит. {self.duration:.2f} с")
        return self

    def resample(self, new_sr: int) -> "AudioFile":
        """
        Передискретизирует сигнал на новую частоту.

        Args:
            new_sr (int): целевая частота дискретизации.

        Returns:
            AudioFile: сам объект.
        """
        if self.data is None:
            raise ValueError("Аудиоданные не загружены")
        if new_sr == self.sample_rate:
            return self
        self.data = librosa.resample(self.data, orig_sr=self.sample_rate, target_sr=new_sr)
        self.sample_rate = new_sr
        self.duration = librosa.get_duration(y=self.data, sr=self.sample_rate)
        logger.debug(f"Передискретизация на {new_sr} Гц выполнена")
        return self

    def get_waveform(self) -> np.ndarray:
        """
        Возвращает массив аудиоданных.

        Returns:
            np.ndarray: массив сэмплов.
        """
        if self.data is None:
            raise ValueError("Аудиоданные не загружены")
        return self.data

    def to_mono(self) -> "AudioFile":
        """
        Преобразует стереосигнал в моно путём усреднения каналов (если стерео).

        Returns:
            AudioFile: сам объект.
        """
        if self.data is None:
            raise ValueError("Аудиоданные не загружены")
        if self.channels == 1:
            return self
        if self.data.ndim == 2:
            self.data = np.mean(self.data, axis=1)
            self.channels = 1
            logger.debug("Преобразование в моно выполнено")
        return self