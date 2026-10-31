"""
ECG Signal Processing Pipeline.
Includes signal validation, Butterworth bandpass filtering, notch filtering,
resampling, and per-lead / per-record normalization.
"""
from __future__ import annotations

from typing import Tuple, Optional, Union
import numpy as np
from scipy import signal


class ECGProcessor:
    def __init__(
        self,
        target_fs: float = 500.0,
        lowcut: float = 0.5,
        highcut: float = 45.0,
        filter_order: int = 4,
        notch_freq: Optional[float] = 50.0,
        notch_q: float = 30.0,
        nominal_samples: int = 2500,
        norm_method: str = "zscore"  # 'zscore' or 'robust'
    ):
        self.target_fs = target_fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.notch_freq = notch_freq
        self.notch_q = notch_q
        self.nominal_samples = nominal_samples
        self.norm_method = norm_method

    def validate(self, ecg: np.ndarray) -> np.ndarray:
        """
        Validate input ECG shape, dimensionality, and finite values.
        Expected shape: [leads, samples] (2D) or [batch, leads, samples] (3D).
        """
        arr = np.asarray(ecg, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]  # convert 1-lead [samples] to [1, samples]
        if arr.ndim not in (2, 3):
            raise ValueError(f"ECG array must be 2D [leads, samples] or 3D [batch, leads, samples], got ndim={arr.ndim}")
        if not np.isfinite(arr).all():
            raise ValueError("ECG contains NaN or infinite values")
        
        # Check samples count (last dimension)
        samples = arr.shape[-1]
        if samples < 100:
            raise ValueError(f"ECG sample count must be at least 100 samples, got {samples}")
            
        return arr

    def bandpass_filter(self, ecg: np.ndarray, fs: float) -> np.ndarray:
        """
        Apply zero-phase Butterworth bandpass filter (lowcut to highcut Hz).
        """
        nyq = 0.5 * fs
        low = max(0.01, self.lowcut / nyq)
        high = min(0.99, self.highcut / nyq)
        if low >= high:
            return ecg  # Bypass filter if Nyquist constraint violated
            
        sos = signal.butter(self.filter_order, [low, high], btype='bandpass', output='sos')
        # Apply along last dimension (samples)
        filtered = signal.sosfiltfilt(sos, ecg, axis=-1)
        return filtered.astype(np.float32)

    def notch_filter(self, ecg: np.ndarray, fs: float) -> np.ndarray:
        """
        Apply notch filter at powerline interference frequency (e.g. 50Hz or 60Hz).
        """
        if self.notch_freq is None or self.notch_freq >= (0.5 * fs):
            return ecg
        b, a = signal.iirnotch(w0=self.notch_freq, Q=self.notch_q, fs=fs)
        filtered = signal.filtfilt(b, a, ecg, axis=-1)
        return filtered.astype(np.float32)

    def resample(self, ecg: np.ndarray, orig_fs: float) -> Tuple[np.ndarray, float]:
        """
        Resample ECG to target_fs if different.
        """
        if abs(orig_fs - self.target_fs) < 1e-3:
            return ecg, orig_fs
        
        orig_samples = ecg.shape[-1]
        target_samples = int(round(orig_samples * (self.target_fs / orig_fs)))
        resampled = signal.resample(ecg, target_samples, axis=-1)
        return resampled.astype(np.float32), self.target_fs

    def normalize(self, ecg: np.ndarray) -> np.ndarray:
        """
        Per-lead normalization (z-score or robust median/IQR).
        """
        if self.norm_method == "robust":
            median = np.median(ecg, axis=-1, keepdims=True)
            q75, q25 = np.percentile(ecg, [75, 25], axis=-1, keepdims=True)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            return (ecg - median) / iqr
        else:
            mean = np.mean(ecg, axis=-1, keepdims=True)
            std = np.std(ecg, axis=-1, keepdims=True)
            std[std == 0] = 1.0
            return (ecg - mean) / std

    def crop_or_pad(self, ecg: np.ndarray, target_length: Optional[int] = None) -> np.ndarray:
        """
        Crop or pad to target length.
        """
        length = target_length or self.nominal_samples
        curr_len = ecg.shape[-1]
        if curr_len == length:
            return ecg
        elif curr_len > length:
            # Center crop
            start = (curr_len - length) // 2
            return ecg[..., start:start + length]
        else:
            # Center zero pad
            pad_total = length - curr_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            pad_width = [(0, 0)] * (ecg.ndim - 1) + [(pad_left, pad_right)]
            return np.pad(ecg, pad_width, mode='constant', constant_values=0)

    def process(self, ecg: Union[np.ndarray, list], orig_fs: float) -> np.ndarray:
        """
        Execute full ECG preprocessing pipeline.
        Returns: preprocessed array of shape [leads, nominal_samples] or [batch, leads, nominal_samples]
        """
        arr = self.validate(np.asarray(ecg, dtype=np.float32))
        filtered = self.bandpass_filter(arr, orig_fs)
        notched = self.notch_filter(filtered, orig_fs)
        resampled, fs = self.resample(notched, orig_fs)
        fixed_len = self.crop_or_pad(resampled, self.nominal_samples)
        normalized = self.normalize(fixed_len)
        return normalized.astype(np.float32)
