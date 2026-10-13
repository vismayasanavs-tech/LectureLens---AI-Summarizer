"""
Audio Preprocessor — denoise, normalise, chunk with 5s overlap
"""
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class AudioPreprocessor:

    def __init__(self, sample_rate: int = 16000, chunk_duration: int = 60,
                 overlap_duration: int = 5):
        self.sample_rate  = sample_rate
        self.chunk_size   = sample_rate * chunk_duration
        self.overlap_size = sample_rate * overlap_duration

    def process(self, audio_path: Path) -> Tuple[List[np.ndarray], float]:
        audio = self._load(audio_path)
        audio = self._denoise(audio)
        audio = self._normalize(audio)
        chunks = self._chunk(audio)
        duration = len(audio) / self.sample_rate
        logger.info(f"  Audio: {duration:.1f}s → {len(chunks)} chunks")
        return chunks, duration

    def _load(self, path: Path) -> np.ndarray:
        try:
            import soundfile as sf
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if sr != self.sample_rate:
                data = self._resample(data, sr, self.sample_rate)
            return data
        except ImportError:
            pass
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(str(path))
            data = data.astype(np.float32)
            if data.ndim == 2:
                data = data.mean(axis=1)
            data /= np.iinfo(np.int16).max
            if sr != self.sample_rate:
                data = self._resample(data, sr, self.sample_rate)
            return data
        except ImportError:
            raise ImportError("pip install soundfile")

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        ratio = target_sr / orig_sr
        n_out = int(len(audio) * ratio)
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, n_out)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    @staticmethod
    def _denoise(audio: np.ndarray, n_fft: int = 512) -> np.ndarray:
        """
        Frame-by-frame spectral subtraction that preserves the original length.
        Estimates noise from the first ~0.5 s of audio (assumed to be silence/background),
        then subtracts the noise floor from every frame with 50 % overlap-add.
        Falls back to returning audio unchanged on any error.
        """
        try:
            hop          = n_fft // 2
            noise_frames = max(1, 8000 // hop)          # ~0.5 s at 16 kHz

            # ── Noise profile from the first few frames ──
            profile = np.zeros(n_fft // 2 + 1, dtype=np.float32)
            count   = 0
            for k in range(noise_frames):
                start = k * hop
                end   = start + n_fft
                if end > len(audio):
                    break
                frame   = audio[start:end] * np.hanning(n_fft)
                profile += np.abs(np.fft.rfft(frame))
                count   += 1
            if count == 0:
                return audio
            profile /= count

            # ── Overlap-add spectral subtraction ──
            output = np.zeros(len(audio), dtype=np.float32)
            window = np.hanning(n_fft).astype(np.float32)

            for start in range(0, len(audio) - n_fft + 1, hop):
                frame     = audio[start : start + n_fft] * window
                spec      = np.fft.rfft(frame)
                mag       = np.abs(spec)
                phase     = np.angle(spec)
                mag_clean = np.maximum(mag - profile * 1.5, 0.0)
                frame_out = np.fft.irfft(mag_clean * np.exp(1j * phase)).astype(np.float32)
                output[start : start + n_fft] += frame_out * window

            # Normalise overlap-add gain
            norm = np.sum(window ** 2)
            if norm > 0:
                output *= hop / norm

            return np.clip(output, -1.0, 1.0)
        except Exception:
            return audio

    @staticmethod
    def _normalize(audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-8:
            audio = audio * (target_rms / rms)
        return np.clip(audio, -1.0, 1.0)

    def _chunk(self, audio: np.ndarray) -> List[np.ndarray]:
        if len(audio) <= self.chunk_size:
            return [audio]
        chunks, start = [], 0
        while start < len(audio):
            end = min(start + self.chunk_size, len(audio))
            chunks.append(audio[start:end])
            if end == len(audio):
                break
            start += self.chunk_size - self.overlap_size
        return chunks
