"""
ASR Model — faster-whisper (int8 CPU) with fallback to openai-whisper

Speed optimizations applied:
  OPT-1  Larger chunks passed in from pipeline (120 s instead of 60 s)
  OPT-2  beam_size=1 / best_of=1  — greedy decode, ~2x faster, near-same accuracy
  OPT-3  Default model: large-v3-turbo (set in pipeline.py config)
  OPT-4  Parallel chunk transcription via ThreadPoolExecutor
"""
import logging
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

logger = logging.getLogger(__name__)


class WhisperASR:
    SUPPORTED_SIZES = ["tiny", "base", "small", "medium", "large-v3-turbo",
                       "large-v2", "large-v3"]

    def __init__(
        self,
        model_size:  str = "base",
        device:      str = "cpu",
        language:    str = "en",
        beam_size:   int = 1,       # OPT-2: greedy by default
        num_workers: int = 4,       # OPT-4: parallel chunk threads
    ):
        if model_size not in self.SUPPORTED_SIZES:
            raise ValueError(f"model_size must be one of {self.SUPPORTED_SIZES}")
        self.model_size  = model_size
        self.device      = device
        self.language    = language
        self.beam_size   = beam_size    # OPT-2
        self.num_workers = num_workers  # OPT-4
        self.model       = None
        self._backend    = None
        self._load_model()

    # ──────────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper '{self.model_size}' (int8)…")
            self.model    = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="int8",          # OPT-1: int8 quantization
                cpu_threads=self.num_workers, # let faster-whisper use all cores internally
                num_workers=1,                # we handle parallelism ourselves (OPT-4)
            )
            self._backend = "faster_whisper"
            logger.info("faster-whisper loaded.")
        except ImportError:
            logger.warning("faster-whisper not found; falling back to openai-whisper.")
            self._load_openai_whisper()

    def _load_openai_whisper(self):
        try:
            import whisper
            logger.info(f"Loading openai-whisper '{self.model_size}'…")
            self.model    = whisper.load_model(self.model_size)
            self._backend = "openai_whisper"
        except ImportError as e:
            raise ImportError("Install: pip install faster-whisper") from e

    # ──────────────────────────────────────────────────────────────────────────
    # OPT-4: Parallel chunk transcription
    # ──────────────────────────────────────────────────────────────────────────

    def transcribe_chunks(self, chunks: List[np.ndarray], sample_rate: int = 16000) -> str:
        """
        Transcribe a list of audio chunks in parallel using ThreadPoolExecutor.
        Results are re-ordered to match original chunk order before joining.
        Falls back to sequential processing for openai-whisper (not thread-safe).
        """
        if not chunks:
            return ""

        # openai-whisper's model is not thread-safe — run sequentially
        if self._backend != "faster_whisper":
            return self._transcribe_sequential(chunks, sample_rate)

        n        = len(chunks)
        workers  = min(self.num_workers, n)
        results  = [None] * n

        logger.info(f"  Transcribing {n} chunk(s) with {workers} parallel worker(s)…")

        def _job(idx: int, chunk: np.ndarray):
            logger.info(f"  Chunk {idx+1}/{n} ({len(chunk)/sample_rate:.1f}s) starting…")
            text = self._transcribe_chunk_faster(chunk)
            logger.info(f"  Chunk {idx+1}/{n} done.")
            return idx, text.strip()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_job, i, c): i for i, c in enumerate(chunks)}
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text

        joined = " ".join(t for t in results if t)
        return self._clean_transcript(joined)

    def _transcribe_sequential(self, chunks: List[np.ndarray], sample_rate: int) -> str:
        """Sequential fallback used for openai-whisper backend."""
        parts = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  Chunk {i+1}/{len(chunks)} ({len(chunk)/sample_rate:.1f}s)…")
            text = self._transcribe_single(chunk, sample_rate)
            if text.strip():
                parts.append(text.strip())
        return self._clean_transcript(" ".join(parts))

    # ──────────────────────────────────────────────────────────────────────────
    # Single-chunk transcription
    # ──────────────────────────────────────────────────────────────────────────

    def transcribe_file(self, audio_path: str) -> str:
        if self._backend == "faster_whisper":
            return self._transcribe_file_faster(audio_path)
        return self._transcribe_file_openai(audio_path)

    def _transcribe_single(self, chunk: np.ndarray, sample_rate: int) -> str:
        if self._backend == "faster_whisper":
            return self._transcribe_chunk_faster(chunk)
        return self._transcribe_chunk_openai(chunk, sample_rate)

    def _transcribe_chunk_faster(self, chunk: np.ndarray) -> str:
        # OPT-2: beam_size=1 (greedy) — ~2x faster, minimal accuracy loss for lectures
        segments, _ = self.model.transcribe(
            chunk,
            language=self.language,
            beam_size=self.beam_size,           # OPT-2
            best_of=1,                          # OPT-2
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            condition_on_previous_text=True,
        )
        # faster-whisper returns a generator — must consume it here (not outside the lock)
        return " ".join(seg.text for seg in segments)

    def _transcribe_file_faster(self, path: str) -> str:
        segments, _ = self.model.transcribe(
            path,
            language=self.language,
            beam_size=self.beam_size,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            condition_on_previous_text=True,
        )
        return self._clean_transcript(" ".join(seg.text for seg in segments))

    def _transcribe_chunk_openai(self, chunk: np.ndarray, sample_rate: int) -> str:
        result = self.model.transcribe(chunk, language=self.language, fp16=False, verbose=False)
        return result["text"]

    def _transcribe_file_openai(self, path: str) -> str:
        result = self.model.transcribe(path, language=self.language, fp16=False, verbose=False)
        return self._clean_transcript(result["text"])

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_transcript(text: str) -> str:
        text = re.sub(r"([.!?,])\1+", r"\1", text)   # remove repeated punctuation
        text = re.sub(r" +",          " ",   text)   # collapse whitespace
        text = re.sub(r"\[.*?\]",     "",    text)   # remove [noise], [music] tags
        text = re.sub(r"\(.*?\)",     "",    text)   # remove (inaudible) etc.
        return text.strip()

    @property
    def backend(self) -> str:
        return self._backend or "not_loaded"
