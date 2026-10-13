"""
Audio Extractor
===============
Extracts / converts audio from video files using ffmpeg (subprocess).
Returns path to a 16kHz mono WAV file ready for Whisper.
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


class AudioExtractor:
    """
    Converts any audio/video file to 16kHz mono WAV.
    Requires ffmpeg installed on the system.
    """

    def __init__(self, output_dir: str = "/tmp/lecture_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_ffmpeg()

    def extract(self, input_path: Union[str, Path]) -> Path:
        """
        Convert input file to 16kHz mono WAV.

        Args:
            input_path: Path to audio or video file.

        Returns:
            Path to the converted WAV file.
        """
        input_path = Path(input_path)
        suffix     = input_path.suffix.lower()

        output_path = self.output_dir / (input_path.stem + "_16k.wav")

        if output_path.exists():
            logger.info(f"Using cached audio: {output_path}")
            return output_path

        if suffix in AUDIO_EXTENSIONS:
            logger.info(f"Converting audio: {input_path.name} → 16kHz WAV")
        elif suffix in VIDEO_EXTENSIONS:
            logger.info(f"Extracting audio from video: {input_path.name}")
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}\n"
                f"Supported audio: {AUDIO_EXTENSIONS}\n"
                f"Supported video: {VIDEO_EXTENSIONS}"
            )

        self._run_ffmpeg(input_path, output_path)
        return output_path

    # ------------------------------------------------------------------

    def _run_ffmpeg(self, src: Path, dst: Path):
        cmd = [
            "ffmpeg", "-y",           # overwrite without asking
            "-i", str(src),           # input
            "-vn",                    # no video
            "-acodec", "pcm_s16le",   # PCM 16-bit
            "-ar", "16000",           # 16 kHz
            "-ac", "1",               # mono
            str(dst),
        ]
        logger.debug(f"ffmpeg cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
            )
        logger.info(f"Audio saved: {dst}")

    @staticmethod
    def _check_ffmpeg():
        if shutil.which("ffmpeg") is None:
            raise EnvironmentError(
                "ffmpeg not found. Install it:\n"
                "  Ubuntu/Debian: sudo apt install ffmpeg\n"
                "  macOS:         brew install ffmpeg\n"
                "  Windows:       https://ffmpeg.org/download.html"
            )
