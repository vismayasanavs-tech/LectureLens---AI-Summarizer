"""
Link Downloader
===============
Downloads audio from:
  • YouTube  — any youtube.com / youtu.be URL       → via yt-dlp
  • Spotify  — track / episode / podcast URL        → via spotdl
  • Direct   — .mp3/.wav/.mp4 HTTP URLs             → via requests

All outputs are saved as 16 kHz mono WAV for direct Whisper ingestion.

Install dependencies:
    pip install yt-dlp spotdl requests
    sudo apt install ffmpeg          # required by both yt-dlp and spotdl
"""

import os
import re
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Platform regexes ──────────────────────────────────────────────────────────
_YT_RE      = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/(watch|shorts|playlist|live)|youtu\.be/)", re.I
)
_SPOTIFY_RE = re.compile(
    r"(https?://)?open\.spotify\.com/(track|episode|show|album|playlist)/", re.I
)


class LinkDownloader:
    """
    Accepts a URL string → downloads audio → returns (wav_path, metadata_dict).
    """

    def __init__(self, output_dir: str = "/tmp/lecture_downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────────────

    def download(self, url: str) -> Tuple[Path, dict]:
        """
        Returns:
            (path_to_wav, metadata)
            metadata keys: title, duration_sec, source, url, thumbnail
        """
        url = url.strip()
        if _YT_RE.search(url):
            return self._download_youtube(url)
        elif _SPOTIFY_RE.search(url):
            return self._download_spotify(url)
        elif url.lower().startswith("http"):
            return self._download_direct(url)
        else:
            raise ValueError(
                f"Unsupported URL: {url}\n"
                "Supported: YouTube (youtube.com / youtu.be) | "
                "Spotify (open.spotify.com) | Direct HTTP audio/video URL"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # YouTube via yt-dlp
    # ─────────────────────────────────────────────────────────────────────────

    def _download_youtube(self, url: str) -> Tuple[Path, dict]:
        self._check_tool("yt-dlp", "pip install yt-dlp")
        logger.info(f"[YouTube] Fetching metadata: {url}")

        # ── Fetch metadata first ──
        meta_cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-playlist",
            url,
        ]
        meta_result = subprocess.run(meta_cmd, capture_output=True, text=True)
        metadata = {}
        if meta_result.returncode == 0:
            import json
            raw = json.loads(meta_result.stdout)
            metadata = {
                "title":        raw.get("title", "Unknown"),
                "duration_sec": int(raw.get("duration", 0)),
                "uploader":     raw.get("uploader", ""),
                "thumbnail":    raw.get("thumbnail", ""),
                "source":       "youtube",
                "url":          url,
            }
            logger.info(f"  Title: {metadata['title']} | Duration: {metadata['duration_sec']}s")

        # ── Download best audio, convert to 16kHz mono WAV ──
        safe_title = re.sub(r"[^\w\-]", "_", metadata.get("title", "lecture"))[:60]
        out_path   = self.output_dir / f"{safe_title}_yt.wav"

        if out_path.exists():
            logger.info(f"  Using cached file: {out_path}")
            return out_path, metadata

        dl_cmd = [
            "yt-dlp",
            "--no-playlist",
            "-x",                                    # extract audio only
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args",
            "ffmpeg:-ar 16000 -ac 1",                # 16 kHz mono
            "-o", str(out_path.with_suffix("")),     # yt-dlp appends .wav
            url,
        ]
        logger.info("  Downloading audio from YouTube...")
        result = subprocess.run(dl_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"yt-dlp download failed (exit {result.returncode}):\n{result.stderr}\n\n"
                "Make sure yt-dlp is up-to-date:  pip install -U yt-dlp"
            )

        # yt-dlp may name the file with or without the .wav suffix we passed
        out_path = self._find_output(out_path.with_suffix(""), ".wav")

        if not out_path:
            raise RuntimeError(
                "yt-dlp finished successfully but the output .wav file could not be located.\n"
                f"Check the download directory: {self.output_dir}\n"
                "This can happen with some yt-dlp versions — try: pip install -U yt-dlp"
            )

        logger.info(f"  Saved: {out_path}")
        return out_path, metadata

    # ─────────────────────────────────────────────────────────────────────────
    # Spotify via spotdl
    # ─────────────────────────────────────────────────────────────────────────

    def _download_spotify(self, url: str) -> Tuple[Path, dict]:
        """
        spotdl downloads from Spotify by matching on YouTube Music.
        Works for tracks, podcast episodes, and playlists.

        Note: Spotify podcast episodes can also be downloaded as audio streams.
        """
        self._check_tool("spotdl", "pip install spotdl")
        logger.info(f"[Spotify] Downloading: {url}")

        work_dir = self.output_dir / "spotify"
        work_dir.mkdir(exist_ok=True)

        cmd = [
            "spotdl",
            "download",
            url,
            "--output",    str(work_dir / "{title}"),
            "--format",    "wav",
            "--bitrate",   "disable",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work_dir))

        # Find the downloaded WAV
        wav_files = sorted(work_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
        if not wav_files:
            # spotdl may produce mp3 — convert if so
            mp3_files = sorted(work_dir.glob("*.mp3"), key=os.path.getmtime, reverse=True)
            if mp3_files:
                wav_path = self._convert_to_wav(mp3_files[0])
                wav_files = [wav_path]

        if not wav_files:
            raise RuntimeError(
                f"spotdl failed or produced no output.\nstderr:\n{result.stderr}\n\n"
                "Troubleshooting:\n"
                "  1. pip install -U spotdl\n"
                "  2. Spotify podcasts may require the --cookie-file flag\n"
                "  3. Some tracks may be region-locked"
            )

        wav_path = wav_files[0]
        # Resample to 16kHz if needed
        wav_path = self._ensure_16k_mono(wav_path)

        metadata = {
            "title":     wav_path.stem,
            "source":    "spotify",
            "url":       url,
            "thumbnail": "",
        }
        logger.info(f"  Saved: {wav_path}")
        return wav_path, metadata

    # ─────────────────────────────────────────────────────────────────────────
    # Direct HTTP URL (mp3, wav, mp4, etc.)
    # ─────────────────────────────────────────────────────────────────────────

    def _download_direct(self, url: str) -> Tuple[Path, dict]:
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")

        parsed = urlparse(url)
        fname  = Path(parsed.path).name or "audio_download"
        out_raw = self.output_dir / fname

        logger.info(f"[Direct] Downloading: {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(out_raw, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        logger.debug(f"  {pct:.0f}%")

        # Convert to 16kHz WAV
        wav_path = self._convert_to_wav(out_raw)
        metadata = {
            "title":     out_raw.stem,
            "source":    "direct",
            "url":       url,
            "thumbnail": "",
        }
        return wav_path, metadata

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _convert_to_wav(self, src: Path) -> Path:
        """Convert any audio/video file to 16kHz mono WAV via ffmpeg."""
        dst = src.with_suffix(".wav")
        if dst == src and self._is_16k_mono(src):
            return src
        cmd = ["ffmpeg", "-y", "-i", str(src),
               "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(dst)]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {r.stderr.decode()}")
        return dst

    def _ensure_16k_mono(self, src: Path) -> Path:
        """Re-encode to 16kHz mono only if necessary."""
        return self._convert_to_wav(src)

    @staticmethod
    def _is_16k_mono(path: Path) -> bool:
        """Quick check via ffprobe."""
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", str(path)],
                capture_output=True, text=True
            )
            import json
            info = json.loads(r.stdout)
            stream = next((s for s in info.get("streams", [])
                           if s.get("codec_type") == "audio"), None)
            if stream:
                return (int(stream.get("sample_rate", 0)) == 16000 and
                        int(stream.get("channels", 0)) == 1)
        except Exception:
            pass
        return False

    def _find_output(self, base: Path, ext: str) -> Optional[Path]:
        """Handle yt-dlp's various output naming patterns, including double-extension .wav.wav."""
        candidates = [
            base.with_suffix(ext),              # stem.wav
            Path(str(base) + ext),              # stem (no suffix) + .wav
            Path(str(base) + ext + ext),        # stem.wav.wav  (yt-dlp double-ext bug)
            *base.parent.glob(base.name + "*" + ext),
        ]
        for candidate in candidates:
            p = Path(candidate)
            if p.exists():
                # Rename .wav.wav → .wav automatically
                if p.suffix == ext and str(p).endswith(ext + ext):
                    clean = Path(str(p)[: -len(ext)])
                    p.rename(clean)
                    return clean
                return p
        return None

    @staticmethod
    def _check_tool(tool: str, install_cmd: str):
        import shutil
        if shutil.which(tool) is None:
            raise EnvironmentError(
                f"'{tool}' not found. Install it:\n  {install_cmd}"
            )
