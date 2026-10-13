"""
Lecture Summarization Pipeline  — optimized for speed + accuracy
"""
import os, time, logging
from pathlib import Path
from typing import Optional, Dict, Any

from utils.link_downloader import LinkDownloader
from utils.audio_extractor import AudioExtractor
from utils.preprocessor    import AudioPreprocessor
from models.asr_model      import WhisperASR
from models.summarizer     import LectureSummarizer
from utils.postprocessor   import SummaryPostProcessor
from utils.metrics         import compute_wer, compute_rouge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_URL_PREFIXES = ("http://", "https://", "www.", "youtu", "open.spotify")


class LectureSummarizationPipeline:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self._default_config(), **(config or {})}
        logger.info("Initialising pipeline…")
        self.downloader   = LinkDownloader()
        self.extractor    = AudioExtractor()
        self.preprocessor = AudioPreprocessor(
            sample_rate=self.config["sample_rate"],
            chunk_duration=self.config["chunk_duration"],
        )
        self.asr = WhisperASR(
            model_size=self.config["whisper_model"],
            device=self.config["device"],
            language=self.config["language"],
            beam_size=self.config["asr_beam_size"],
            num_workers=self.config["asr_workers"],
        )
        self.summarizer = LectureSummarizer(
            model_name=self.config["summarizer_model"],
            device=self.config["device"],
            max_input_tokens=self.config["max_input_tokens"],
            max_summary_length=self.config["max_summary_length"],
            min_summary_length=self.config["min_summary_length"],
        )
        self.postprocessor = SummaryPostProcessor()
        logger.info("Pipeline ready.")

    def process(self, source: str, output_format: str = "both") -> Dict[str, Any]:
        t0 = time.time()
        results: Dict[str, Any] = {"source": source, "metadata": {}}
        is_url = any(source.startswith(p) for p in _URL_PREFIXES)

        # Stage 1: Get audio
        if is_url:
            logger.info("Stage 1/5 — Downloading from URL…")
            audio_path, link_meta = self.downloader.download(source)
            results["metadata"].update(link_meta)
        else:
            logger.info("Stage 1/5 — Extracting audio…")
            audio_path = self.extractor.extract(Path(source))
            results["metadata"]["source_type"] = "local_file"

        results["metadata"]["audio_path"] = str(audio_path)

        # Stage 2: Preprocess
        logger.info("Stage 2/5 — Preprocessing…")
        chunks, duration = self.preprocessor.process(audio_path)
        results["metadata"]["audio_duration_sec"] = round(duration, 1)
        results["metadata"]["num_chunks"]         = len(chunks)

        # Stage 3: ASR
        logger.info("Stage 3/5 — Transcribing…")
        t_asr = time.time()
        transcript = self.asr.transcribe_chunks(chunks)
        results["transcript"]               = transcript
        results["metadata"]["asr_time_sec"] = round(time.time() - t_asr, 2)
        results["metadata"]["word_count"]   = len(transcript.split())

        # Stage 4: Summarise
        logger.info("Stage 4/5 — Summarising…")
        t_sum = time.time()
        raw_summary = self.summarizer.summarize(transcript)
        results["metadata"]["summarization_time_sec"] = round(time.time() - t_sum, 2)

        # Stage 5: Format
        logger.info("Stage 5/5 — Formatting…")
        processed = self.postprocessor.format(raw_summary, output_format)
        results["summary"]       = processed["paragraph"]
        results["bullet_points"] = processed["bullets"]

        total = time.time() - t0
        results["metadata"]["total_time_sec"]    = round(total, 2)
        results["metadata"]["speed_ratio"]       = round(total / max(duration, 1), 2)
        results["metadata"]["compression_ratio"] = round(
            len(transcript.split()) / max(len(raw_summary.split()), 1), 1)

        logger.info(f"Done in {total:.1f}s")
        return results

    def evaluate(self, source: str, ref_transcript: str = "",
                 ref_summary: str = "") -> Dict[str, float]:
        results = self.process(source)
        metrics: Dict[str, float] = {}
        if ref_transcript:
            metrics["wer"] = compute_wer(ref_transcript, results["transcript"])
        if ref_summary:
            metrics.update(compute_rouge(ref_summary, results["summary"]))
        return metrics

    @staticmethod
    def _default_config():
        # Override the Whisper model without editing code:
        #   WHISPER_MODEL=small python app.py
        #
        # CPU model size guide:
        #   "base"           — 140 MB, ~1 min/hr audio, good accuracy   <- DEFAULT
        #   "small"          — 460 MB, ~2 min/hr audio, better accuracy
        #   "large-v3-turbo" — 800 MB, ~5 min/hr audio, best accuracy (needs >4 GB RAM)
        whisper_model = os.environ.get("WHISPER_MODEL", "base")
        return dict(
            whisper_model      = whisper_model,
            summarizer_model   = "sshleifer/distilbart-cnn-12-6",
            device             = "cpu",
            language           = "en",
            sample_rate        = 16000,
            chunk_duration     = 120,      # OPT-1: 120s chunks, half as many iterations
            max_input_tokens   = 512,
            max_summary_length = 200,
            min_summary_length = 60,
            asr_beam_size      = 1,        # OPT-2: greedy decode, ~2x faster
            asr_workers        = 4,        # OPT-4: parallel chunk threads
        )


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--format", default="both", choices=["bullets","paragraph","both"])
    parser.add_argument("--model", default="base",
                        choices=["tiny","base","small","medium","large-v3-turbo"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pipeline = LectureSummarizationPipeline({"whisper_model": args.model})
    results  = pipeline.process(args.source, output_format=args.format)
    print("\nTRANSCRIPT:", results["transcript"][:400], "…")
    print("\nSUMMARY:", results["summary"])
    print("\nBULLETS:")
    for b in results["bullet_points"]: print(f"  • {b}")
    if args.output:
        with open(args.output, "w") as f: json.dump(results, f, indent=2)
