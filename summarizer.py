"""
Lecture Summarizer — DistilBART (fast) with hierarchical chunking for long transcripts
"""
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class LectureSummarizer:
    """
    Fast abstractive summarizer.
    Default: sshleifer/distilbart-cnn-12-6  (~600 MB, 3x faster than bart-large)
    Better:  philschmid/bart-large-cnn-samsum (dialogue-trained, more accurate)
    Best:    facebook/bart-large-cnn          (highest quality, slowest)
    """

    def __init__(
        self,
        model_name: str         = "sshleifer/distilbart-cnn-12-6",
        device: str             = "cpu",
        max_input_tokens: int   = 512,
        max_summary_length: int = 200,
        min_summary_length: int = 60,
    ):
        self.model_name         = model_name
        self.device             = device
        self.max_input_tokens   = max_input_tokens
        self.max_summary_length = max_summary_length
        self.min_summary_length = min_summary_length
        self.tokenizer          = None
        self.model              = None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading summarizer: {self.model_name}…")
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model     = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("Summarizer loaded.")
        except ImportError as e:
            raise ImportError("pip install transformers torch") from e

    def summarize(self, text: str) -> str:
        text = self._preprocess_text(text)
        if not text.strip():
            return "No content to summarize."
        token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
        logger.info(f"  Input: {token_count} tokens")
        if token_count <= self.max_input_tokens:
            return self._summarize_chunk(text)
        return self._hierarchical_summarize(text)

    def _hierarchical_summarize(self, text: str, _depth: int = 0) -> str:
        MAX_DEPTH = 5
        if _depth >= MAX_DEPTH:
            # Safety valve: just truncate and summarize what we have
            logger.warning(f"Hierarchical summarization hit depth limit ({MAX_DEPTH}); truncating.")
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            truncated = self.tokenizer.decode(tokens[:self.max_input_tokens], skip_special_tokens=True)
            return self._summarize_chunk(truncated)

        segments = self._split_into_segments(text)
        logger.info(f"  Hierarchical depth={_depth}: {len(segments)} segments")
        summaries = []
        for i, seg in enumerate(segments):
            logger.info(f"  Segment {i+1}/{len(segments)}…")
            s = self._summarize_chunk(seg, max_length=120, min_length=30)
            summaries.append(s)
        combined = " ".join(summaries)
        combined_tokens = len(self.tokenizer.encode(combined, add_special_tokens=False))
        if combined_tokens <= self.max_input_tokens:
            return self._summarize_chunk(combined,
                                          max_length=self.max_summary_length,
                                          min_length=self.min_summary_length)
        return self._hierarchical_summarize(combined, _depth=_depth + 1)

    def _split_into_segments(self, text: str) -> List[str]:
        sentences = self._sentence_tokenize(text)
        segments, current, current_tokens = [], [], 0
        for sent in sentences:
            t = len(self.tokenizer.encode(sent, add_special_tokens=False))
            if current_tokens + t > self.max_input_tokens and current:
                segments.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(sent)
            current_tokens += t
        if current:
            segments.append(" ".join(current))
        return segments

    def _summarize_chunk(self, text: str,
                          max_length: Optional[int] = None,
                          min_length: Optional[int] = None) -> str:
        import torch
        max_length = max_length or self.max_summary_length
        min_length = min_length or self.min_summary_length
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                min_length=min_length,
                num_beams=5,
                length_penalty=1.5,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    @staticmethod
    def _preprocess_text(text: str) -> str:
        # Only remove pure filler sounds, keep connecting words
        fillers = r"\b(um|uh|er|ah)\b"
        text = re.sub(fillers, "", text, flags=re.IGNORECASE)
        text = re.sub(r" +", " ", text).strip()
        return text

    @staticmethod
    def _sentence_tokenize(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
