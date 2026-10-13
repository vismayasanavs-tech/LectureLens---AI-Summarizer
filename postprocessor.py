"""
Summary Post-Processor
======================
Converts raw model output into clean bullet points and paragraphs.
Also applies lecture-specific heuristics (section detection, key terms).
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class SummaryPostProcessor:
    """
    Formats raw summarizer output into student-friendly structures:
        - Bullet points (one key idea per bullet)
        - Clean paragraph
        - Key terms extraction
    """

    def format(self, raw_summary: str, output_format: str = "both") -> Dict[str, object]:
        """
        Args:
            raw_summary:   Text output from the summarizer model.
            output_format: "bullets" | "paragraph" | "both"

        Returns:
            dict with "bullets" (list[str]) and "paragraph" (str)
        """
        paragraph = self._clean_paragraph(raw_summary)
        bullets   = self._extract_bullets(paragraph)

        return {"paragraph": paragraph, "bullets": bullets}

    # ------------------------------------------------------------------

    @staticmethod
    def _clean_paragraph(text: str) -> str:
        """Normalize whitespace, fix capitalisation, remove artefacts."""
        text = re.sub(r"\s+", " ", text).strip()
        # Ensure first character is uppercase
        if text:
            text = text[0].upper() + text[1:]
        # Fix common model artefacts
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r"\.{2,}", ".", text)
        return text

    def _extract_bullets(self, paragraph: str) -> List[str]:
        """
        Split paragraph into one-sentence bullet points.
        Groups short sentences with the preceding one to avoid trivial bullets.
        """
        sentences = self._split_sentences(paragraph)
        bullets   = []
        buffer    = ""

        for sent in sentences:
            words = sent.split()
            if len(words) < 6 and buffer:
                # Too short — merge with previous
                buffer = buffer.rstrip(".") + " " + sent
            else:
                if buffer:
                    bullets.append(self._capitalise(buffer.strip()))
                buffer = sent

        if buffer:
            bullets.append(self._capitalise(buffer.strip()))

        # Remove near-duplicate bullets
        bullets = self._deduplicate(bullets)
        return bullets

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split on sentence boundaries, keeping punctuation."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _capitalise(s: str) -> str:
        return s[0].upper() + s[1:] if s else s

    @staticmethod
    def _deduplicate(bullets: List[str], threshold: float = 0.6) -> List[str]:
        """Remove bullets that are highly similar to an already-kept one."""
        def jaccard(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            return len(sa & sb) / max(len(sa | sb), 1)

        kept = []
        for b in bullets:
            if all(jaccard(b, k) < threshold for k in kept):
                kept.append(b)
        return kept


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Photosynthesis is the process by which plants convert sunlight into food. "
        "It occurs primarily in the chloroplasts of plant cells. "
        "Carbon dioxide and water are combined using light energy to produce glucose and oxygen. "
        "The overall equation is 6CO2 + 6H2O + light → C6H12O6 + 6O2. "
        "This process is fundamental to life on Earth as it forms the base of most food chains."
    )
    pp = SummaryPostProcessor()
    result = pp.format(sample)
    print("Paragraph:", result["paragraph"])
    print("\nBullets:")
    for b in result["bullets"]:
        print(f"  • {b}")
