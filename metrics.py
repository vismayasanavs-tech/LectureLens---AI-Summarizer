"""
Evaluation Metrics
==================
WER  — Word Error Rate for ASR quality
ROUGE — ROUGE-1, ROUGE-2, ROUGE-L for summarization quality

Both implemented from scratch to avoid heavy NLP dependencies,
but will use jiwer/rouge_score if available for accuracy.
"""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


# ============================================================
# Word Error Rate (WER)
# ============================================================

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between reference and hypothesis transcripts.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=ref words.

    Returns value in [0, 1]. Lower is better. Target: ≤ 0.25.
    """
    try:
        import jiwer
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.SentencesToListOfWords(),
            jiwer.RemoveEmptyStrings(),
        ])
        return jiwer.wer(reference, hypothesis, truth_transform=transformation,
                         hypothesis_transform=transformation)
    except ImportError:
        logger.warning("jiwer not installed; using built-in WER implementation.")
        return _wer_builtin(reference, hypothesis)


def _wer_builtin(reference: str, hypothesis: str) -> float:
    """Pure-Python WER via dynamic-programming edit distance."""
    ref_words  = _normalise_text(reference).split()
    hyp_words  = _normalise_text(hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    r, h = len(ref_words), len(hyp_words)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): dp[i][0] = i
    for j in range(h + 1): dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                    dp[i][j-1],    # insertion
                                    dp[i-1][j-1])  # substitution

    return dp[r][h] / r


# ============================================================
# ROUGE Scores
# ============================================================

def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L (F1 scores).
    Returns dict with keys: rouge1, rouge2, rougeL.
    Higher is better.
    """
    try:
        from rouge_score import rouge_scorer
        scorer  = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores  = scorer.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        logger.warning("rouge_score not installed; using built-in ROUGE implementation.")
        return _rouge_builtin(reference, hypothesis)


def _rouge_builtin(reference: str, hypothesis: str) -> Dict[str, float]:
    """Pure-Python ROUGE-1, ROUGE-2, ROUGE-L."""
    ref_tokens = _normalise_text(reference).split()
    hyp_tokens = _normalise_text(hypothesis).split()

    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def f1(ref_ng, hyp_ng):
        ref_set = {}
        for g in ref_ng:
            ref_set[g] = ref_set.get(g, 0) + 1
        overlap = 0
        for g in hyp_ng:
            if ref_set.get(g, 0) > 0:
                overlap += 1
                ref_set[g] -= 1
        p = overlap / max(len(hyp_ng), 1)
        r = overlap / max(len(ref_ng), 1)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    r1 = f1(ngrams(ref_tokens, 1), ngrams(hyp_tokens, 1))
    r2 = f1(ngrams(ref_tokens, 2), ngrams(hyp_tokens, 2))
    rl = _lcs_f1(ref_tokens, hyp_tokens)

    return {
        "rouge1": round(r1, 4),
        "rouge2": round(r2, 4),
        "rougeL": round(rl, 4),
    }


def _lcs_f1(ref: list, hyp: list) -> float:
    """LCS-based ROUGE-L."""
    m, n = len(ref), len(hyp)
    if m == 0 or n == 0:
        return 0.0
    # LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1] + 1 if ref[i-1] == hyp[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / max(n, 1)
    r = lcs / max(m, 1)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _normalise_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
