"""
Evaluation Pipeline for Speech Recognition Quality
Calculates Word Error Rate (WER) and Character Error Rate (CER)
with specialized normalization for English, Hindi, and Hinglish.
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional
import jiwer


def normalize_text_for_eval(text: str, preserve_devanagari: bool = True) -> str:
    """
    Standardize text for fair error-rate comparison:
    - Unicode NFC normalization
    - Lowercases Latin characters
    - Removes common punctuation and extra whitespace
    - Preserves Devanagari characters, matras, and virama
    """
    if not text:
        return ""

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Lowercase Latin letters
    text = text.lower()

    # Strip punctuation while preserving Devanagari (\u0900-\u097F) and alphanumeric
    # Keep letters, numbers, and Devanagari range
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_metrics(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Compute WER and CER between ground truth reference and model hypothesis.

    Returns:
        Dict with wer, cer, substitutions, deletions, insertions, hits.
    """
    norm_ref = normalize_text_for_eval(reference)
    norm_hyp = normalize_text_for_eval(hypothesis)

    if not norm_ref:
        return {
            "reference": reference,
            "hypothesis": hypothesis,
            "normalized_reference": norm_ref,
            "normalized_hypothesis": norm_hyp,
            "wer": 0.0 if not norm_hyp else 1.0,
            "cer": 0.0 if not norm_hyp else 1.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(norm_hyp.split()),
            "hits": 0,
        }

    # Word Error Rate via jiwer
    wer_output = jiwer.process_words(norm_ref, norm_hyp)
    # Character Error Rate via jiwer
    cer_output = jiwer.process_characters(norm_ref, norm_hyp)

    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "normalized_reference": norm_ref,
        "normalized_hypothesis": norm_hyp,
        "wer": round(float(wer_output.wer), 4),
        "cer": round(float(cer_output.cer), 4),
        "substitutions": wer_output.substitutions,
        "deletions": wer_output.deletions,
        "insertions": wer_output.insertions,
        "hits": wer_output.hits,
    }


def evaluate_batch(dataset: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Evaluate a batch of samples grouped by language category.
    Each sample: {'audio_path': ..., 'reference': ..., 'language': 'en'|'hi'|'hinglish'}
    """
    from backend.app.services.transcription.whisper_service import get_whisper_service

    service = get_whisper_service()
    results_by_lang: Dict[str, List[Dict[str, Any]]] = {}

    for item in dataset:
        lang = item.get("language", "unknown")
        ref = item["reference"]
        audio = item["audio_path"]

        # Run transcription
        transcription_res = service.transcribe(audio)
        hyp = transcription_res.transcript

        metrics = compute_metrics(ref, hyp)
        metrics["audio_path"] = audio
        metrics["detected_language"] = transcription_res.detected_language
        metrics["processing_time_sec"] = transcription_res.processing_time_sec

        if lang not in results_by_lang:
            results_by_lang[lang] = []
        results_by_lang[lang].append(metrics)

    # Compute aggregate stats per language
    summary = {}
    for lang, items in results_by_lang.items():
        avg_wer = sum(x["wer"] for x in items) / len(items) if items else 0.0
        avg_cer = sum(x["cer"] for x in items) / len(items) if items else 0.0
        summary[lang] = {
            "samples_count": len(items),
            "average_wer": round(avg_wer, 4),
            "average_cer": round(avg_cer, 4),
            "samples": items,
        }

    return summary


def format_report(summary: Dict[str, Any]) -> str:
    """Format evaluation results into a clean markdown table."""
    lines = [
        "# VocalSense Speech Recognition Evaluation Report",
        "",
        "| Language | Samples | Avg WER (Word Error Rate) | Avg CER (Character Error Rate) |",
        "|---|---|---|---|",
    ]
    for lang, stats in summary.items():
        wer_pct = f"{stats['average_wer'] * 100:.1f}%"
        cer_pct = f"{stats['average_cer'] * 100:.1f}%"
        lines.append(f"| {lang.capitalize()} | {stats['samples_count']} | {wer_pct} | {cer_pct} |")

    lines.append("")
    lines.append("### Detailed Breakdown")
    for lang, stats in summary.items():
        lines.append(f"\n#### {lang.upper()}")
        for idx, sample in enumerate(stats.get("samples", []), 1):
            lines.append(f"- **Sample {idx}**:")
            lines.append(f"  - Ref: `{sample['reference']}`")
            lines.append(f"  - Hyp: `{sample['hypothesis']}`")
            lines.append(f"  - WER: {sample['wer'] * 100:.1f}% | CER: {sample['cer'] * 100:.1f}% | Detected: `{sample.get('detected_language', 'N/A')}`")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate VocalSense Speech Recognition with WER and CER")
    parser.add_argument("--ref", type=str, help="Reference ground truth transcript")
    parser.add_argument("--hyp", type=str, help="Hypothesis predicted transcript")
    parser.add_argument("--audio", type=str, help="Optional audio file path to transcribe and evaluate")
    parser.add_argument("--lang", type=str, default=None, help="Language code override (en, hi)")

    args = parser.parse_args()

    if args.audio and args.ref:
        from backend.app.services.transcription.whisper_service import get_whisper_service
        service = get_whisper_service()
        res = service.transcribe(args.audio, language=args.lang)
        m = compute_metrics(args.ref, res.transcript)
        m["detected_language"] = res.detected_language
        print(json.dumps(m, indent=2, ensure_ascii=False))
    elif args.ref and args.hyp:
        m = compute_metrics(args.ref, args.hyp)
        print(json.dumps(m, indent=2, ensure_ascii=False))
    else:
        # Example demonstration of WER/CER for English, Hindi, Hinglish
        print("Running demonstration test cases across English, Hindi, and Hinglish...\n")
        demo_cases = [
            ("I want to complete my project presentation today.", "I want to complete my project presentation today.", "English Perfect"),
            ("I want to complete my project presentation today.", "I want to complete my project presentation to day.", "English Minor Spacing"),
            ("मुझे आज अपना प्रोजेक्ट पूरा करना है", "मुझे आज अपना प्रोजेक्ट पूरा करना है", "Hindi Devanagari"),
            ("Yaar mujhe kal apne project ka presentation complete karna hai", "Yaar mujhe kal apne project ka presentation complete karna hai", "Hinglish Romanized"),
            ("Yaar mujhe kal presentation complete karna hai", "Yar mujhe kal presentation complete karni hai", "Hinglish Dialect Variations"),
        ]
        for ref, hyp, label in demo_cases:
            res = compute_metrics(ref, hyp)
            print(f"[{label}]")
            print(f"  Ref: {ref}")
            print(f"  Hyp: {hyp}")
            print(f"  WER: {res['wer'] * 100:.1f}% | CER: {res['cer'] * 100:.1f}%\n")
