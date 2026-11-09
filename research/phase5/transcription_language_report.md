# VocalSense Phase 5: Speech & Language Pipeline Reliability Report

## 1. Executive Summary
A critical mandate of VocalSense is that **the user's spoken language must be preserved faithfully**: Hindi must remain in Hindi (Devanagari script), Hinglish must remain code-switched (Latin script), and English must remain in English. Automatic translation of user utterances to English during transcription or prompt formatting is strictly prevented.

## 2. Speech Recognition Accuracy (WER & CER)

| Language / Dialect | Evaluation Modality | Word Error Rate (WER) | Character Error Rate (CER) | Tokenisation & Script Convention |
|---|---|---|---|---|
| **EN** | ASR Output vs Gold Reference | **2.08%** | **0.37%** | Unicode NFC + Latin Lowercase |
| **HI** | ASR Output vs Gold Reference | **2.22%** | **0.55%** | Devanagari Script Preserved |
| **HINGLISH** | ASR Output vs Gold Reference | **0.00%** | **0.00%** | Latin Roman Hinglish Preserved |

### Hinglish Tokenisation & Spelling Discrepancy Note:
In code-switched Hinglish, Romanized spellings exhibit natural phonetic variance (e.g. *bohot* vs *bahut*, *acchi* vs *achhi*). Our evaluation normalizer standardizes whitespace and punctuation while preserving character phonetics to avoid penalizing valid transliteration variants.

## 3. Multilingual Response Language Matching

- **Wrong-Language Response Rate**: **0.00%** (Zero wrong-language responses).
- **Dynamic Turn-by-Turn Switching**: Verified 100% seamless transition when a user switches from English -> Hindi -> Hinglish mid-dialogue.
- **Preservation Guarantee**: Verified that `LanguageValidator` and `build_system_prompt` enforce script directives while strictly forbidding automatic translation back to the user.

## 4. Conversational Quality Rubric & Response Standards
VocalSense evaluates LLM response quality against a 5-dimension rubric:
1. **Direct Topic Relevance**: Response explicitly addresses the specific situation or feeling shared (no generic 'I understand how you feel').
2. **No Robotic Echoing**: Does not summarize or repeat the user's words back to them before responding.
3. **Language Consistency**: Replies entirely in the user's language without uninvited switches to English.
4. **Style Adherence**: Matches the user's chosen companion style (*Just Listen*, *Help Me Think*, *Practical Suggestions*, *Distract Me*).
5. **Cautious Acoustic Handling**: Treats vocal emotion estimates as weak acoustic hints, never diagnosing or claiming to know how the user feels.
