"""Split text into sentences using regex heuristics."""

import re

# Common abbreviations that end with a period but should not trigger a split.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "st",
        "gen",
        "gov",
        "sgt",
        "corp",
        "inc",
        "ltd",
        "co",
        "dept",
        "univ",
        "vol",
        "rev",
        "fig",
        "no",
        "op",
        "al",
        "approx",
        "avg",
        "est",
        "min",
        "max",
    }
)

# Matches single uppercase letter followed by a period (e.g. the letters in D.C.)
_INITIAL_RE = re.compile(r"^[A-Z]$")

# Matches multi-part abbreviations like "e.g", "i.e" (single letters separated by dots)
_MULTI_PART_ABBREV_RE = re.compile(r"^[a-zA-Z](\.[a-zA-Z])+$")


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences.

    Handles:
    - Period / question-mark / exclamation-mark as terminators.
    - Common abbreviations (Dr., Mr., etc.) — not treated as terminators.
    - Single-letter initials like U.S. or D.C. — not treated as terminators.
    - Newlines as sentence boundaries.

    Returns a list of stripped, non-empty sentence strings.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # First, split on newlines to respect paragraph / line boundaries.
    lines = re.split(r"\n+", text)

    sentences: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        sentences.extend(_split_line(line))

    return sentences


def _split_line(line: str) -> list[str]:
    """Split a single line (no newlines) into sentences."""
    # We walk through potential split points: positions right after [.!?]
    # followed by whitespace and an uppercase letter / digit / opening quote.
    #
    # At each candidate we check whether the period belongs to an abbreviation
    # or single-letter initial — if so we skip the split.

    result: list[str] = []
    start = 0
    i = 0
    length = len(line)

    while i < length:
        ch = line[i]

        if ch in ".!?":
            # Check if this is a sentence boundary.
            # Look ahead: need whitespace then uppercase/digit/quote
            j = i + 1

            # Skip any additional sentence-ending punctuation (e.g. "?!")
            while j < length and line[j] in ".!?":
                j += 1

            # Need at least one whitespace character after punctuation
            if j < length and line[j] in " \t":
                k = j
                while k < length and line[k] in " \t":
                    k += 1

                # Check what follows the whitespace
                if k < length and (
                    line[k].isupper() or line[k].isdigit() or line[k] in "\"'(["
                ):
                    # Candidate split point. Check if abbreviation.
                    if ch == "." and _is_abbreviation(line, i, start):
                        i = k
                        continue
                    # It's a real sentence boundary.
                    result.append(line[start:j].strip())
                    start = k
                    i = k
                    continue

        i += 1

    # Remaining text
    remaining = line[start:].strip()
    if remaining:
        result.append(remaining)

    return result


def _is_abbreviation(line: str, dot_pos: int, seg_start: int) -> bool:
    """Decide whether the period at *dot_pos* belongs to an abbreviation."""
    # Extract the word immediately before the dot.
    word_end = dot_pos
    p = dot_pos - 1
    while p >= seg_start and line[p] not in " \t":
        p -= 1
    word_start = p + 1
    word = line[word_start:word_end]

    # Strip any leading punctuation (quotes, parens)
    word = word.lstrip("\"'([")

    if not word:
        return False

    # Single uppercase letter (initials like D.C., U.S.)
    if _INITIAL_RE.match(word):
        return True

    # Check against known abbreviation list
    if word.lower().rstrip(".") in _ABBREVIATIONS:
        return True

    # Detect multi-part abbreviations like e.g., i.e., etc.
    # Pattern: sequences of single lowercase letters separated by periods (e.g. "e.g")
    if _MULTI_PART_ABBREV_RE.match(word):
        return True

    return False
