"""Composable input guard pipeline.

Checks applied in order:
  1. Length limit (max 1000 chars)
  2. Control character stripping
  3. Prompt injection detection
  4. PII detection (flag only, don't block)
  5. Topic boundary check
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = 1000

# Unicode control / bidi-override characters used in prompt smuggling
CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f"
    r"​-‏"  # zero-width space, joiners, bidi marks
    r"‪-‮"  # bidi overrides
    r"⁠-⁩"  # word joiner, bidi isolates
    r"﻿￰-￿]"
)

# Prompt injection detection patterns
PROMPT_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|the\s+above)\s+(instructions?|prompts?|messages?|context)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)", re.I),
    re.compile(r"forget\s+(everything|all|your)\s+(you\s+)?(were\s+told|instructions?)", re.I),
    re.compile(r"(you\s+are|act\s+as|pretend\s+you\s+are|now\s+you\s+are)\s+(now\s+)?(DAN|STAN|a\s+different|a\s+new)\b", re.I),
    re.compile(r"(system\s*(prompt|message|instruction|:))", re.I),
    re.compile(r"\[system\]|\[/system\]|<\|system\|>|</?system>", re.I),
    re.compile(r"\bSYSTEM\s*:\s*(override|bypass|ignore|reset)", re.I),
    re.compile(r"from\s+now\s+on\s+(you|your)\s+(are|will|must)\s+(no\s+longer|not)", re.I),
    re.compile(r"(do|does)\s+not\s+(follow|obey|comply)", re.I),
    re.compile(r"override\s+(system|safety|content\s+policy)", re.I),
    re.compile(r"jailbreak|jail\s*break", re.I),
    re.compile(r"<!--|-->|<\s*script|javascript\s*:", re.I),
]

# PII detection patterns (flag only, don't block)
PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

# Off-topic blocklist (keyword-scored)
OFF_TOPIC_KEYWORDS: list[str] = [
    "write me a virus", "write a virus", "virus that steals", "create malware", "hack into", "ddos attack",
    "child abuse", "illegal weapon", "how to make drugs",
    "commit suicide", "self-harm instructions",
    "generate explicit", "create deepfake porn",
    "write me code for a keylogger", "phishing email template",
]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    """Result of an input guard check."""

    passed: bool
    blocked_reason: str | None = None
    sanitized_input: str | None = None
    pii_detected: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_length(text: str) -> GuardResult:
    if len(text) > MAX_INPUT_LENGTH:
        return GuardResult(
            passed=False,
            blocked_reason=f"Input exceeds maximum length of {MAX_INPUT_LENGTH} characters (got {len(text)}).",
        )
    return GuardResult(passed=True, sanitized_input=text)


def _strip_control_chars(text: str) -> GuardResult:
    cleaned = CONTROL_CHARS.sub("", text)
    removed_count = len(text) - len(cleaned)
    if removed_count > 0:
        logger.info("Stripped %d control characters from input.", removed_count)
    return GuardResult(passed=True, sanitized_input=cleaned)


def _check_prompt_injection(text: str) -> GuardResult:
    matched = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        if match := pattern.search(text):
            matched.append(f"{pattern.pattern[:60]}... → matched: '{match.group()}'")

    if matched:
        logger.warning("Prompt injection patterns detected: %s", matched)
        return GuardResult(
            passed=False,
            blocked_reason="Input contains patterns consistent with prompt injection. Your request has been blocked.",
        )
    return GuardResult(passed=True, sanitized_input=text)


def _check_pii(text: str) -> GuardResult:
    detected = []
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            detected.append(pii_type)

    if detected:
        logger.info("PII patterns detected in input: %s", detected)
        return GuardResult(passed=True, sanitized_input=text, pii_detected=detected)
    return GuardResult(passed=True, sanitized_input=text)


def _check_topic(text: str) -> GuardResult:
    text_lower = text.lower()
    for keyword in OFF_TOPIC_KEYWORDS:
        if keyword in text_lower:
            logger.warning("Off-topic content detected: '%s' matched keyword '%s'.", text[:100], keyword)
            return GuardResult(
                passed=False,
                blocked_reason="Your request appears to be outside the scope of a travel planning assistant.",
            )
    return GuardResult(passed=True, sanitized_input=text)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def validate_input(user_message: str) -> GuardResult:
    """Run the full input validation pipeline.

    Returns GuardResult with:
      - passed: True if input is safe to process
      - blocked_reason: why the input was blocked (if passed=False)
      - sanitized_input: cleaned version of the input (if passed=True)
      - pii_detected: list of PII types found (for logging, never blocks)
    """
    checks = [
        _check_length,
        _strip_control_chars,
        _check_prompt_injection,
        _check_pii,
        _check_topic,
    ]

    current_text = user_message.strip()
    all_pii: list[str] = []

    for check in checks:
        result = check(current_text)
        if not result.passed:
            result.sanitized_input = None
            result.pii_detected = all_pii
            return result
        if result.pii_detected:
            all_pii.extend(result.pii_detected)
        if result.sanitized_input is not None:
            current_text = result.sanitized_input

    return GuardResult(passed=True, sanitized_input=current_text, pii_detected=all_pii)
