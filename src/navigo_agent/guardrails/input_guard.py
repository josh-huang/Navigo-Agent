"""Input guard: sanitise and validate user input before graph invocation.

Composable check pipeline — each check returns whether the input is safe.
Checks are lightweight (regex + keyword heuristics), no LLM calls.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 1000

# ── Guard Result ───────────────────────────────────────────────────────

@dataclass
class GuardResult:
    """Result of input guard checks."""
    passed: bool
    blocked_reason: str | None = None
    sanitized_input: str = ""
    pii_detected: list[str] = field(default_factory=list)


# ── Check 1: Length ────────────────────────────────────────────────────

def _check_length(user_input: str) -> str | None:
    """Reject inputs over MAX_INPUT_LENGTH — prevents context-window DoS."""
    if len(user_input) > MAX_INPUT_LENGTH:
        return f"Input too long ({len(user_input)} chars, max {MAX_INPUT_LENGTH})."
    return None


# ── Check 2: Control Characters ────────────────────────────────────────

# Null bytes, Unicode bidi-override / direction-control characters
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f‎‏‪-‮⁦-⁩]")

def _sanitize_control_chars(user_input: str) -> str:
    """Strip null bytes and Unicode direction-control characters."""
    return CONTROL_CHAR_PATTERN.sub("", user_input)


# ── Check 3: Prompt Injection Detection ────────────────────────────────

# Known jailbreak / prompt-injection patterns — lightweight regex + keyword
INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Direct override attempts
    ("system_override", re.compile(
        r"(ignore|forget|disregard)\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)",
        re.I,
    )),
    ("role_override", re.compile(
        r"(you\s+are\s+now|act\s+as\s+(a|an)|pretend\s+you\s+are|you\s+must\s+(obey|follow))",
        re.I,
    )),
    ("dan_jailbreak", re.compile(
        r"\bDAN\b.*\b(do\s+anything\s+now|jailbreak)\b", re.I,
    )),
    ("delimiter_attack", re.compile(
        r"<\/?system>|<\/?instruction>|\[system\]|\[/system\]|<\|im_start\|>|<\|im_end\|>",
        re.I,
    )),
    ("prompt_leak", re.compile(
        r"(reveal|show|print|display|tell\s+me)\s+(your\s+)?(system\s+)?(prompt|instructions?|rules?)",
        re.I,
    )),
    ("output_format_hijack", re.compile(
        r"(respond\s+only\s+with|output\s+in\s+JSON|format\s+your\s+response)",
        re.I,
    )),
]

def _detect_injection(user_input: str) -> str | None:
    """Scan for known prompt-injection patterns. Returns reason or None."""
    for name, pattern in INJECTION_PATTERNS:
        if pattern.search(user_input):
            logger.warning("Input guard: detected injection pattern '%s' in: %.100s", name, user_input)
            return f"Input blocked: potential prompt injection detected ({name})."
    return None


# ── Check 4: PII Detection ─────────────────────────────────────────────

PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("phone", re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
]

def _detect_pii(user_input: str) -> list[str]:
    """Detect PII in input — log-only, does NOT block."""
    found: list[str] = []
    for name, pattern in PII_PATTERNS:
        if pattern.search(user_input):
            found.append(name)
    if found:
        logger.info("Input guard: PII detected (%s) — not blocked, logged for audit.", ", ".join(found))
    return found


# ── Check 5: Topic Boundary ────────────────────────────────────────────

# Lightweight keyword heuristic — reject clearly off-topic requests
OFF_TOPIC_KEYWORDS: list[tuple[str, re.Pattern]] = [
    ("malware", re.compile(r"\b(write|create|generate)\s+(a\s+)?(virus|malware|ransomware|trojan|worm|exploit)\b", re.I)),
    ("hacking", re.compile(r"\b(hack\s+(into|the)|crack\s+(a\s+)?password|ddos\s+attack)\b", re.I)),
    ("illegal_content", re.compile(r"\b(child\s+(porn|abuse)|snuff\s+film|how\s+to\s+(make|manufacture)\s+(drugs?|bombs?))\b", re.I)),
    ("self_harm", re.compile(r"\b(how\s+to\s+(commit\s+)?suicide|ways\s+to\s+(kill|harm)\s+(myself|yourself))\b", re.I)),
]

def _check_topic_boundary(user_input: str) -> str | None:
    """Block clearly off-topic or harmful requests."""
    for name, pattern in OFF_TOPIC_KEYWORDS:
        if pattern.search(user_input):
            logger.warning("Input guard: off-topic request blocked (%s).", name)
            return f"Input blocked: your request appears to be outside the scope of this travel assistant."
    return None


# ── Main Guard Pipeline ────────────────────────────────────────────────

def guard_input(user_input: str) -> GuardResult:
    """Run the full input guard pipeline.

    Order: length → control chars → injection → pii → topic.

    Returns GuardResult with sanitized input (always) + pass/fail flag.
    """
    # 1. Length
    length_error = _check_length(user_input)
    if length_error:
        return GuardResult(passed=False, blocked_reason=length_error)

    # 2. Sanitize control characters
    sanitized = _sanitize_control_chars(user_input).strip()
    if not sanitized:
        return GuardResult(passed=False, blocked_reason="Input is empty after sanitisation.")

    # 3. Prompt injection
    injection_error = _detect_injection(sanitized)
    if injection_error:
        return GuardResult(passed=False, blocked_reason=injection_error)

    # 4. PII (log-only, never blocks)
    pii_found = _detect_pii(sanitized)

    # 5. Topic boundary
    topic_error = _check_topic_boundary(sanitized)
    if topic_error:
        return GuardResult(passed=False, blocked_reason=topic_error)

    return GuardResult(passed=True, sanitized_input=sanitized, pii_detected=pii_found)
