from __future__ import annotations


SENSITIVE_HINTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "auth",
    "key",
)


def _is_sensitive_key(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(h in t for h in SENSITIVE_HINTS)


def redact_cli_args(argv: list[str]) -> list[str]:
    """Redact obvious secrets from command-line args before writing logs."""
    out: list[str] = []
    redact_next_value = False

    for arg in argv:
        raw = str(arg)

        if redact_next_value and not raw.startswith("-"):
            out.append("***")
            redact_next_value = False
            continue
        redact_next_value = False

        if "=" in raw:
            key, value = raw.split("=", 1)
            if _is_sensitive_key(key):
                out.append(f"{key}=***")
            else:
                out.append(raw)
            continue

        if raw.startswith("-"):
            out.append(raw)
            if _is_sensitive_key(raw):
                redact_next_value = True
            continue

        if _is_sensitive_key(raw):
            out.append("***")
        else:
            out.append(raw)

    return out


def csv_safe_text(value: str) -> str:
    """Prevent spreadsheet formula execution when CSV is opened in Excel/Sheets."""
    text = str(value)
    if text and text[0] in ("=", "+", "-", "@"):
        return "'" + text
    return text

