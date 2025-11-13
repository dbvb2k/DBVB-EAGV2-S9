"""
Heuristic validation utilities for GenAI / Agentic AI workflows.

The module exposes a suite of lightweight validation helpers that return
`ValidationResult` instances. These heuristics aim to catch common data quality,
security, or policy issues before tasks are delegated to external tools or LLM
calls. None of the checks below are exhaustive—each should be treated as a
signal rather than as formal proofs of correctness.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class ValidationResult:
    """Represents the outcome of a heuristic validation."""

    is_valid: bool
    reason: str = "Validation succeeded."

    @staticmethod
    def success(reason: str = "Validation succeeded.") -> "ValidationResult":
        return ValidationResult(True, reason)

    @staticmethod
    def failure(reason: str) -> "ValidationResult":
        return ValidationResult(False, reason)


def validate_file_metadata(
    file_name: str,
    file_size_bytes: int,
    allowed_types: Optional[Iterable[str]] = None,
    max_size_bytes: int = 50 * 1024 * 1024,
) -> ValidationResult:
    """
    Validate file extension and size against allowed values.
    File types should be provided without the leading dot.
    """
    allowed = {ft.lower() for ft in (allowed_types or {"pdf", "xlsx", "xls", "docx", "doc", "txt", "pptx", "ppt", "csv"})}
    if "." not in file_name:
        return ValidationResult.failure("File name has no extension.")
    ext = file_name.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return ValidationResult.failure(f"File type '.{ext}' is not permitted.")
    if file_size_bytes <= 0:
        return ValidationResult.failure("File size must be positive.")
    if file_size_bytes > max_size_bytes:
        return ValidationResult.failure("File size exceeds allowed limit.")
    return ValidationResult.success()


def validate_sql_query(
    query: str,
    allowed_statements: Optional[Iterable[str]] = None,
    disallowed_patterns: Optional[Iterable[str]] = None,
) -> ValidationResult:
    """
    Check whether a SQL query adheres to basic guardrails such as statement type
    and absence of common destructive keywords.
    """
    if not query or not query.strip():
        return ValidationResult.failure("Query is empty.")

    normalized = query.strip().upper()
    allowed = tuple((s.upper() for s in (allowed_statements or ("SELECT", "WITH"))))
    if not normalized.startswith(allowed):
        return ValidationResult.failure("Query must start with a read-only statement.")

    patterns = disallowed_patterns or ("DROP", "TRUNCATE", "ALTER", "DELETE", "--", ";--", "/*", "*/")
    for pattern in patterns:
        if pattern.upper() in normalized:
            return ValidationResult.failure(f"Query contains disallowed pattern: '{pattern}'.")
    return ValidationResult.success()


def validate_math_equation(equation: str) -> ValidationResult:
    """
    Validate that an equation contains only arithmetic operations.
    Ensures the expression can be parsed and consists of safe AST nodes.
    """
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Num,  # For compatibility with older Python versions
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )
    try:
        tree = ast.parse(equation, mode="eval")
    except SyntaxError as exc:
        return ValidationResult.failure(f"Invalid mathematical expression: {exc}.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            return ValidationResult.failure("Function calls are not permitted in equations.")
        if isinstance(node, ast.Name):
            if node.id not in {"x", "y", "z"}:
                return ValidationResult.failure(f"Unknown identifier '{node.id}'.")
        if not isinstance(node, allowed_nodes):
            return ValidationResult.failure(f"Disallowed token detected: '{type(node).__name__}'.")
    return ValidationResult.success()


def validate_input_format(value: str, expected_type: str, date_formats: Optional[Sequence[str]] = None) -> ValidationResult:
    """
    Validate a string value against an expected type.
    Supported types: number, integer, float, text, email, date, datetime.
    """
    if expected_type == "number":
        if not re.fullmatch(r"[+-]?(\d+(\.\d+)?|\.\d+)", value.strip()):
            return ValidationResult.failure("Value is not a valid number.")
    elif expected_type == "integer":
        if not re.fullmatch(r"[+-]?\d+", value.strip()):
            return ValidationResult.failure("Value is not a valid integer.")
    elif expected_type == "float":
        if not re.fullmatch(r"[+-]?\d*\.\d+", value.strip()):
            return ValidationResult.failure("Value is not a valid float.")
    elif expected_type == "text":
        if not value.strip():
            return ValidationResult.failure("Text input is empty.")
    elif expected_type == "email":
        return validate_email_address(value)
    elif expected_type in {"date", "datetime"}:
        formats = date_formats or ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S")
        for fmt in formats:
            try:
                datetime.strptime(value.strip(), fmt)
                return ValidationResult.success()
            except ValueError:
                continue
        return ValidationResult.failure(f"Value does not match expected {expected_type} formats.")
    else:
        return ValidationResult.failure(f"Unsupported expected type '{expected_type}'.")
    return ValidationResult.success()


def detect_prompt_injection(
    text: str,
    suspicious_patterns: Optional[Sequence[str]] = None,
    max_length: int = 4000,
) -> ValidationResult:
    """
    Heuristic detection of prompt-injection attempts.
    Looks for override instructions, unusual length, and jailbreak keywords.
    """
    if not text:
        return ValidationResult.failure("Prompt text is empty.")
    if len(text) > max_length:
        return ValidationResult.failure("Prompt length exceeds policy threshold.")

    default_patterns = suspicious_patterns or [
        r"ignore\s+previous\s+instructions",
        r"system\s*override",
        r"disregard\s+all\s+rules",
        r"reset\s+instructions",
        r"sudo\s+rm",
        r"\bshutdown\b",
        r"disable\s+logging",
        r"exfiltrate",
        r"prompt\s+leak",
    ]
    lowered = text.lower()
    for pattern in default_patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return ValidationResult.failure(f"Prompt appears to contain injection pattern: '{pattern}'.")
    return ValidationResult.success()


def validate_ascii_content(text: str, allow_extended: bool = False) -> ValidationResult:
    """Check whether the text contains only ASCII (or extended ASCII) characters."""
    if allow_extended:
        try:
            text.encode("latin-1")
        except UnicodeEncodeError:
            return ValidationResult.failure("Text contains characters outside extended ASCII.")
        return ValidationResult.success()

    if not text.isascii():
        return ValidationResult.failure("Text contains non-ASCII characters.")
    return ValidationResult.success()


def validate_tool_availability(tool_name: str, registered_tools: Iterable[str]) -> ValidationResult:
    """Validate that a requested tool is available in the registry."""
    normalized_registry = {t.lower() for t in registered_tools}
    if tool_name.lower() not in normalized_registry:
        return ValidationResult.failure(f"Tool '{tool_name}' is not registered.")
    return ValidationResult.success()


def validate_url(
    url: str,
    allowed_schemes: Iterable[str] = ("http", "https"),
    allow_local_hosts: bool = False,
) -> ValidationResult:
    """Validate a URL for scheme, host presence, and optional locality restrictions."""
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in {s.lower() for s in allowed_schemes}:
        return ValidationResult.failure(f"Scheme '{parsed.scheme}' is not permitted.")
    if not parsed.netloc:
        return ValidationResult.failure("URL must include a network location.")

    host = parsed.hostname or ""
    if not allow_local_hosts and host in {"localhost", "127.0.0.1"}:
        return ValidationResult.failure("Local hosts are disallowed.")
    return ValidationResult.success()


def validate_json_payload(
    payload: str,
    required_keys: Optional[Iterable[str]] = None,
    schema: Optional[Mapping[str, Tuple[type, ...]]] = None,
) -> ValidationResult:
    """
    Validate JSON payload structure, checking required keys and optional schema.
    Schema should map keys to tuples of acceptable Python types.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        return ValidationResult.failure(f"Invalid JSON: {exc}.")

    if required_keys:
        missing = [key for key in required_keys if key not in data]
        if missing:
            return ValidationResult.failure(f"JSON is missing required keys: {missing}.")

    if schema:
        for key, expected_types in schema.items():
            if key in data and not isinstance(data[key], expected_types):
                expected_names = ", ".join(t.__name__ for t in expected_types)
                return ValidationResult.failure(f"Key '{key}' must be of type(s): {expected_names}.")
    return ValidationResult.success()


def validate_email_address(email: str) -> ValidationResult:
    """Validate email address format using a conservative regular expression."""
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    if not re.fullmatch(pattern, email.strip()):
        return ValidationResult.failure("Email address format is invalid.")
    return ValidationResult.success()


def validate_datetime_string(
    value: str,
    allowed_formats: Sequence[str] = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"),
) -> ValidationResult:
    """Ensure a datetime string matches one of the allowed formats."""
    for fmt in allowed_formats:
        try:
            datetime.strptime(value.strip(), fmt)
            return ValidationResult.success()
        except ValueError:
            continue
    return ValidationResult.failure("Datetime value does not match allowed formats.")


def validate_language_code(language_code: str) -> ValidationResult:
    """
    Validate a language code against ISO 639-1 pattern with optional region.
    Examples: 'en', 'en-US', 'pt-BR'.
    """
    if re.fullmatch(r"^[a-z]{2}(-[A-Z]{2})?$", language_code.strip()):
        return ValidationResult.success()
    return ValidationResult.failure("Language code is not in ISO 639-1 format.")


def validate_temperature(value: float, min_value: float = 0.0, max_value: float = 1.0) -> ValidationResult:
    """Validate that a sampling temperature is within policy range."""
    if not (min_value <= value <= max_value):
        return ValidationResult.failure(f"Temperature {value} is outside allowed range [{min_value}, {max_value}].")
    return ValidationResult.success()


def validate_model_identifier(
    model: str,
    allowed_prefixes: Sequence[str] = ("gpt-", "llama-", "mistral-", "claude-", "custom-"),
) -> ValidationResult:
    """Check that a model identifier matches known prefixes."""
    normalized = model.lower()
    if not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        return ValidationResult.failure("Model identifier is not recognized.")
    if not re.fullmatch(r"[a-z0-9._\-]+", model):
        return ValidationResult.failure("Model identifier contains invalid characters.")
    return ValidationResult.success()


def validate_vector_dimension(vector: Sequence[Any], expected_dimension: int) -> ValidationResult:
    """Validate that a vector-like sequence has the expected dimensionality and numeric entries."""
    if len(vector) != expected_dimension:
        return ValidationResult.failure(
            f"Vector length {len(vector)} does not match expected dimension {expected_dimension}."
        )
    for index, value in enumerate(vector):
        if not isinstance(value, (int, float)):
            return ValidationResult.failure(f"Vector entry at index {index} is not numeric.")
    return ValidationResult.success()


def validate_token_budget(estimated_tokens: int, max_tokens: int, safety_margin: float = 0.1) -> ValidationResult:
    """
    Validate token usage against budget, optionally reserving a safety margin.
    """
    if estimated_tokens < 0:
        return ValidationResult.failure("Estimated tokens must be non-negative.")
    effective_limit = int(max_tokens * (1.0 - max(0.0, min(safety_margin, 0.9))))
    if estimated_tokens > effective_limit:
        return ValidationResult.failure(
            f"Estimated tokens {estimated_tokens} exceed safe budget {effective_limit} (margin {safety_margin:.0%})."
        )
    return ValidationResult.success()


def validate_response_length(text: str, max_characters: int = 2000) -> ValidationResult:
    """Heuristic check to keep responses within configured character limits."""
    length = len(text)
    if length > max_characters:
        return ValidationResult.failure(f"Response length {length} exceeds limit {max_characters}.")
    if length == 0:
        return ValidationResult.failure("Response is empty.")
    return ValidationResult.success()


def validate_metadata_schema(
    metadata: Mapping[str, Any],
    required_fields: Mapping[str, type],
    optional_fields: Optional[Mapping[str, type]] = None,
) -> ValidationResult:
    """
    Validate a metadata object for required fields and type hints.
    This is useful for verifying tool or agent capability descriptors.
    """
    missing = [field for field in required_fields if field not in metadata]
    if missing:
        return ValidationResult.failure(f"Metadata missing required fields: {missing}.")

    def _check_types(fields: Mapping[str, type]) -> Optional[str]:
        for key, expected_type in fields.items():
            if key in metadata and not isinstance(metadata[key], expected_type):
                return f"Field '{key}' must be of type {expected_type.__name__}."
        return None

    message = _check_types(required_fields)
    if message:
        return ValidationResult.failure(message)
    if optional_fields:
        message = _check_types(optional_fields)
        if message:
            return ValidationResult.failure(message)
    return ValidationResult.success()


def validate_api_key_format(api_key: str, pattern: str = r"^[A-Za-z0-9]{32,}$") -> ValidationResult:
    """
    Validate API keys used for external integrations.
    Default pattern expects alphanumeric strings of length >= 32.
    """
    if not api_key:
        return ValidationResult.failure("API key is empty.")
    if not re.fullmatch(pattern, api_key):
        return ValidationResult.failure("API key does not match required format.")
    return ValidationResult.success()


def validate_rate_limit(
    requests: int,
    interval_seconds: int,
    max_requests: int,
) -> ValidationResult:
    """
    Validate that a planned call rate complies with a known rate limit.
    """
    if interval_seconds <= 0:
        return ValidationResult.failure("Interval must be greater than zero seconds.")
    if max_requests <= 0:
        return ValidationResult.failure("Maximum requests must be positive.")
    if requests < 0:
        return ValidationResult.failure("Request count cannot be negative.")
    if requests > max_requests:
        return ValidationResult.failure(
            f"Requests {requests} exceed allowed maximum {max_requests} per {interval_seconds} seconds."
        )
    return ValidationResult.success()


__all__ = [
    "ValidationResult",
    "validate_file_metadata",
    "validate_sql_query",
    "validate_math_equation",
    "validate_input_format",
    "detect_prompt_injection",
    "validate_ascii_content",
    "validate_tool_availability",
    "validate_url",
    "validate_json_payload",
    "validate_email_address",
    "validate_datetime_string",
    "validate_language_code",
    "validate_temperature",
    "validate_model_identifier",
    "validate_vector_dimension",
    "validate_token_budget",
    "validate_response_length",
    "validate_metadata_schema",
    "validate_api_key_format",
    "validate_rate_limit",
]


