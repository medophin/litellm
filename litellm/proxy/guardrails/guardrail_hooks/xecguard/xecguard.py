"""
XecGuard guardrail integration for LiteLLM.

Calls the CyCraft XecGuard API (https://api-xecguard.cycraft.ai)
to scan the full conversation history against configured policies
(prompt-injection, PII, harmful-content, custom rules) and, when
grounding documents are supplied via request metadata, also validates
the assistant response against those reference documents via the
/grounding endpoint.

Design notes (intentional divergences from the framework defaults):
  * The full conversation history (system + user + assistant) is always
    forwarded to XecGuard regardless of ``scan_type``. This bypasses the
    framework's optional ``skip_system_message_in_guardrail`` behaviour
    on purpose - policy enforcement depends on system-prompt visibility.
  * ``apply_guardrail`` is defined directly on this class so the
    ``during_call`` dispatch (proxy/utils.py checks for the method on
    ``type(callback).__dict__``) reaches our implementation.
  * ``async_logging_hook`` is overridden because the framework calls it
    directly for ``logging_only`` mode - it does NOT bridge to
    ``apply_guardrail``. Our override runs the scan non-blockingly and
    swallows every exception.
  * ``async_post_call_streaming_iterator_hook`` is defined directly on this
    class (not via ``use_native_lifecycle_hooks``) so it is picked up as an
    "override" callback purely by having its own entry in the class's
    ``__dict__`` (see ``ProxyLogging._callback_capabilities``). pre_call,
    during_call and non-streaming post_call keep going through the generic
    ``apply_guardrail``/``unified_guardrail`` dispatch untouched.
  * When ``send_meta`` is enabled the scan payload carries a ``meta``
    object identifying the calling virtual key. It is correlation data
    for XecGuard's SIEM export only and never affects the verdict; the
    backend's flat-scalar contract for it is enforced client-side so a
    malformed key metadata entry cannot fail an otherwise valid scan.
"""

import asyncio
import copy
import json
import os
import re
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from functools import reduce
from typing import TYPE_CHECKING, Any, Final, Literal, Optional, cast

import httpx
from fastapi.exceptions import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.core_helpers import redact_nested_match_and_regex_keys
from litellm.litellm_core_utils.sensitive_data_masker import mask_credentials_in_payload
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_system_message_for_guardrail,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import (
    LiteLLM_ManagementEndpoint_MetadataFields,
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    Delta,
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    GuardrailTracingDetail,
    ModelResponseStream,
    StreamingChoices,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp import MCPPostCallResponseObject
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )


def _poc_safe(obj: object, depth: int = 0) -> object:
    """Reduce an arbitrary value to JSON-safe primitives for the PoC field dump."""
    if depth > 6:
        return f"<max-depth {type(obj).__name__}>"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= 4000 else obj[:4000] + f"...<+{len(obj) - 4000} chars>"
    if hasattr(obj, "model_dump"):
        try:
            dumped = obj.model_dump()  # pyright: ignore[reportAttributeAccessIssue]  # duck-typed pydantic BaseModel
            return _poc_safe(dumped, depth + 1)
        except Exception:
            return str(obj)[:4000]
    if isinstance(obj, Mapping):
        return {str(k): _poc_safe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_poc_safe(v, depth + 1) for v in obj]
    return str(obj)[:4000]


def _poc_dump(hook: str, guardrail_name: str | None, **fields: object) -> None:
    """PoC-only field capture, gated by env XECGUARD_POC_LOG. Never raises."""
    path: Final = os.environ.get("XECGUARD_POC_LOG")
    if not path:
        return
    try:
        record: Final = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "hook": hook,
            "guardrail": guardrail_name,
            **{k: _poc_safe(v) for k, v in fields.items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        verbose_proxy_logger.debug("XecGuard PoC dump failed (ignored): %s", exc)


def _split_list_param(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Items of a list-valued setting that is stored as a comma-separated string
    (what the admin UI text box submits) or written as a YAML list."""
    if not value:
        return ()
    raw: Final = value.split(",") if isinstance(value, str) else value
    return tuple(item.strip() for item in raw if isinstance(item, str) and item.strip())


def _sanitize_scan_result_for_logging(scan_result: dict) -> dict:
    without_secrets: Final = {key: value for key, value in scan_result.items() if key != "secret_fields"}
    redacted: Final = redact_nested_match_and_regex_keys(without_secrets)
    masked: Final = mask_credentials_in_payload(redacted if isinstance(redacted, dict) else without_secrets)
    return masked if isinstance(masked, dict) else without_secrets


class _TimedResult(dict):
    """A XecGuard answer that remembers when its request started and ended, so the guardrail record
    carries the real scan time whichever path (request, reply, stream window) made the call."""

    started: float
    ended: float


def _scan_window(scan_result: Mapping[str, object]) -> tuple[float | None, float | None]:
    if isinstance(scan_result, _TimedResult):
        return scan_result.started, scan_result.ended
    return None, None


_LOGGED_SCAN_LABEL: Final = {"input": "request scan", "response": "response scan"}
_DEFAULT_API_BASE: Final = "https://api-xecguard.cycraft.ai"
_SCAN_ENDPOINT: Final = "/xecguard/v1/scan"
_GROUNDING_ENDPOINT: Final = "/xecguard/v1/grounding"
_TRACE_ENDPOINT: Final = "/xecguard/v1/service/trace"
_SKIPPED_POLICIES_HEADER: Final = "x-xecguard-skipped-policies"
_DEFAULT_RATIONALE_POLL_SECONDS: Final = 120.0
_TRACE_DENIED_STATUSES: Final = frozenset({401, 403})
_TRACE_NOT_FOUND_GRACE_SECONDS: Final = 60.0
_NO_REQUEST_TEXT: Final = "(request text unavailable)"
_DEFAULT_MODEL: Final = "xecguard_v2"
_DEFAULT_CUSTOM_POLICY_MIN_CHARS: Final = 500
_DEFAULT_GROUNDING_STRICTNESS: Final = "BALANCED"
_METADATA_GROUNDING_KEY: Final = "xecguard_grounding_documents"
_RATIONALE_TRUNCATE_CHARS: Final = 200
_DEFAULT_POLICIES: Final = [
    "Default_Policy_SystemPromptEnforcement",
    "Default_Policy_HarmfulContentProtection",
    "Default_Policy_GeneralPromptAttackProtection",
]

# ``meta`` contract of POST /xecguard/v1/scan: an optional object carrying caller
# context that takes no part in detection. XecGuard flattens it into the SIEM
# event (``virtualkey`` -> ``ctx_virtualkey``, ``data.X`` -> ``ctx_X``), and SIEM
# index fields only accept flat scalars - anything else is rejected with 400. So
# every value is coerced or dropped here rather than risking a scan failure.
_METADATA_KEY_METADATA_FIELD: Final = "user_api_key_metadata"
_META_NAME_PATTERN: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")
# Chars a custom-header-derived meta.data field name must not carry (spaces,
# colons, etc. - whatever _META_NAME_PATTERN's body class excludes with the
# leading-char rule stripped out, since that's handled separately below).
_META_INVALID_NAME_CHARS: Final = re.compile(r"[^A-Za-z0-9_.\-]+")
_META_CONTROL_CHARS: Final = re.compile(r"[\x00-\x1f\x7f]")
_META_MAX_DATA_FIELDS: Final = 32
_META_MAX_VALUE_CHARS: Final = 512
_META_MAX_SERIALIZED_BYTES: Final = 4096
# Never forwarded, and ``meta_data_fields`` cannot opt them back in: these slots
# hold credentials, so there is no configuration under which shipping them to an
# external SIEM is right.
_META_EXCLUDED_DATA_FIELDS: Final = frozenset({"logging", "callback_settings", "secret_manager_settings"})

# The proxy stores its own per-key control settings inside key metadata - rate
# limits, budget knobs, enforced params, ``disable_global_guardrails``. They sit
# in the same dict as the admin's own fields but they are proxy configuration,
# not caller identity: noise in a SIEM, they eat the 32-field / 4096-byte budget,
# and a couple of them describe the key's security posture. Skipped by default,
# but an admin who explicitly names one in ``meta_data_fields`` gets it - unlike
# the credential slots above, forwarding these is a judgement call, not a bug.
#
# Taken from the proxy's own lists rather than copied, so a field litellm adds
# later is covered without an edit here.
_META_CONTROL_DATA_FIELDS: Final = (
    frozenset(LiteLLM_ManagementEndpoint_MetadataFields) | frozenset(LiteLLM_ManagementEndpoint_MetadataFields_Premium)
) - _META_EXCLUDED_DATA_FIELDS

# Two shapes for ``meta.virtualkey``. "string" is the identity as a bare string,
# which is all the currently deployed backend accepts. "object" carries the alias
# and the key id side by side, so a SIEM event is attributable even when the alias
# is absent, renamed, or reused - it needs a backend that validates the object
# form, hence the switch rather than a straight cutover.
_META_IDENTITY_FORMATS: Final = ("string", "object")
_DEFAULT_META_IDENTITY_FORMAT: Final = "string"

# Virtual-key attributes the proxy injects alongside every request, forwarded as
# ``meta.data`` so a SIEM event can be attributed without a lookup back into the
# proxy database. Ordered: identity first, then tenancy, then commercials, so the
# fields that survive the 32-field / 4096-byte caps are the ones worth keeping.
#
# This set deliberately includes PII (``user_email``) and commercial figures
# (``spend``, ``max_budget``). Both leave the proxy only when ``send_meta`` is
# explicitly enabled, and ``meta_data_fields`` narrows the set for deployments
# that must not egress them.
_META_AUTO_DATA_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("key_id", "user_api_key_hash"),
    ("key_alias", "user_api_key_alias"),
    ("team_id", "user_api_key_team_id"),
    ("team_alias", "user_api_key_team_alias"),
    ("user_id", "user_api_key_user_id"),
    ("user_email", "user_api_key_user_email"),
    ("org_id", "user_api_key_org_id"),
    ("org_alias", "user_api_key_org_alias"),
    ("project_id", "user_api_key_project_id"),
    ("project_alias", "user_api_key_project_alias"),
    ("end_user_id", "user_api_key_end_user_id"),
    ("spend", "user_api_key_spend"),
    ("max_budget", "user_api_key_max_budget"),
    ("request_route", "user_api_key_request_route"),
)


_METADATA_BUCKETS: Final = ("litellm_metadata", "metadata")


def _proxy_metadata_field(request_data: Mapping[str, Any], name: str) -> object:
    """``name`` from the proxy's metadata, wherever the proxy put it.

    Chat keeps its fields in ``metadata``; ``/v1/responses`` and ``/v1/messages`` keep them in
    ``litellm_metadata`` and leave ``metadata`` to the caller, so the first bucket that is a dict
    is not necessarily the one that has the field. ``litellm_metadata`` is read first: it is only
    ever written by the proxy, while a caller can put any name into ``metadata``.
    """
    for bucket in _METADATA_BUCKETS:
        candidate = request_data.get(bucket)
        if isinstance(candidate, dict) and candidate.get(name) is not None:
            return candidate[name]
    return None


def _sanitized_meta_text(text: str) -> str | None:
    """Strip control characters and cap the length, or None when nothing is left."""
    cleaned: Final = _META_CONTROL_CHARS.sub("", text)[:_META_MAX_VALUE_CHARS]
    return cleaned or None


_SSE_TEXT_FIELD = re.compile(r'"(?:text|partial_json)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _tool_call_dict(tool_call: object) -> dict:
    """A tool call as plain JSON. Some translations hand over pydantic objects rather than dicts."""
    dumped: Final = tool_call.model_dump(exclude_none=True) if hasattr(tool_call, "model_dump") else tool_call
    return dumped if isinstance(dumped, dict) else {}


class _TextCapture(CustomGuardrail):
    """Takes the reply text a litellm translation hands to ``apply_guardrail`` and nothing
    else - how XecGuard reads a response with litellm's own per-endpoint parsing."""

    def __init__(self) -> None:
        super().__init__(guardrail_name="xecguard-text-capture")
        self.text = ""
        self.tool_calls: list = []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        self.text = "\n".join(t for t in inputs.get("texts") or () if t)
        self.tool_calls = [_tool_call_dict(call) for call in inputs.get("tool_calls") or ()]
        return inputs


_RATIONALE_READY: Final = "READY"
_RATIONALE_PENDING: Final = "ANALYSIS_REQUIRED"
_POLL_FIRST_INTERVAL_SECONDS: Final = 2.0
_POLL_MAX_INTERVAL_SECONDS: Final = 30.0
_SPEND_LOG_ATTEMPTS: Final = 15
_SPEND_LOG_RETRY_SECONDS: Final = 2.0
_SPEND_LOG_LOOKBACK: Final = timedelta(hours=1)

_FIND_SPEND_LOG_SQL: Final = """
SELECT request_id, metadata FROM "LiteLLM_SpendLogs"
WHERE "startTime" >= $2::timestamp AND (request_id = $1 OR metadata->>'litellm_call_id' = $1)
LIMIT 1
"""
_UPDATE_SPEND_LOG_SQL: Final = 'UPDATE "LiteLLM_SpendLogs" SET metadata = $2::jsonb WHERE request_id = $1'

_FetchTrace = Callable[[str], Awaitable[Mapping[str, object] | None]]
_StoreRationale = Callable[[str, Mapping[str, str]], Awaitable[bool]]

_backfill_tasks: Final[set["asyncio.Task[None]"]] = set()  # mutable-ok: keeps background tasks from being collected


def _has_pending_rationale(violations: Sequence[Mapping[str, object]]) -> bool:
    return any(v.get("rationale_status") == _RATIONALE_PENDING for v in violations)


def _ready_rationales(trace: Mapping[str, object]) -> tuple[Mapping[str, str], bool]:
    """Policy name -> rationale for every reason that is READY, and whether all of them are."""
    scan_result: Final = trace.get("scan_result")
    raw: Final = scan_result.get("reasons") if isinstance(scan_result, Mapping) else None
    reasons: Final = tuple(r for r in raw if isinstance(r, Mapping)) if isinstance(raw, list) else ()
    ready: Final = {
        str(r["violated_policy_name"]): str(r["rationale"])
        for r in reasons
        if r.get("rationale_status") == _RATIONALE_READY and r.get("violated_policy_name") and r.get("rationale")
    }
    return ready, bool(reasons) and all(r.get("rationale_status") == _RATIONALE_READY for r in reasons)


def _patch_spend_log_metadata(
    metadata: Mapping[str, object], guardrail_name: str, trace_id: str, rationales: Mapping[str, str]
) -> dict[str, object] | None:
    """``metadata`` with the real rationales in this scan's guardrail entry, or None when no entry holds them.

    An entry whose ``guardrail_response`` was redacted carries no scan to patch, so it is left alone.
    """
    entries: Final = metadata.get("guardrail_information")
    if not isinstance(entries, list):
        return None
    patched: Final = tuple(
        _patch_entry(entry, trace_id, rationales) if _is_scan_entry(entry, guardrail_name, trace_id) else entry
        for entry in entries
    )
    if all(new is old for new, old in zip(patched, entries)):
        return None
    return {**metadata, "guardrail_information": list(patched)}


def _is_scan_entry(entry: object, guardrail_name: str, trace_id: str) -> bool:
    if not isinstance(entry, Mapping) or entry.get("guardrail_name") != guardrail_name:
        return False
    response: Final = entry.get("guardrail_response")
    return isinstance(response, Mapping) and response.get("trace_id") == trace_id


def _patch_error_message(message: str, trace_id: str, rationales: Mapping[str, str]) -> str | None:
    """``message`` with the placeholder rationales of this scan's block replaced, or None when it is not
    that block or nothing in it changes. The message is the repr of the 400 the client got: the summary
    after ``rationale=`` and each violation's own ``'rationale': '...'``, found by the policy named before it."""
    if f"trace_id={trace_id}" not in message:
        return None
    patched = message
    for policy, rationale in rationales.items():
        pattern = re.compile(
            r"(\{[^{}]*?'violated_policy_name': '" + re.escape(policy) + r"'[^{}]*?'rationale': )'[^']*'"
            r"(, 'rationale_status': ')" + _RATIONALE_PENDING + "'"
        )
        patched = pattern.sub(lambda m: f"{m.group(1)}{rationale!r}{m.group(2)}{_RATIONALE_READY}'", patched)
    first: Final = next(iter(rationales.values()), None)
    if first is not None:
        patched = re.sub(
            r"(trace_id=" + re.escape(trace_id) + r" rationale=)[^']*(')",
            lambda m: f"{m.group(1)}{first[:_RATIONALE_TRUNCATE_CHARS]}{m.group(2)}",
            patched,
            count=1,
        )
    return None if patched == message else patched


def _patch_error_information(
    metadata: Mapping[str, object], trace_id: str, rationales: Mapping[str, str]
) -> dict[str, object] | None:
    error: Final = metadata.get("error_information")
    message: Final = error.get("error_message") if isinstance(error, Mapping) else None
    if not isinstance(error, Mapping) or not isinstance(message, str):
        return None
    patched: Final = _patch_error_message(message, trace_id, rationales)
    return None if patched is None else {**metadata, "error_information": {**error, "error_message": patched}}


def _patch_entry(entry: object, trace_id: str, rationales: Mapping[str, str]) -> object:
    entry_map: Final = cast(Mapping[str, object], entry)
    response: Final = cast(Mapping[str, object], entry_map["guardrail_response"])
    violations: Final = response.get("xecguard_result")
    if not isinstance(violations, list):
        return entry
    patched_violations: Final = [_patch_violation(v, rationales) for v in violations]
    context: Final = response.get("request_context")
    first: Final = next(
        (v["rationale"] for v in patched_violations if isinstance(v, Mapping) and v.get("rationale")), None
    )
    patched_context: Final = {**context, "rationale": first} if isinstance(context, Mapping) and first else context
    return {
        **entry_map,
        "guardrail_response": {
            **response,
            "xecguard_result": patched_violations,
            **({"request_context": patched_context} if patched_context is not None else {}),
        },
    }


def _patch_violation(violation: object, rationales: Mapping[str, str]) -> object:
    if not isinstance(violation, Mapping) or violation.get("rationale_status") != _RATIONALE_PENDING:
        return violation
    real: Final = rationales.get(str(violation.get("violated_policy_name")))
    if real is None:
        return violation
    return {**violation, "rationale": real, "rationale_status": _RATIONALE_READY}


async def _poll_ready_rationales(
    fetch_trace: _FetchTrace,
    trace_id: str,
    timeout_seconds: float,
    first_interval_seconds: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    now: Callable[[], float] = time.monotonic,
) -> Mapping[str, str]:
    """Poll the trace until every reason is READY or the time is up, keeping what was READY by then.
    The wait doubles after each try up to ``_POLL_MAX_INTERVAL_SECONDS``, as the trace endpoint asks."""
    started: Final = now()
    deadline: Final = started + timeout_seconds
    ready: Mapping[str, str] = {}
    interval = first_interval_seconds
    while True:
        trace = await fetch_trace(trace_id)
        if trace is not None:
            ready, complete = _ready_rationales(trace)
            if complete:
                return ready
        elif not ready and now() - started >= _TRACE_NOT_FOUND_GRACE_SECONDS:
            return ready
        if now() + interval > deadline:
            return ready
        await sleep(interval)
        interval = min(interval * 2, _POLL_MAX_INTERVAL_SECONDS)


async def _backfill_rationale(
    *,
    fetch_trace: _FetchTrace,
    store_rationale: _StoreRationale,
    request_id: str | None,
    trace_id: str,
    timeout_seconds: float,
    first_interval_seconds: float = _POLL_FIRST_INTERVAL_SECONDS,
    store_retry_seconds: float = _SPEND_LOG_RETRY_SECONDS,
    store_attempts: int = _SPEND_LOG_ATTEMPTS,
    log_context: Mapping[str, object] | None = None,
) -> None:
    try:
        rationales: Final = await _poll_ready_rationales(fetch_trace, trace_id, timeout_seconds, first_interval_seconds)
        if not rationales:
            verbose_proxy_logger.warning("XecGuard rationale for trace_id=%s was not ready in time", trace_id)
            return
        verbose_proxy_logger.info(
            "XecGuard rationale ready: %s",
            json.dumps(
                {**(log_context or {}), "request_id": request_id, "trace_id": trace_id, "rationales": rationales},
                ensure_ascii=False,
                default=str,
            ),
        )
        if request_id is None:
            return
        for attempt in range(store_attempts):
            if await store_rationale(request_id, rationales):
                return
            if attempt + 1 < store_attempts:
                await asyncio.sleep(store_retry_seconds)
        verbose_proxy_logger.warning("XecGuard rationale for trace_id=%s found no spend log row to update", trace_id)
    except Exception as exc:
        verbose_proxy_logger.warning("XecGuard rationale backfill for trace_id=%s failed: %s", trace_id, exc)


def _start_backfill(
    *,
    fetch_trace: _FetchTrace,
    store_rationale: _StoreRationale,
    request_id: str | None,
    trace_id: str,
    timeout_seconds: float,
    log_context: Mapping[str, object] | None = None,
) -> None:
    try:
        loop: Final = asyncio.get_running_loop()
    except RuntimeError:
        return
    task: Final = loop.create_task(
        _backfill_rationale(
            fetch_trace=fetch_trace,
            store_rationale=store_rationale,
            request_id=request_id,
            trace_id=trace_id,
            timeout_seconds=timeout_seconds,
            log_context=log_context,
        )
    )
    _backfill_tasks.add(task)
    task.add_done_callback(_backfill_tasks.discard)


async def _update_spend_log(guardrail_name: str, trace_id: str, request_id: str, rationales: Mapping[str, str]) -> bool:
    """Rewrite the guardrail entry of the spend log row for ``request_id``. False means try again later:
    the row is written in a batch after the request, so it may not exist yet."""
    from litellm.proxy.proxy_server import prisma_client
    from litellm.proxy.spend_tracking.spend_tracking_utils import should_store_prompts_and_responses_in_spend_logs

    if prisma_client is None:
        return True
    since: Final = (datetime.now(timezone.utc) - _SPEND_LOG_LOOKBACK).strftime("%Y-%m-%dT%H:%M:%S")
    rows: Final = await prisma_client.db.query_raw(_FIND_SPEND_LOG_SQL, request_id, since)
    if not rows:
        return False
    row_id: Final = str(rows[0]["request_id"])
    stored: Final = rows[0].get("metadata")
    metadata: Final = json.loads(stored) if isinstance(stored, str) else stored
    if not isinstance(metadata, dict):
        return True
    with_error: Final = _patch_error_information(metadata, trace_id, rationales) or metadata
    with_entry: Final = (
        _patch_spend_log_metadata(with_error, guardrail_name, trace_id, rationales)
        if should_store_prompts_and_responses_in_spend_logs()
        else None
    )
    patched: Final = with_entry or (with_error if with_error is not metadata else None)
    if patched is None:
        return True
    await prisma_client.db.execute_raw(_UPDATE_SPEND_LOG_SQL, row_id, json.dumps(patched, default=str))
    return True


class XecGuardMissingCredentials(Exception):
    pass


class XecGuardGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        xecguard_model: str | None = None,
        policy_names: list[str] | None = None,
        custom_policy_names: str | Sequence[str] | None = None,
        custom_policy_min_chars: int | None = None,
        apply_to_aliases: str | Sequence[str] | None = None,
        except_aliases: str | Sequence[str] | None = None,
        send_meta: bool | None = None,
        meta_data_fields: str | Sequence[str] | None = None,
        custom_header_meta_fields: str | Sequence[str] | None = None,
        meta_identity_format: str | None = None,
        block_on_error: bool | None = None,
        grounding_strictness: str | None = None,
        stream_mode: Literal["window", "buffer", "audit"] = "window",
        stream_chunk: int = 1200,
        stream_context: int = 120,
        stream_first: bool = False,
        rationale_poll_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("XECGUARD_API_KEY")
        if not self.api_key:
            raise XecGuardMissingCredentials(
                "XecGuard API key is required. "
                "Set XECGUARD_API_KEY in the "
                "environment or pass api_key in "
                "the guardrail config."
            )

        self.api_base = (api_base or os.environ.get("XECGUARD_API_BASE") or _DEFAULT_API_BASE).rstrip("/")

        self.xecguard_model = xecguard_model or _DEFAULT_MODEL
        self.policy_names = policy_names
        self.custom_policy_names = _split_list_param(custom_policy_names)
        self.custom_policy_min_chars = (
            _DEFAULT_CUSTOM_POLICY_MIN_CHARS if custom_policy_min_chars is None else max(0, custom_policy_min_chars)
        )
        # Guardrail-side key targeting (free, OSS). Normalized to lists.
        self.apply_to_aliases = _split_list_param(apply_to_aliases)
        self.except_aliases = _split_list_param(except_aliases)

        # Caller context forwarded as the scan payload's ``meta``. Opt-in: turning
        # it on sends the calling key's alias and its admin-set metadata to
        # XecGuard, which is a data-egress change no upgrade should make silently.
        if send_meta is None:
            self.send_meta = os.environ.get("XECGUARD_SEND_META", "false").lower() in (
                "true",
                "1",
                "yes",
            )
        else:
            self.send_meta = send_meta
        self.meta_data_fields = _split_list_param(meta_data_fields)
        # Request headers to forward into meta.data, named as the caller sends
        # them (e.g. "ComputerName"). Matched against proxy_server_request.headers
        # case-insensitively (the proxy lower-cases header names before storing
        # them); the meta.data field name is derived from the configured header
        # name - see _custom_header_meta_items.
        self.custom_header_meta_fields = _split_list_param(custom_header_meta_fields)

        # Wire shape of ``meta.virtualkey``. Defaults to the string form: a backend
        # that only accepts strings answers the object form with 400, and with
        # ``block_on_error`` on (the default) that turns every request into a block.
        # An unknown value falls back rather than raising - a typo in the UI should
        # not take the gateway down.
        requested_format: Final = (
            (meta_identity_format or os.environ.get("XECGUARD_META_IDENTITY_FORMAT") or "").strip().lower()
        )
        if requested_format and requested_format not in _META_IDENTITY_FORMATS:
            verbose_proxy_logger.warning(
                "XecGuard: unknown meta_identity_format %r - falling back to %r (valid: %s)",
                requested_format,
                _DEFAULT_META_IDENTITY_FORMAT,
                ", ".join(_META_IDENTITY_FORMATS),
            )
        self.meta_identity_format = (
            requested_format if requested_format in _META_IDENTITY_FORMATS else _DEFAULT_META_IDENTITY_FORMAT
        )

        if block_on_error is None:
            env: Final = os.environ.get("XECGUARD_BLOCK_ON_ERROR", "true")
            self.block_on_error = env.lower() in (
                "true",
                "1",
                "yes",
            )
        else:
            self.block_on_error = block_on_error

        self.grounding_strictness = grounding_strictness or _DEFAULT_GROUNDING_STRICTNESS

        self.stream_mode = stream_mode
        self.stream_chunk = max(1, stream_chunk)
        self.stream_context = max(0, stream_context)
        self.stream_first = stream_first
        self.rationale_poll_seconds = (
            _DEFAULT_RATIONALE_POLL_SECONDS if rationale_poll_seconds is None else max(0.0, rationale_poll_seconds)
        )
        self._warned_system_stripped = False

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))

        super().__init__(**kwargs)

        self.mask_response_content = True

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.xecguard import (
            XecGuardConfigModel,
        )

        return XecGuardConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
            GuardrailEventHooks.pre_mcp_call,
            GuardrailEventHooks.during_mcp_call,
            GuardrailEventHooks.post_mcp_call,
        ]

    async def async_post_mcp_tool_call_hook(
        self, kwargs: dict, response_obj: "MCPPostCallResponseObject", start_time: object, end_time: object
    ) -> "MCPPostCallResponseObject | None":
        _poc_dump(
            "async_post_mcp_tool_call_hook",
            self.guardrail_name,
            kwargs_keys=sorted(str(k) for k in kwargs),
            mcp_tool_call_metadata=kwargs.get("mcp_tool_call_metadata"),
            standard_logging_mcp=(kwargs.get("standard_logging_object") or {}).get("mcp_tool_call_metadata")
            if isinstance(kwargs.get("standard_logging_object"), dict)
            else None,
            response_obj=response_obj,
        )
        return None

    @staticmethod
    def _calling_key_identity(
        request_data: Mapping[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        """Return (key_alias, key_hash) of the calling virtual key from the
        proxy-injected request metadata. Both may be None (e.g. master key)."""
        if not isinstance(request_data, dict):
            return None, None
        sources: Final = tuple(
            md
            for md in (request_data.get(meta_key) for meta_key in ("metadata", "litellm_metadata"))
            if isinstance(md, dict)
        )
        alias: Final = next((md["user_api_key_alias"] for md in sources if md.get("user_api_key_alias")), None)
        key_hash: Final = next((md["user_api_key_hash"] for md in sources if md.get("user_api_key_hash")), None)
        return alias, key_hash

    def _key_is_targeted(self, request_data: Mapping[str, Any] | None) -> bool:
        """Guardrail-side key targeting. With no allow/block list configured,
        every key is scanned. Otherwise the calling key is matched by alias
        (preferred) or hashed token:
          * blocklist (except_aliases): listed keys are NOT scanned;
          * allowlist (apply_to_aliases): only listed keys are scanned.
        When both are set, a key is scanned iff it is in the allowlist AND not
        in the blocklist.
        """
        allowlist: Final = self.apply_to_aliases or ()
        blocklist: Final = self.except_aliases or ()
        if not allowlist and not blocklist:
            return True

        alias, key_hash = self._calling_key_identity(request_data)
        identifiers: Final = tuple(ident for ident in (alias, key_hash) if ident)

        # Deny wins, and it is checked first so that precedence stays visible
        # rather than folded into the allowlist expression below.
        if blocklist and any(ident in blocklist for ident in identifiers):
            return False
        if not allowlist:
            return True
        return any(ident in allowlist for ident in identifiers)

    # Metadata fields the proxy injects to identify the calling virtual key.
    _KEY_IDENTITY_FIELDS = ("user_api_key_alias", "user_api_key_hash")

    @classmethod
    def _key_context(cls, data: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        """Return a mapping ``_calling_key_identity`` can read the key fields from.

        That reader looks for top-level ``metadata`` / ``litellm_metadata``. On the
        pre/during/post_call paths the proxy already puts the injected key fields
        there, so ``data`` is handed back untouched -- reshaping to a single key would
        drop the other location it also reads. Only the logging path needs help: there
        ``data`` is ``model_call_details``, which carries the same fields one level
        down under ``litellm_params``.
        """
        if not isinstance(data, dict):
            return data
        for meta_key in ("metadata", "litellm_metadata"):
            md = data.get(meta_key)
            if isinstance(md, dict) and any(field in md for field in cls._KEY_IDENTITY_FIELDS):
                return data
        nested: Final = data.get("litellm_params")
        if isinstance(nested, dict):
            for meta_key in ("metadata", "litellm_metadata"):
                md = nested.get(meta_key)
                if isinstance(md, dict):
                    return {meta_key: md}  # mutable-ok: lifts nested metadata to the readers' shape
        return data

    def should_run_guardrail(self, data: Mapping[str, object], event_type: GuardrailEventHooks) -> bool:
        """Gate on the calling virtual key in addition to the native checks.

        Deciding here rather than inside ``apply_guardrail`` is what makes LiteLLM
        record the guardrail as not having run for a key this guardrail does not
        cover, instead of logging a "success"/"allow" entry for a request it never
        evaluated. ``super()`` is consulted first so the native decisions -- global
        opt-outs, event-hook matching, tag-based modes -- keep precedence.

        The gates are still enforced inside ``apply_guardrail`` and
        ``async_logging_hook`` as well: ``POST /guardrails/apply_guardrail`` invokes
        ``apply_guardrail`` directly and never reaches this method.
        """
        if not super().should_run_guardrail(data, event_type):
            return False
        if event_type == GuardrailEventHooks.logging_only and not self._selected_for_logging(data):
            return False
        return self._key_is_targeted(self._key_context(data))

    def _selected_for_logging(self, data: Mapping[str, Any]) -> bool:
        """LiteLLM runs every logging_only guardrail on every request. Apply the rule the other modes
        follow instead: a default_on guardrail always runs, any other only when the request names it."""
        if self.default_on is True:
            return True
        nested: Final = data.get("litellm_params")
        sources: Final = (data, nested) if isinstance(nested, dict) else (data,)
        return any(
            self._guardrail_is_in_requested_guardrails(self.get_guardrail_from_metadata(source)) for source in sources
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        _poc_dump(
            "apply_guardrail",
            self.guardrail_name,
            input_type=input_type,
            inputs=inputs,
            request_data_keys=sorted(str(k) for k in request_data),
            request_data=request_data,
        )
        # Guardrail-side key targeting (allowlist / blocklist by key alias):
        # skip scanning entirely for keys this guardrail does not cover.
        # should_run_guardrail already gates the proxy's own dispatch paths; this
        # also covers POST /guardrails/apply_guardrail, which calls straight in.
        if not self._key_is_targeted(self._key_context(request_data)):
            return inputs

        mcp_scan: Final = self._mcp_tool_scan(request_data, inputs, input_type)
        messages: Final = (
            list(mcp_scan[0])
            if mcp_scan is not None
            else self._build_full_history(request_data=request_data, inputs=inputs, input_type=input_type)
        )
        if not messages:
            return inputs

        scan_type: Final = mcp_scan[1] if mcp_scan is not None else ("input" if input_type == "request" else "response")
        scan_result: Final = await self._call_scan(
            messages=messages,
            scan_type=scan_type,
            request_data=request_data,
        )
        if scan_result is None:
            return inputs

        event_type: Final = self._scan_event_type(request_data, input_type)
        if scan_result.get("decision") == "UNSAFE":
            self._record_block(request_data, scan_result, event_type)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": self._format_scan_block_message(scan_result),
                    "guardrail_name": self.guardrail_name or "xecguard",
                    "xecguard_response": scan_result,
                },
            )

        if input_type == "response":
            documents: Final = self._extract_grounding_documents(request_data)
            if documents:
                grounding_result: Final = await self._call_grounding(
                    messages=messages,
                    documents=documents,
                )
                if grounding_result is not None and grounding_result.get("decision") == "UNSAFE":
                    self._record_block(request_data, grounding_result, event_type)
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": self._format_grounding_block_message(grounding_result),
                            "guardrail_name": self.guardrail_name or "xecguard",
                            "xecguard_response": grounding_result,
                        },
                    )

        self._record_scan_outcome(
            request_data=request_data,
            scan_result=scan_result,
            event_type=event_type,
            guardrail_status="success",
            tracing_detail=GuardrailTracingDetail(guardrail_action="ALLOWED"),
        )
        return inputs

    def _scan_event_type(self, request_data: Mapping[str, object], input_type: str) -> GuardrailEventHooks:
        """The hook this scan ran under. ``apply_guardrail`` is reached from pre_call and
        during_call alike with no signal of which, so the configured mode decides."""
        is_mcp: Final = isinstance(request_data.get("mcp_tool_call_metadata"), Mapping)
        if input_type == "response":
            return GuardrailEventHooks.post_mcp_call if is_mcp else GuardrailEventHooks.post_call
        configured: Final[Sequence[object]] = (
            self.event_hook
            if isinstance(self.event_hook, list)
            else ()
            if self.event_hook is None
            else (self.event_hook,)
        )
        hooks: Final = frozenset(str(h) for h in configured)
        if is_mcp:
            only_during: Final = "during_mcp_call" in hooks and "pre_mcp_call" not in hooks
            return GuardrailEventHooks.during_mcp_call if only_during else GuardrailEventHooks.pre_mcp_call
        return (
            GuardrailEventHooks.during_call
            if "during_call" in hooks and "pre_call" not in hooks
            else GuardrailEventHooks.pre_call
        )

    def _record_block(
        self,
        request_data: dict,
        result: dict,
        event_type: GuardrailEventHooks,
        action: Literal["BLOCKED", "FLAGGED"] = "BLOCKED",
    ) -> None:
        self._record_scan_outcome(
            request_data=request_data,
            scan_result=result,
            event_type=event_type,
            guardrail_status="guardrail_intervened" if action == "BLOCKED" else "guardrail_flagged",
            tracing_detail=GuardrailTracingDetail(
                guardrail_action=action,
                violation_categories=list(self._violated_policy_names(result)),
            ),
        )
        verbose_proxy_logger.info(
            "XecGuard %s: %s",
            action.lower(),
            json.dumps(self._request_context(request_data, result), ensure_ascii=False, default=str),
        )

    async def async_logging_hook(
        self,
        kwargs: dict,
        result: Any,
        call_type: str,
    ) -> tuple[dict, Any]:
        """Observe-only scan for logging_only mode.

        Never blocks, never raises - all errors are swallowed. Records a
        StandardLoggingGuardrailInformation entry so the scan decision
        reaches downstream loggers (Langfuse, DataDog, etc.).
        """
        if self._already_recorded(kwargs):
            return kwargs, result

        # Same key targeting as apply_guardrail. logging_only reaches the guardrail
        # through this hook rather than apply_guardrail, so the gate is repeated here;
        # without it an excluded key's content would still be sent to XecGuard.
        if not self._key_is_targeted(self._key_context(kwargs)):
            return kwargs, result

        try:
            scratch: Final = {**kwargs, "metadata": {}}
            scans: Final = await self._logged_call_scans(kwargs, result, call_type)
            await asyncio.gather(
                *(self._record_logged_scan(scratch, messages, scan_type) for messages, scan_type in scans)
            )
            recorded: Final = scratch["metadata"].get("standard_logging_guardrail_information") or []
            standard_logging_object: Final = kwargs.get("standard_logging_object")
            if recorded and isinstance(standard_logging_object, dict):
                existing: Final = standard_logging_object.get("guardrail_information")
                standard_logging_object["guardrail_information"] = [
                    *(existing if isinstance(existing, list) else ()),
                    *recorded,
                ]
        except Exception as exc:
            verbose_proxy_logger.debug(
                "XecGuard logging_only swallowed exception: %s",
                str(exc),
            )
        return kwargs, result

    def _already_recorded(self, kwargs: Mapping[str, Any]) -> bool:
        litellm_params: Final = kwargs.get("litellm_params")
        metadata: Final = litellm_params.get("metadata") if isinstance(litellm_params, dict) else None
        recorded: Final = metadata.get("standard_logging_guardrail_information") if isinstance(metadata, dict) else None
        return isinstance(recorded, list) and any(
            isinstance(entry, dict) and entry.get("guardrail_name") == (self.guardrail_name or "xecguard")
            for entry in recorded
        )

    async def _record_logged_scan(self, scratch: dict, messages: list[dict], scan_type: str) -> None:
        """One scan of a logged call, recorded the way every other mode records its scans. Observe
        only: a failed scan records nothing and a violation is flagged, never raised."""
        started: Final = datetime.now()
        scan_result: Final = await self._call_scan(
            messages=messages, scan_type=scan_type, request_data=scratch, suppress_errors=True
        )
        if scan_result is None:
            return
        violated: Final = scan_result.get("decision") == "UNSAFE"
        self._record_scan_outcome(
            request_data=scratch,
            scan_result=scan_result,
            event_type=GuardrailEventHooks.logging_only,
            guardrail_status="guardrail_intervened" if violated else "success",
            tracing_detail=GuardrailTracingDetail(
                guardrail_action="BLOCKED" if violated else "ALLOWED",
                violation_categories=list(self._violated_policy_names(scan_result)),
                detection_method=_LOGGED_SCAN_LABEL[scan_type],
            ),
            start_time=started.timestamp(),
            end_time=datetime.now().timestamp(),
        )

    @staticmethod
    def _logged_reply_translation(result: Any, call_type: str) -> "BaseTranslation | None":
        """litellm's translation that reads this logged result. The logged result is not always in
        the shape ``call_type`` implies (``/v1/messages`` on an OpenAI model logs a chat response),
        so it is picked by what the result is, and ``call_type`` only decides when that is unclear."""
        from litellm.llms import get_guardrail_translation_mapping
        from litellm.types.llms.openai import ResponseCompletedEvent, ResponsesAPIResponse
        from litellm.types.utils import CallTypes, ModelResponse

        by_result: Final = (
            CallTypes.aresponses
            if isinstance(result, (ResponsesAPIResponse, ResponseCompletedEvent))
            else CallTypes.acompletion
            if isinstance(result, ModelResponse)
            else None
        )
        try:
            return get_guardrail_translation_mapping(by_result or CallTypes(call_type))()
        except ValueError:
            return None

    @staticmethod
    def _logged_original_request(kwargs: Mapping[str, Any]) -> dict | None:
        """The body the client sent, which the logging payload keeps under ``proxy_server_request``.
        Unlike the logged ``messages`` it is in the endpoint's own shape, system prompt, tool turns
        and all, so litellm's translation for that endpoint reads it the way it does on the way in."""
        litellm_params: Final = kwargs.get("litellm_params")
        proxy_request: Final = litellm_params.get("proxy_server_request") if isinstance(litellm_params, dict) else None
        body: Final = proxy_request.get("body") if isinstance(proxy_request, dict) else None
        return body if isinstance(body, dict) and body else None

    @staticmethod
    def _logged_system_prompt(kwargs: Mapping[str, Any]) -> str | None:
        """The system prompt of a logged request when only its chat ``messages`` are left. It is not
        part of them: a Responses request logs it as ``instructions`` and a Messages request as
        ``system``, each under ``optional_params`` or at the top level."""
        sources: Final = (kwargs, kwargs.get("optional_params") or {})
        for name in ("instructions", "system"):
            for source in sources:
                value = source.get(name)
                if isinstance(value, str) and value:
                    return value
        return None

    def _logged_request_turns(self, kwargs: dict, call_type: str) -> list[dict]:
        """The request conversation of a logged call: the original body read through litellm's
        translation for its endpoint, else the logged chat messages with their system prompt back."""
        from litellm.llms import get_guardrail_translation_mapping
        from litellm.types.utils import CallTypes

        original: Final = self._logged_original_request(kwargs)
        if original is not None:
            try:
                translation: Final = get_guardrail_translation_mapping(CallTypes(call_type))()
                from_body: Final = self._request_turns(translation, original)
            except Exception as exc:
                verbose_proxy_logger.debug("XecGuard logging_only: could not read the original request (%s)", exc)
                from_body = []
            if from_body:
                return from_body
        logged_request: Final = {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "input": kwargs.get("input"),
            **(kwargs.get("optional_params") or {}),
        }
        request_turns: Final = self._request_turns(
            get_guardrail_translation_mapping(CallTypes.acompletion)(), logged_request
        )
        system_prompt: Final = self._logged_system_prompt(kwargs)
        if system_prompt and not any(t.get("role") == "system" for t in request_turns):
            return [{"role": "system", "content": system_prompt}, *request_turns]
        return request_turns

    async def _logged_call_scans(
        self, kwargs: dict, result: Any, call_type: str
    ) -> list[tuple[list[dict], Literal["input", "response"]]]:
        """The scans a logged call gets. A blocking mode sees the request in ``pre_call`` and the
        reply in ``post_call``, so a logged call gets both: the request on its own as ``input``
        (XecGuard judges the user side of the conversation only then), and the request with its
        reply as ``response``. Both are read by litellm's translations."""
        from litellm.types.llms.openai import ResponseCompletedEvent

        turns: Final = self._logged_request_turns(kwargs, call_type)
        translation: Final = self._logged_reply_translation(result, call_type)
        capture: Final = _TextCapture()
        reply: Final = getattr(result, "response", result) if isinstance(result, ResponseCompletedEvent) else result
        if translation is not None and reply is not None:
            try:
                await translation.process_output_response(
                    response=copy.deepcopy(reply), guardrail_to_apply=capture, request_data={}
                )
            except Exception as exc:
                verbose_proxy_logger.debug("XecGuard logging_only: could not read the reply: %s", exc)
        request_scan: Final = [(turns, "input")] if turns and turns[-1].get("role") in ("user", "tool") else []
        if capture.text or capture.tool_calls:
            assistant: Final = self._assistant_turn(capture.text, capture.tool_calls)
            return [*request_scan, (self._reply_conversation(turns, assistant), "response")]
        return request_scan

    # ------------------------------------------------------------------
    # Streaming (post_call): self-contained, does not use apply_guardrail
    # ------------------------------------------------------------------

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        """Streaming-output scan, mode chosen by ``stream_mode``.

        - ``window``: rolling-window scan every ``stream_chunk`` bytes
          (``stream_context`` overlap bytes + new text); UNSAFE cuts the
          stream mid-flight. ``stream_first`` (default False) holds the
          newest events back until a scan has covered them; True releases each
          delta before it's covered by a scan (NeMo's behaviour);
          scan errors fail open (never block on a XecGuard outage).
        - ``buffer``: hold every event, scan the assembled text once, then
          replay the original events. Zero leak, no token-by-token feel. Scan
          errors respect ``block_on_error`` like every other non-streaming scan.
        - ``audit``: same cadence as ``window`` but never acts on UNSAFE -
          for observing what the window scan would have done.

        What goes down the wire is whatever the route's own events are (chat
        chunks, Responses events). A block is built by the route's litellm
        translation handler, so it ends the stream in that route's format.
        """
        if not self._key_is_targeted(self._key_context(request_data)):
            async for chunk in response:
                yield chunk
            return
        route: Final = getattr(user_api_key_dict, "request_route", None)
        if self.stream_mode == "buffer":
            async for chunk in self._buffer_all_and_scan(response=response, request_data=request_data, route=route):
                yield chunk
            return
        async for chunk in self._window_scan_stream(
            response=response,
            request_data=request_data,
            route=route,
            audit_only=self.stream_mode == "audit",
        ):
            yield chunk

    @staticmethod
    def _translation_for_route(route: str | None, sample: object = None) -> "BaseTranslation | None":
        """litellm's own request/response translation for the endpoint this call came in on,
        or - with no route to go by - for the response shape ``sample`` has, the way the
        unified guardrail resolves it."""
        from litellm.cost_calculator import (
            _infer_call_type,  # pyright: ignore[reportPrivateUsage]  # same resolver the unified guardrail uses
        )
        from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
        from litellm.llms import get_guardrail_translation_mapping
        from litellm.types.utils import CallTypes

        call_types: Final = get_call_types_for_route(route) if route else None
        call_type: Final = call_types[0] if call_types else _infer_call_type(None, sample)
        if call_type is None:
            return None
        try:
            return get_guardrail_translation_mapping(CallTypes(call_type))()
        except ValueError:
            return None

    @classmethod
    async def _stream_reply(cls, translation: "BaseTranslation | None", chunks: Sequence[Any]) -> _TextCapture:
        """The reply streamed so far, as litellm's translation for the route reads it.

        The translation hands the text to ``apply_guardrail`` as ``inputs.texts`` and the tool
        calls as ``inputs.tool_calls``; a capturing guardrail takes them from there. Chunks are
        copied first because some translations rewrite the list they are given.
        """
        capture: Final = _TextCapture()
        if translation is None or not chunks:
            return capture
        await translation.process_output_streaming_response(
            responses_so_far=copy.deepcopy(list(chunks)), guardrail_to_apply=capture, request_data={}
        )
        return capture

    @classmethod
    async def _stream_text(cls, translation: "BaseTranslation | None", chunks: Sequence[Any]) -> str:
        return (await cls._stream_reply(translation, chunks)).text

    @staticmethod
    def _text_after(scanned: str, read: str) -> str:
        """The part of ``read`` that has not been scanned yet. A translation reads the events it is
        handed, but the one that ends a Responses stream (``response.completed``) carries the whole
        reply, so a slice that holds it reads back as the full text and not as the tail."""
        return read[len(scanned) :] if scanned and read.startswith(scanned) else read

    @staticmethod
    def _assistant_turn(text: str, tool_calls: Sequence[Any]) -> dict:
        calls: Final = {"tool_calls": list(tool_calls)} if tool_calls else {}
        return {"role": "assistant", "content": text or None, **calls}

    def _block_events(
        self,
        scan_result: dict,
        request_data: dict,
        route: str | None,
        released: Sequence[Any],
    ) -> list[Any]:
        """The events that end a blocked stream, in the route's own format.

        ``released`` is what the client already has. With nothing released the
        block is a whole standalone response; otherwise it continues the open one.
        """
        from litellm.exceptions import ModifyResponseException
        from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
        from litellm.llms import get_guardrail_translation_mapping
        from litellm.types.utils import CallTypes

        message: Final = f"[Blocked by XecGuard] {self._block_message(scan_result)}"
        call_types: Final = get_call_types_for_route(route) if route else None
        if call_types:
            try:
                handler: Final = get_guardrail_translation_mapping(CallTypes(call_types[0].value))()
                exc: Final = ModifyResponseException(
                    message=message,
                    model=str(request_data.get("model") or ""),
                    request_data=request_data,
                    guardrail_name=self.guardrail_name,
                )
                events: Final = handler.build_block_sse_chunks(
                    exc, stream_started=bool(released), responses_so_far=list(released)
                )
                if events:
                    return list(events)
            except Exception as exc_info:
                verbose_proxy_logger.debug("XecGuard: route block builder unavailable (%s)", exc_info)
        block_chunk: Final = self._build_block_stream_chunk(scan_result)
        return self._as_generate_content_events(request_data, block_chunk) or [block_chunk]

    @staticmethod
    def _as_generate_content_events(request_data: Mapping[str, Any], chunk: ModelResponseStream) -> list[bytes]:
        """The block chunk as a Gemini SSE event, through litellm's own OpenAI -> Gemini stream
        conversion, for a ``generateContent`` request whose route has no block builder."""
        if not request_data.get("contents"):
            return []
        from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter, GoogleGenAIStreamWrapper

        try:
            event: Final = GoogleGenAIAdapter().translate_streaming_completion_to_generate_content(
                chunk, GoogleGenAIStreamWrapper(iter(()))
            )
        except Exception as exc:
            verbose_proxy_logger.warning("XecGuard: could not convert block to a generateContent event (%s)", exc)
            return []
        return [f"data: {json.dumps(event)}\n\n".encode()] if event else []

    def _build_block_stream_chunk(self, scan_result: dict) -> ModelResponseStream:
        notice: Final = f"\n\n[Blocked by XecGuard] {self._block_message(scan_result)}"
        return ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(role="assistant", content=notice),
                    finish_reason="content_filter",
                )
            ]
        )

    @staticmethod
    def _unavailable_result() -> dict:
        """What a stream scan that could not reach XecGuard stands for under ``block_on_error``.
        A started stream cannot answer with an HTTP error, so it ends with a block like any other."""
        return {
            "decision": "UNSAFE",
            "trace_id": "",
            "xecguard_result": [
                {
                    "violated_policy_name": "XecGuardUnavailable",
                    "rationale": "XecGuard API unreachable (block_on_error=True)",
                }
            ],
        }

    async def _stream_scan(self, messages: list[dict], request_data: dict, audit_only: bool) -> dict | None:
        result: Final = await self._call_scan(
            messages=messages, scan_type="response", request_data=request_data, suppress_errors=True
        )
        if result is None and self.block_on_error and not audit_only:
            return self._unavailable_result()
        return result

    @staticmethod
    def _raw_size(chunk: Any) -> int:
        """Rough size of the reply text one stream event carries, used only to decide when a
        window is due; the exact text is read through litellm's translation at scan time."""
        if isinstance(chunk, (bytes, str)):
            raw = chunk.decode("utf-8", errors="ignore") if isinstance(chunk, bytes) else chunk
            return sum(len(m) for m in _SSE_TEXT_FIELD.findall(raw))
        for holder in (getattr(chunk, "choices", None), None):
            try:
                content = holder[0].delta.content if holder else getattr(chunk, "delta", None)
            except (AttributeError, IndexError, TypeError):
                content = None
            if isinstance(content, str):
                return len(content.encode("utf-8"))
        return 0

    @staticmethod
    def _reply_conversation(turns: Sequence[dict], reply: str | dict) -> list[dict]:
        """The request turns followed by the reply. XecGuard rejects a scan with no user turn,
        so a route that exposes no request conversation (Gemini) gets a placeholder one."""
        has_user_turn: Final = any(t.get("role") == "user" and t.get("content") for t in turns)
        lead: Final = list(turns) if has_user_turn else [*turns, {"role": "user", "content": _NO_REQUEST_TEXT}]
        return [*lead, reply if isinstance(reply, dict) else {"role": "assistant", "content": reply}]

    def _held_tail_start(self, held: Sequence[Any]) -> int:
        """Index of the first withheld event to keep back after a SAFE window: the trailing
        events carrying the last ``stream_context`` bytes, which the next window scans together
        with what follows, so an entity cut by the window edge is never half released."""
        covered = 0
        for index in range(len(held) - 1, -1, -1):
            if covered >= self.stream_context:
                return index + 1
            covered += self._raw_size(held[index])
        return 0

    async def _window_scan_stream(
        self,
        response: Any,
        request_data: dict,
        route: str | None,
        audit_only: bool,
    ) -> AsyncGenerator[Any, None]:
        """Once ``stream_chunk`` bytes have arrived since the last scan, the events since then are
        read through litellm's translation for the route; the scanned window is that new text,
        preceded by the last ``stream_context`` bytes of the text already scanned."""
        translations: Final[list[BaseTranslation | None]] = []  # mutable-ok: resolved from the first event
        chunks: Final[list[Any]] = []  # mutable-ok: every event pulled so far; each window reads only the new ones
        scanned: Final[list[int]] = [0]  # mutable-ok: how many events the last window covered
        previous: Final[list[str]] = [""]  # mutable-ok: reply text the last window covered
        pending: Final[list[int]] = [0]  # mutable-ok: rough bytes received since the last window
        held: Final[list[Any]] = []  # mutable-ok: events withheld until their window is scanned (stream_first off)
        released: Final[list[Any]] = []  # mutable-ok: events the client already holds
        flagged: Final[list[dict]] = []  # mutable-ok: audit hits collected while the stream runs
        outcome: Final[list[dict]] = []  # mutable-ok: last scan that came back, for the end-of-stream record

        def note(result: dict | None) -> dict | None:
            if result is not None and result.get("decision") == "SAFE":
                outcome[:] = [result]
            if result is not None and result.get("decision") == "UNSAFE" and audit_only:
                flagged.append(result)
            return result

        async def scan_tool_calls() -> dict | None:
            translation: Final = translations[0] if translations else None
            reply: Final = await self._stream_reply(translation, chunks)
            if not reply.tool_calls:
                return None
            messages: Final = self._reply_conversation(
                self._request_turns(translation, request_data), self._assistant_turn("", reply.tool_calls)
            )
            return note(await self._stream_scan(messages, request_data, audit_only))

        async def scan_window() -> dict | None:
            translation: Final = translations[0] if translations else None
            read: Final = await self._stream_text(translation, chunks[scanned[0] :])
            new_text: Final = self._text_after(previous[0], read)
            scanned[0] = len(chunks)
            pending[0] = 0
            if not new_text:
                return None
            overlap: Final = (
                previous[0].encode("utf-8")[-self.stream_context :].decode("utf-8", errors="ignore")
                if self.stream_context
                else ""
            )
            window: Final = overlap + new_text
            previous[0] += new_text
            messages: Final = self._reply_conversation(self._request_turns(translation, request_data), window)
            return note(await self._stream_scan(messages, request_data, audit_only))

        async for chunk in response:
            if not chunks:
                translations.append(self._translation_for_route(route, chunk))
            chunks.append(chunk)
            if self.stream_first:
                released.append(chunk)
                yield chunk
            else:
                held.append(chunk)
            pending[0] += self._raw_size(chunk)
            if pending[0] < self.stream_chunk:
                continue
            result = await scan_window()
            if result is not None and result.get("decision") == "UNSAFE" and not audit_only:
                self._record_block(request_data, result, GuardrailEventHooks.post_call)
                for event in self._block_events(result, request_data, route, released):
                    yield event
                return
            tail_start: Final = self._held_tail_start(held)
            safe_events: Final = held[:tail_start]
            released.extend(safe_events)
            for held_chunk in safe_events:
                yield held_chunk
            del held[:tail_start]

        result = await scan_window() if len(chunks) > scanned[0] else None
        scan_blocked: Final = result is not None and result.get("decision") == "UNSAFE"
        tool_result: Final = None if scan_blocked else await scan_tool_calls()
        blocking = next((r for r in (result, tool_result) if r is not None and r.get("decision") == "UNSAFE"), None)
        if blocking is None:
            grounding_result = await self._stream_grounding(
                request_data, translations[0] if translations else None, previous[0], audit_only
            )
            if grounding_result is not None and audit_only:
                flagged.append(grounding_result)
            blocking = grounding_result
        if blocking is not None and not audit_only:
            self._record_block(request_data, blocking, GuardrailEventHooks.post_call)
            for event in self._block_events(blocking, request_data, route, released):
                yield event
            return
        self._record_stream_end(request_data, flagged, outcome[0] if outcome else None)
        for held_chunk in held:
            yield held_chunk

    def _record_stream_end(
        self,
        request_data: dict[str, object],
        flagged: Sequence[dict[str, object]],
        last_result: dict[str, object] | None,
    ) -> None:
        """A stream that reached its end unblocked: flagged when audit mode saw an UNSAFE
        window, otherwise allowed - provided at least one scan actually came back."""
        if flagged:
            self._record_block(request_data, flagged[0], GuardrailEventHooks.post_call, action="FLAGGED")
            return
        if last_result is None:
            return
        self._record_scan_outcome(
            request_data=request_data,
            scan_result=last_result,
            event_type=GuardrailEventHooks.post_call,
            guardrail_status="success",
            tracing_detail=GuardrailTracingDetail(guardrail_action="ALLOWED"),
        )

    async def _buffer_all_and_scan(
        self,
        response: Any,
        request_data: dict,
        route: str | None,
    ) -> AsyncGenerator[Any, None]:
        all_chunks: Final[list[Any]] = []
        async for chunk in response:
            all_chunks.append(chunk)

        translation: Final = self._translation_for_route(route, all_chunks[0] if all_chunks else None)
        reply: Final = await self._stream_reply(translation, all_chunks)
        if reply.text or reply.tool_calls:
            xg_messages: Final = self._reply_conversation(
                self._request_turns(translation, request_data), self._assistant_turn(reply.text, reply.tool_calls)
            )
            scan_result: Final = await self._stream_scan(xg_messages, request_data, audit_only=False)
            if scan_result is not None and scan_result.get("decision") == "UNSAFE":
                self._record_block(request_data, scan_result, GuardrailEventHooks.post_call)
                for event in self._block_events(scan_result, request_data, route, []):
                    yield event
                return
            grounding_result = await self._stream_grounding(request_data, translation, reply.text)
            if grounding_result is not None:
                self._record_block(request_data, grounding_result, GuardrailEventHooks.post_call)
                for event in self._block_events(grounding_result, request_data, route, []):
                    yield event
                return
            self._record_stream_end(request_data, (), scan_result)

        for chunk in all_chunks:
            yield chunk

    # ------------------------------------------------------------------
    # Caller context (scan payload ``meta``) - SIEM correlation only
    # ------------------------------------------------------------------

    def _build_scan_meta(self, context: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        """Assemble the scan payload's ``meta`` object, or None to omit it.

        ``virtualkey`` is the identity this guardrail filtered on and ``data`` is
        the calling key's proxy-injected attributes plus its own metadata as set on
        the Virtual Keys page. Neither participates in detection - XecGuard forwards
        them to the SIEM so a scan can be traced back to the virtual key that caused
        it.

        ``meta`` is optional in the contract, so anything that cannot be made to
        satisfy it is left out instead of turning a scan into a 400.
        """
        if not self.send_meta:
            return None

        virtualkey: Final[str | Mapping[str, str] | None] = (
            self._scan_meta_virtualkey_object(context)
            if self.meta_identity_format == "object"
            else self._scan_meta_virtualkey(context)
        )
        if not virtualkey:
            verbose_proxy_logger.debug(
                "XecGuard: omitting scan meta - the calling key has no alias or hash matching the "
                "backend's virtualkey pattern (give the key a key_alias to enable SIEM correlation)"
            )
            return None

        meta: Final[dict[str, Any]] = {"virtualkey": virtualkey}  # mutable-ok: the JSON object being assembled
        data: Final = self._build_scan_meta_data(context, virtualkey=virtualkey)
        if data:
            meta["data"] = data
        return meta

    def _scan_meta_virtualkey(self, context: Mapping[str, Any] | None) -> str | None:
        """The key identity to report, or None when there is no usable one.

        Alias first: that is what an operator types into ``apply_to_aliases`` /
        ``except_aliases``, so the value in the SIEM matches the value in the
        guardrail config. The hashed token is the fallback for keys created
        without an alias; a master-key call has neither. Note that the token
        hash only satisfies the backend's pattern when it happens to start with
        a hex letter - aliasless keys are not reliably correlatable.
        """
        for candidate in self._calling_key_identity(context):
            if isinstance(candidate, str) and _META_NAME_PATTERN.match(candidate):
                return candidate
        return None

    def _scan_meta_virtualkey_object(self, context: Mapping[str, Any] | None) -> Mapping[str, str] | None:
        """The object form of ``virtualkey``: ``{"alias": ..., "key_id": ...}``.

        Either member may be absent - a key created without an alias has only an
        id, and a master-key call has neither (in which case ``meta`` is omitted).
        Unlike the string form this does not require the alias to satisfy the
        backend's identifier pattern: the pattern exists because a bare string
        becomes a SIEM field *value* directly, whereas here each member is
        sanitized the same way ``meta.data`` values are. That makes keys whose
        alias contains spaces or CJK correlatable, which the string form cannot do.
        """
        alias, key_hash = self._calling_key_identity(context)
        obj: Final[dict[str, str]] = {}  # mutable-ok: the JSON object being assembled
        for name, raw in (("alias", alias), ("key_id", key_hash)):
            value = self._coerce_meta_value(raw)
            if value is not None:
                obj[name] = value
        return obj or None

    @staticmethod
    def _calling_key_metadata(context: Mapping[str, Any] | None) -> Mapping[object, Any]:
        """The calling virtual key's own metadata, as injected by the proxy.

        This is the JSON an admin typed into the key's Metadata box on the
        Virtual Keys page (minus the callback-credential slots, which the proxy
        strips before injecting). Team metadata is deliberately not merged in:
        ``meta.data`` is meant to describe the key that made the call.

        The key type is ``object``, not ``str``: nothing between the database and
        here validates it, and the caller drops a non-str key rather than letting
        it reach ``re.match`` and raise. Narrowing this to ``str`` would make that
        guard look redundant to a type checker and invite its removal.
        """
        if not isinstance(context, dict):
            return {}  # mutable-ok: "this key has no metadata"; the caller only reads it
        for meta_key in ("metadata", "litellm_metadata"):
            md = context.get(meta_key)
            if isinstance(md, dict):
                key_metadata = md.get(_METADATA_KEY_METADATA_FIELD)
                if isinstance(key_metadata, dict):
                    return key_metadata
        return {}  # mutable-ok: same empty result, no metadata field was injected

    @classmethod
    def _auto_meta_data_items(cls, context: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
        """The proxy-injected virtual-key attributes, in ``_META_AUTO_DATA_FIELDS``
        order regardless of how the proxy ordered its metadata dict.

        Absent and null fields are skipped, so a key with no team contributes no
        ``team_id`` rather than an empty one - a SIEM query for "scans with no
        team" then means it, instead of matching every key.
        """
        injected: Final[dict[str, Any]] = {}  # mutable-ok: accumulator keyed by meta.data name
        if isinstance(context, dict):
            for meta_key in ("metadata", "litellm_metadata"):
                md = context.get(meta_key)
                if not isinstance(md, dict):
                    continue
                for name, source_field in _META_AUTO_DATA_FIELDS:
                    if name not in injected and md.get(source_field) is not None:
                        injected[name] = md[source_field]
        return tuple((name, injected[name]) for name, _ in _META_AUTO_DATA_FIELDS if name in injected)

    def _custom_header_meta_items(self, context: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
        """Caller-supplied request headers named in ``self.custom_header_meta_fields``,
        read from ``proxy_server_request.headers``.

        The proxy lower-cases header names before storing them (see
        ``clean_headers`` in ``litellm_pre_call_utils.py``), so lookup here
        lower-cases the configured header name to match. The meta.data field
        name is derived from the configured header name (lower-cased, every
        run of characters ``_META_NAME_PATTERN`` rejects collapsed to a single
        underscore) rather than configured separately, so adding a new header
        is a one-line config change - no separate field-name mapping to keep
        in sync.
        """
        if not self.custom_header_meta_fields:
            return ()
        if not isinstance(context, dict):
            return ()
        proxy_server_request: Final = context.get("proxy_server_request")
        if not isinstance(proxy_server_request, dict):
            return ()
        headers: Final = proxy_server_request.get("headers")
        if not isinstance(headers, dict):
            return ()
        items: Final[list[tuple[str, Any]]] = []  # mutable-ok: built once per call, returned as a tuple
        for header_name in self.custom_header_meta_fields:
            value = headers.get(header_name.lower())
            if value is None:
                continue
            field_name = _META_INVALID_NAME_CHARS.sub("_", header_name.lower()).strip("_")
            if field_name:
                items.append((field_name, value))
        return tuple(items)

    def _build_scan_meta_data(
        self, context: Mapping[str, Any] | None, virtualkey: str | Mapping[str, str]
    ) -> Mapping[str, str]:
        """Coerce the calling key's attributes and metadata into ``meta.data``.

        Three sources, in this order: the attributes the proxy injects about the
        calling key (identity, tenancy, budget), then caller-supplied request
        headers named in ``_META_CUSTOM_HEADER_FIELDS``, then the free-form
        metadata an admin typed into the key's Metadata box. Earlier sources win
        a name collision, so a request header cannot shadow ``key_id`` and an
        admin's own metadata cannot shadow either of the first two.

        Fields are kept while they satisfy the contract: a name matching the
        backend's pattern, a flat scalar value, at most 32 fields, and a
        serialized ``meta`` within the 4096-byte cap. Oversize fields are skipped
        rather than ending the scan, so a later small field still gets through.
        Dropped names are logged without their values - both sources can hold
        sensitive strings.
        """
        source: Final = self._calling_key_metadata(context)
        data: Final[dict[str, str]] = {}  # mutable-ok: accumulator, re-measured as it grows
        # Re-measured against the real payload shape each time, so the cap holds
        # regardless of how long the virtualkey and the field names are.
        probe: Final[dict[str, Any]] = {"virtualkey": virtualkey, "data": data}  # mutable-ok: views `data`
        dropped: Final[list[str]] = []  # mutable-ok: skipped field names, for one debug line

        for name, raw_value in (
            *self._auto_meta_data_items(context),
            *self._custom_header_meta_items(context),
            *source.items(),
        ):
            if self.meta_data_fields:
                if name not in self.meta_data_fields:
                    continue
            elif name in _META_CONTROL_DATA_FIELDS:
                # proxy config rather than caller identity - opt in by name
                continue
            if name in _META_EXCLUDED_DATA_FIELDS:
                continue
            if name in data:  # a proxy-injected attribute already claimed this name
                dropped.append(str(name))
                continue
            if not isinstance(name, str) or not _META_NAME_PATTERN.match(name):
                dropped.append(str(name))
                continue
            if len(data) >= _META_MAX_DATA_FIELDS:
                dropped.append(name)
                continue
            value = self._coerce_meta_value(raw_value)
            if value is None:
                dropped.append(name)
                continue
            data[name] = value
            if len(json.dumps(probe, ensure_ascii=False).encode("utf-8")) > _META_MAX_SERIALIZED_BYTES:
                del data[name]
                dropped.append(name)

        if dropped:
            verbose_proxy_logger.debug(
                "XecGuard: scan meta.data dropped %d field(s) (names only): %s",
                len(dropped),
                dropped,
            )
        return data

    @staticmethod
    def _coerce_meta_value(value: object) -> str | None:
        """Coerce one key-metadata value to the contract, or None to drop it.

        Scalars are stringified so an admin writing ``{"tier": 3}`` still gets a
        usable ``ctx_tier``. Nested objects and lists have no flat representation
        a SIEM index field can hold, so they are dropped.
        """
        if isinstance(value, bool):
            return _sanitized_meta_text("true" if value else "false")
        if isinstance(value, str):
            return _sanitized_meta_text(value)
        if isinstance(value, (int, float)):
            return _sanitized_meta_text(str(value))
        return None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _resolve_policy_names(self, messages: list[dict]) -> list[str]:
        base: Final = list(self.policy_names or ())
        custom_applies: Final = self._last_user_text_length(messages) > self.custom_policy_min_chars
        selected: Final = [*base, *(self.custom_policy_names if custom_applies else ())]
        return list(dict.fromkeys(selected)) or list(_DEFAULT_POLICIES)

    @classmethod
    def _last_user_text_length(cls, messages: list[dict]) -> int:
        return len(cls._extract_last_text_by_role(messages, "user") or "")

    async def _call_scan(
        self,
        messages: list[dict],
        scan_type: str,
        request_data: Mapping[str, Any] | None = None,
        suppress_errors: bool = False,
    ) -> dict | None:
        payload: Final[dict[str, object]] = {
            "model": self.xecguard_model,
            "scan_type": scan_type,
            "messages": messages,
            "policy_names": self._resolve_policy_names(messages),
        }
        meta: Final = self._build_scan_meta(self._key_context(request_data))
        if meta is not None:
            payload["meta"] = meta
        return await self._post(
            path=_SCAN_ENDPOINT,
            payload=payload,
            suppress_errors=suppress_errors,
        )

    async def _stream_grounding(
        self, request_data: dict, translation: "BaseTranslation | None", reply_text: str, audit_only: bool = False
    ) -> dict | None:
        """Grounding check of a finished stream's reply, the way a plain response gets one.
        Only a reply with text can be grounded, so a tool-call-only stream has nothing to check."""
        documents: Final = self._extract_grounding_documents(request_data)
        if not documents or not reply_text:
            return None
        messages: Final = self._reply_conversation(self._request_turns(translation, request_data), reply_text)
        result: Final = await self._call_grounding(messages=messages, documents=documents, suppress_errors=True)
        if result is None and self.block_on_error and not audit_only:
            return self._unavailable_result()
        return result if result is not None and result.get("decision") == "UNSAFE" else None

    async def _call_grounding(
        self,
        messages: list[dict],
        documents: list[dict],
        suppress_errors: bool = False,
    ) -> dict | None:
        prompt: Final = self._extract_last_text_by_role(messages, "user")
        response_text: Final = self._extract_last_text_by_role(messages, "assistant")
        if prompt is None or prompt == _NO_REQUEST_TEXT or response_text is None:
            return None
        payload: Final = {
            "model": self.xecguard_model,
            "prompt": prompt,
            "response": response_text,
            "documents": documents,
            "strictness": self.grounding_strictness,
        }
        return await self._post(path=_GROUNDING_ENDPOINT, payload=payload, suppress_errors=suppress_errors)

    async def _post(
        self,
        path: str,
        payload: dict,
        suppress_errors: bool = False,
    ) -> dict | None:
        endpoint: Final = f"{self.api_base}{path}"
        verbose_proxy_logger.debug(
            "XecGuard: POST %s payload_keys=%s",
            endpoint,
            list(payload.keys()),
        )
        try:
            started: Final = time.time()
            response: Final = await self.async_handler.post(
                url=endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()
            body: Final = response.json()
            if not isinstance(body, dict):
                return body
            timed: Final = _TimedResult(body)
            timed.started, timed.ended = started, time.time()
            return timed
        except Exception as exc:
            verbose_proxy_logger.error("XecGuard API error: %s", str(exc))
            if suppress_errors:
                return None
            if self.block_on_error:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": (f"XecGuard API unreachable (block_on_error=True): {exc}"),
                        "guardrail_name": self.guardrail_name or "xecguard",
                    },
                ) from exc
            return None

    # ------------------------------------------------------------------
    # Message-assembly helpers (respect the full-history requirement)
    # ------------------------------------------------------------------

    _ROUTING_TOKENS: Final = ("[xecguard:tool_call]", "[xecguard:tool_result]")

    @classmethod
    def _strip_routing_tokens(cls, text: str) -> str:
        return reduce(lambda acc, token: acc.replace(token, ""), cls._ROUTING_TOKENS, text).strip()

    @classmethod
    def _mcp_tool_scan(
        cls, request_data: Mapping[str, Any] | None, inputs: Any, input_type: str
    ) -> tuple[tuple[dict[str, Any], ...], Literal["input", "response"]] | None:
        """Rebuild the MCP tool call as the OpenAI wire shape XecGuard routes on.

        XecGuard picks ToolUse_Guard from the shape: an assistant turn carrying
        ``tool_calls`` with null content, optionally followed by a ``role: "tool"``
        result. Without it the tool traffic is judged as ordinary text by whatever
        other policies were requested, so the three tool rules never run.

        The two seams are judged from opposite directions, and only the first one can
        prevent anything: on ``pre_mcp_call`` the tool has not run, so the call alone is
        judged ``response``-side and blocking stops the action. On ``post_mcp_call`` the
        side effect already happened; the pair is judged ``input``-side, which pairs the
        call back in and covers both turns in one scan.

        ``None`` means this is not the MCP seam and the caller falls back to history.
        """
        meta: Final = (request_data or {}).get("mcp_tool_call_metadata")
        if not isinstance(meta, Mapping):
            return None
        name: Final = meta.get("namespaced_tool_name") or meta.get("name")
        if not isinstance(name, str) or not name:
            return None
        call_id: Final = str((request_data or {}).get("litellm_call_id") or "xecguard_mcp_tool_call")
        arguments: Final = json.dumps(meta.get("arguments") or {}, ensure_ascii=False, sort_keys=True)
        call_message: Final = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
        }
        if input_type == "request":
            return (call_message,), "response"
        result_text: Final = cls._join_input_texts(inputs)
        if not result_text:
            return None
        result_message: Final = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": cls._strip_routing_tokens(result_text),
        }
        return (call_message, result_message), "input"

    @staticmethod
    def _request_route(request_data: Mapping[str, Any]) -> str | None:
        route: Final = _proxy_metadata_field(request_data, "user_api_key_request_route")
        return route if isinstance(route, str) else None

    @staticmethod
    def _request_turns(translation: "BaseTranslation | None", request_data: dict) -> list[dict]:
        """The request conversation, as litellm's translation for the route reads the body.

        Falls back to a chat-shaped ``messages`` list when the route has no translation that
        builds one (e.g. a direct ``/guardrails/apply_guardrail`` call).
        """
        structured: Final = translation.get_structured_messages(dict(request_data)) if translation else None
        turns: Final = (
            structured or XecGuardGuardrail._generate_content_turns(request_data) or request_data.get("messages") or []
        )
        return [XecGuardGuardrail._normalize_message(m) for m in turns if isinstance(m, dict)]

    @staticmethod
    def _generate_content_turns(request_data: Mapping[str, Any]) -> list[dict]:
        """Gemini ``generateContent`` bodies through litellm's own Gemini -> OpenAI adapter, for
        routes whose translation does not build structured messages. Empty for any other body."""
        contents: Final = request_data.get("contents")
        if not contents:
            return []
        from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter

        config: Final = request_data.get("config")
        system_instruction: Final = (
            request_data.get("systemInstruction")
            or request_data.get("system_instruction")
            or (
                config.get("systemInstruction") or config.get("system_instruction")
                if isinstance(config, Mapping)
                else None
            )
        )
        try:
            request: Final = GoogleGenAIAdapter().translate_generate_content_to_completion(
                model=str(request_data.get("model") or ""),
                contents=contents,
                systemInstruction=system_instruction,
            )
        except Exception as exc:
            verbose_proxy_logger.warning("XecGuard: could not read generateContent body, scanning without it (%s)", exc)
            return []
        return [m for m in request.get("messages") or [] if isinstance(m, dict)]

    def _build_full_history(
        self,
        request_data: dict,
        inputs: GenericGuardrailAPIInputs,
        input_type: str,
    ) -> list[dict]:
        """The conversation sent to XecGuard.

        Request side: ``inputs['structured_messages']``, which litellm's translation builds
        for every endpoint in OpenAI shape, system prompt included. Response side: the same
        request conversation, rebuilt by that translation from the body, followed by the reply
        text litellm extracted into ``inputs['texts']``.
        """
        structured: Final = inputs.get("structured_messages")
        if input_type == "request" and structured:
            self._warn_once_if_system_prompt_is_stripped()
        turns: Final = (
            [self._normalize_message(m) for m in structured]
            if input_type == "request" and structured
            else self._request_turns(self._translation_for_route(self._request_route(request_data)), request_data)
        )
        texts: Final = self._join_input_texts(inputs)
        if input_type == "response":
            tool_calls: Final = [_tool_call_dict(call) for call in inputs.get("tool_calls") or ()]
            if not texts and not tool_calls:
                return []
            reply: Final = {"role": "assistant", "content": texts, **({"tool_calls": tool_calls} if tool_calls else {})}
            return self._reply_conversation(turns, reply)
        if not turns and texts:
            return [{"role": "user", "content": texts}]
        if not turns or turns[-1].get("role") in ("user", "tool"):
            return turns
        return [*turns, {"role": "user", "content": texts}] if texts else []

    def _warn_once_if_system_prompt_is_stripped(self) -> None:
        if self._warned_system_stripped or not effective_skip_system_message_for_guardrail(self):
            return
        self._warned_system_stripped = True
        verbose_proxy_logger.warning(
            "XecGuard %s: skip_system_message_in_guardrail is in effect, so the system prompt is "
            "removed before the scan and policies that depend on it (e.g. SystemPromptEnforcement) "
            "may miss violations. Set it to false on this guardrail to keep the system prompt.",
            self.guardrail_name,
        )

    @staticmethod
    def _join_input_texts(inputs: Any) -> str | None:
        """Response-path fallback: on the MCP post_mcp_call seam the tool result
        arrives in ``inputs.texts`` (not ``request_data['response']``), so build the
        assistant turn from there when no LLM completion is present."""
        if not isinstance(inputs, dict):
            return None
        texts: Final = inputs.get("texts")
        if not texts:
            return None
        joined: Final = "\n".join(t for t in texts if isinstance(t, str) and t)
        return joined or None

    @staticmethod
    def _normalize_message(message: Mapping[str, object]) -> dict:
        """Flatten multimodal content to a plain string for XecGuard."""
        role: Final = message.get("role") or "user"
        content: Final = message.get("content")
        tool_fields: Final = {
            name: message[name] for name in ("tool_calls", "tool_call_id") if message.get(name) is not None
        }
        if isinstance(content, str):
            return {"role": role, "content": content, **tool_fields}
        if isinstance(content, list):
            parts: Final[list[str]] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return {"role": role, "content": "\n".join(parts), **tool_fields}
        return {"role": role, "content": None if tool_fields.get("tool_calls") else "", **tool_fields}

    @staticmethod
    def _extract_last_text_by_role(messages: list[dict], role: str) -> str | None:
        for message in reversed(messages):
            if message.get("role") == role:
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
                return None
        return None

    # ------------------------------------------------------------------
    # Grounding document extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_grounding_documents(request_data: dict) -> list[dict]:
        raw_docs: Final = _proxy_metadata_field(request_data, _METADATA_GROUNDING_KEY)
        if not isinstance(raw_docs, list) or not raw_docs:
            return []
        valid_docs: Final[list[dict]] = []
        for doc in raw_docs:
            if (
                isinstance(doc, dict)
                and isinstance(doc.get("document_id"), str)
                and isinstance(doc.get("context"), str)
            ):
                valid_docs.append(
                    {
                        "document_id": doc["document_id"],
                        "context": doc["context"],
                    }
                )
            else:
                verbose_proxy_logger.debug(
                    "XecGuard: dropping malformed grounding document: %r",
                    doc,
                )
        return valid_docs

    # ------------------------------------------------------------------
    # Error-message formatting
    # ------------------------------------------------------------------

    def _record_scan_outcome(
        self,
        request_data: dict,
        scan_result: dict,
        event_type: GuardrailEventHooks,
        guardrail_status: GuardrailStatus,
        tracing_detail: GuardrailTracingDetail,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None:
        scan_started, scan_ended = _scan_window(scan_result)
        started: Final = scan_started if start_time is None else start_time
        ended: Final = scan_ended if end_time is None else end_time
        skipped: Final = self._skipped_policies(scan_result)
        if skipped:
            verbose_proxy_logger.warning(
                "XecGuard skipped policies, nothing was checked for them: %s",
                json.dumps(
                    {**self._request_context(request_data, scan_result), "skipped_policies": skipped},
                    ensure_ascii=False,
                    default=str,
                ),
            )
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={
                **_sanitize_scan_result_for_logging(scan_result),
                "request_context": self._request_context(request_data, scan_result),
            },
            request_data=request_data,
            guardrail_status=guardrail_status,
            event_type=event_type,
            tracing_detail=cast(
                GuardrailTracingDetail,
                {**tracing_detail, **self._skipped_tracing_detail(skipped, tracing_detail.get("detection_method"))},
            )
            if skipped
            else tracing_detail,
            start_time=started,
            end_time=ended,
            duration=ended - started if started is not None and ended is not None else None,
        )
        self._backfill_rationale(request_data, scan_result)

    def _backfill_rationale(self, request_data: Mapping[str, Any], scan_result: dict) -> None:
        trace_id: Final = scan_result.get("trace_id")
        if self.rationale_poll_seconds <= 0 or not isinstance(trace_id, str) or not trace_id:
            return
        if not _has_pending_rationale(self._violations(scan_result)):
            return
        guardrail_name: Final = self.guardrail_name or "xecguard"
        call_id: Final = request_data.get("litellm_call_id")
        _start_backfill(
            fetch_trace=self._fetch_trace,
            store_rationale=lambda request_id, rationales: _update_spend_log(
                guardrail_name, trace_id, request_id, rationales
            ),
            request_id=call_id if isinstance(call_id, str) else None,
            trace_id=trace_id,
            timeout_seconds=self.rationale_poll_seconds,
            log_context={k: v for k, v in self._request_context(request_data, scan_result).items() if k != "rationale"},
        )

    async def _fetch_trace(self, trace_id: str) -> dict | None:
        """The stored trace of a scan, or None while it cannot be read yet. A token that is not
        allowed to read traces raises, since asking again cannot change that."""
        try:
            response: Final = await self.async_handler.get(
                url=f"{self.api_base}{_TRACE_ENDPOINT}/{trace_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            response.raise_for_status()
            body: Final = response.json()
            return body if isinstance(body, dict) else None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _TRACE_DENIED_STATUSES:
                raise
            verbose_proxy_logger.debug("XecGuard trace %s not readable yet: %s", trace_id, exc)
            return None
        except Exception as exc:
            verbose_proxy_logger.debug("XecGuard trace %s fetch failed: %s", trace_id, exc)
            return None

    @staticmethod
    def _skipped_policies(scan_result: Mapping[str, object]) -> list[dict[str, str]]:
        """Policies XecGuard did not run, with why (``no_system_prompt``, ``no_tool_call``, ...).
        ``null`` and an empty list both mean every requested policy ran."""
        raw: Final = scan_result.get("skipped_policies")
        items: Final[Sequence[object]] = raw if isinstance(raw, list) else ()
        return [
            {"policy_name": str(item.get("policy_name")), "reason": str(item.get("reason"))}
            for item in items
            if isinstance(item, dict) and item.get("policy_name")
        ]

    @staticmethod
    def _skipped_tracing_detail(skipped: Sequence[Mapping[str, str]], scan_label: str | None) -> dict[str, object]:
        """The skipped policies in the fields the Logs page already draws: ``detection_method`` is
        the badge on the guardrail row and survives spend log redaction, ``classification`` is the
        panel that opens under it and is redacted unless prompts are stored."""
        labels: Final = tuple(f"{item['policy_name']} ({item['reason']})" for item in skipped)
        return {
            "skipped_policies": list(skipped),
            "detection_method": " | ".join(filter(None, (scan_label, f"skipped: {' | '.join(labels)}"))),
            "classification": {"category": "Skipped policies", "reason": "; ".join(labels)},
        }

    async def async_post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: object,
        request_headers: dict[str, str] | None = None,
        litellm_call_info: dict[str, object] | None = None,
    ) -> dict[str, str] | None:
        """``x-xecguard-skipped-policies`` on the response, so a caller sees that a policy it asked
        for was never checked. Read from what the scans recorded, so every XecGuard guardrail
        answers with the same value."""
        skipped: Final = self._recorded_skipped_policies(data)
        if not skipped:
            return None
        value: Final = ",".join(f"{item['policy_name']}:{item['reason']}" for item in skipped)
        return {_SKIPPED_POLICIES_HEADER: "".join(ch for ch in value if " " <= ch <= "~")}

    @staticmethod
    def _recorded_skipped_policies(request_data: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
        entries: Final = tuple(
            entry
            for bucket in _METADATA_BUCKETS
            for container in (request_data.get(bucket),)
            if isinstance(container, dict)
            for recorded in (container.get("standard_logging_guardrail_information"),)
            if isinstance(recorded, list)
            for entry in recorded
            if isinstance(entry, dict)
        )
        seen: Final = {
            (item["policy_name"], item["reason"]): item
            for entry in entries
            for item in XecGuardGuardrail._skipped_policies(entry)
        }
        return tuple(seen.values())

    def _request_context(self, request_data: Mapping[str, Any], scan_result: dict) -> dict[str, Any]:
        fields: Final = {
            **self._block_log_fields(request_data, scan_result),
            **dict(self._custom_header_meta_items(request_data)),
        }
        return {name: value for name, value in fields.items() if value not in (None, "", ())}

    @classmethod
    def _block_log_fields(cls, request_data: Mapping[str, Any], scan_result: dict) -> dict[str, Any]:
        context: Final = cls._key_context(request_data) or {}

        def proxy_field(name: str) -> object:
            return _proxy_metadata_field(context, name)

        rationale: Final = next((v["rationale"] for v in cls._violations(scan_result) if v.get("rationale")), None)
        return {
            "request_id": request_data.get("litellm_call_id"),
            "session_id": request_data.get("litellm_session_id") or proxy_field("session_id"),
            "user_id": proxy_field("user_api_key_user_id"),
            "source_ip": proxy_field("requester_ip_address"),
            "user_agent": proxy_field("user_agent"),
            "trace_id": scan_result.get("trace_id"),
            "policies": cls._violated_policy_names(scan_result),
            "rationale": rationale,
        }

    @staticmethod
    def _violations(result: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
        """``/scan`` returns a list of violations, ``/grounding`` a single object."""
        raw: Final = result.get("xecguard_result")
        items: Final[Sequence[object]] = cast(list[object], raw) if isinstance(raw, list) else (raw,)
        return tuple(cast(dict[str, object], v) for v in items if isinstance(v, dict))

    @classmethod
    def _violated_policy_names(cls, result: Mapping[str, object]) -> tuple[str, ...]:
        names: Final = (v.get("violated_policy_name") for v in cls._violations(result))
        return tuple(dict.fromkeys(name for name in names if isinstance(name, str) and name))

    @classmethod
    def _block_message(cls, result: dict) -> str:
        """``/scan`` answers with a list of violations, ``/grounding`` with a single one."""
        if isinstance(result.get("xecguard_result"), dict):
            return cls._format_grounding_block_message(result)
        return cls._format_scan_block_message(result)

    @staticmethod
    def _format_scan_block_message(result: dict) -> str:
        trace_id: Final = result.get("trace_id", "")
        violations = result.get("xecguard_result")
        if not isinstance(violations, list):
            violations = []
        policies: Final = ",".join(XecGuardGuardrail._violated_policy_names(result)) or "unknown"
        rationale = ""
        for v in violations:
            if isinstance(v, dict):
                candidate = v.get("rationale")
                if isinstance(candidate, str) and candidate:
                    rationale = candidate[:_RATIONALE_TRUNCATE_CHARS]
                    break
        return f"Blocked by XecGuard: policies=[{policies}] trace_id={trace_id} rationale={rationale}"

    @staticmethod
    def _format_grounding_block_message(result: dict) -> str:
        trace_id: Final = result.get("trace_id", "")
        detail: Final = result.get("xecguard_result")
        rules: list[str] = []
        rationale = ""
        if isinstance(detail, dict):
            raw_rules: Final = detail.get("violated_rules_list")
            if isinstance(raw_rules, list):
                rules = [r for r in raw_rules if isinstance(r, str)]
            candidate: Final = detail.get("rationale")
            if isinstance(candidate, str):
                rationale = candidate[:_RATIONALE_TRUNCATE_CHARS]
        rules_str: Final = ",".join(rules) if rules else "unknown"
        return f"Blocked by XecGuard grounding: rules=[{rules_str}] trace_id={trace_id} rationale={rationale}"
