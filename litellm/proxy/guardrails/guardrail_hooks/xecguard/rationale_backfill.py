"""Fills in a rationale XecGuard produced after the verdict.

XecGuard answers an UNSAFE scan at once with ``rationale_status: ANALYSIS_REQUIRED`` and a placeholder
rationale; the real one is READY later and is read from ``GET /xecguard/v1/trace/{trace_id}``. The
client has already been answered by then, so this only rewrites what was recorded: one proxy log line
and the guardrail entry of the spend log row. It never changes a decision.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Final, cast

from litellm._logging import verbose_proxy_logger

RATIONALE_READY: Final = "READY"
RATIONALE_PENDING: Final = "ANALYSIS_REQUIRED"
MAX_POLL_SECONDS: Final = 600
POLL_INTERVAL_SECONDS: Final = 2.0
SPEND_LOG_ATTEMPTS: Final = 15
_SPEND_LOG_LOOKBACK: Final = timedelta(hours=1)

_FIND_SPEND_LOG_SQL: Final = """
SELECT request_id, metadata FROM "LiteLLM_SpendLogs"
WHERE "startTime" >= $2::timestamp AND (request_id = $1 OR metadata->>'litellm_call_id' = $1)
LIMIT 1
"""
_UPDATE_SPEND_LOG_SQL: Final = 'UPDATE "LiteLLM_SpendLogs" SET metadata = $2::jsonb WHERE request_id = $1'

FetchTrace = Callable[[str], Awaitable[Mapping[str, object] | None]]
StoreRationale = Callable[[str, Mapping[str, str]], Awaitable[bool]]

_running: Final[set["asyncio.Task[None]"]] = set()  # mutable-ok: keeps background tasks from being collected


def has_pending_rationale(violations: Sequence[Mapping[str, object]]) -> bool:
    return any(v.get("rationale_status") == RATIONALE_PENDING for v in violations)


def ready_rationales(trace: Mapping[str, object]) -> tuple[Mapping[str, str], bool]:
    """Policy name -> rationale for every reason that is READY, and whether all of them are."""
    scan_result: Final = trace.get("scan_result")
    raw: Final = scan_result.get("reasons") if isinstance(scan_result, Mapping) else None
    reasons: Final = tuple(r for r in raw if isinstance(r, Mapping)) if isinstance(raw, list) else ()
    ready: Final = {
        str(r["violated_policy_name"]): str(r["rationale"])
        for r in reasons
        if r.get("rationale_status") == RATIONALE_READY and r.get("violated_policy_name") and r.get("rationale")
    }
    return ready, bool(reasons) and all(r.get("rationale_status") == RATIONALE_READY for r in reasons)


def patch_spend_log_metadata(
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
    if not isinstance(violation, Mapping) or violation.get("rationale_status") != RATIONALE_PENDING:
        return violation
    real: Final = rationales.get(str(violation.get("violated_policy_name")))
    if real is None:
        return violation
    return {**violation, "rationale": real, "rationale_status": RATIONALE_READY}


async def poll_ready_rationales(
    fetch_trace: FetchTrace, trace_id: str, timeout_seconds: float, interval_seconds: float
) -> Mapping[str, str]:
    """Poll the trace until every reason is READY or the time is up, keeping what was READY by then."""
    loop: Final = asyncio.get_running_loop()
    deadline: Final = loop.time() + timeout_seconds
    ready: Mapping[str, str] = {}
    while True:
        trace = await fetch_trace(trace_id)
        if trace is not None:
            ready, complete = ready_rationales(trace)
            if complete or loop.time() + interval_seconds > deadline:
                return ready
        elif loop.time() + interval_seconds > deadline:
            return ready
        await asyncio.sleep(interval_seconds)


async def backfill_rationale(
    *,
    fetch_trace: FetchTrace,
    store_rationale: StoreRationale,
    request_id: str | None,
    trace_id: str,
    timeout_seconds: float,
    interval_seconds: float = POLL_INTERVAL_SECONDS,
    store_attempts: int = SPEND_LOG_ATTEMPTS,
    log_context: Mapping[str, object] | None = None,
) -> None:
    try:
        rationales: Final = await poll_ready_rationales(fetch_trace, trace_id, timeout_seconds, interval_seconds)
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
                await asyncio.sleep(interval_seconds)
        verbose_proxy_logger.warning("XecGuard rationale for trace_id=%s found no spend log row to update", trace_id)
    except Exception as exc:
        verbose_proxy_logger.warning("XecGuard rationale backfill for trace_id=%s failed: %s", trace_id, exc)


def start_backfill(
    *,
    fetch_trace: FetchTrace,
    store_rationale: StoreRationale,
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
        backfill_rationale(
            fetch_trace=fetch_trace,
            store_rationale=store_rationale,
            request_id=request_id,
            trace_id=trace_id,
            timeout_seconds=timeout_seconds,
            log_context=log_context,
        )
    )
    _running.add(task)
    task.add_done_callback(_running.discard)


async def update_spend_log(guardrail_name: str, trace_id: str, request_id: str, rationales: Mapping[str, str]) -> bool:
    """Rewrite the guardrail entry of the spend log row for ``request_id``. False means try again later:
    the row is written in a batch after the request, so it may not exist yet."""
    from litellm.proxy.proxy_server import prisma_client
    from litellm.proxy.spend_tracking.spend_tracking_utils import should_store_prompts_and_responses_in_spend_logs

    if prisma_client is None or not should_store_prompts_and_responses_in_spend_logs():
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
    patched: Final = patch_spend_log_metadata(metadata, guardrail_name, trace_id, rationales)
    if patched is None:
        return True
    await prisma_client.db.execute_raw(_UPDATE_SPEND_LOG_SQL, row_id, json.dumps(patched, default=str))
    return True
