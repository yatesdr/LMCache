# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.admission import (
    AdmissionAttempt,
    AdmissionFailure,
    AdmissionOutcome,
    AdmissionWaitResult,
    reserve_with_eviction_backpressure,
)


def test_capacity_failure_retries_after_eviction_change() -> None:
    """Capacity admission waits, requests eviction, and retries atomically."""
    attempts = 0
    callbacks: list[str] = []

    def attempt() -> AdmissionAttempt[str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return AdmissionAttempt.failure(AdmissionFailure.CAPACITY)
        return AdmissionAttempt.success("reserved")

    outcome: AdmissionOutcome[str] = reserve_with_eviction_backpressure(
        attempt=attempt,
        get_generation=lambda: 0,
        request_eviction=lambda: callbacks.append("evict"),
        wait_for_change=lambda _generation, _timeout: (
            AdmissionWaitResult.CHANGED,
            1,
        ),
        timeout_seconds=1.0,
        on_wait=lambda: callbacks.append("wait"),
        on_retry=lambda: callbacks.append("retry"),
        on_success_after_eviction=lambda: callbacks.append("success"),
    )

    assert outcome.value == "reserved"
    assert outcome.failure is None
    assert outcome.retries == 1
    assert outcome.waited is True
    assert callbacks == ["wait", "evict", "retry", "success"]


def test_conflict_fails_without_eviction() -> None:
    """Non-capacity failures do not evict unrelated readable objects."""
    eviction_calls = 0

    def request_eviction() -> None:
        nonlocal eviction_calls
        eviction_calls += 1

    outcome: AdmissionOutcome[object] = reserve_with_eviction_backpressure(
        attempt=lambda: AdmissionAttempt.failure(AdmissionFailure.CONFLICT),
        get_generation=lambda: 0,
        request_eviction=request_eviction,
        wait_for_change=lambda _generation, _timeout: (
            AdmissionWaitResult.TIMEOUT,
            0,
        ),
        timeout_seconds=1.0,
    )

    assert outcome.failure is AdmissionFailure.CONFLICT
    assert outcome.waited is False
    assert eviction_calls == 0


def test_capacity_timeout_is_bounded_and_structured() -> None:
    """A zero deadline returns the stable timeout reason without retrying."""
    timed_out = 0

    def on_timeout() -> None:
        nonlocal timed_out
        timed_out += 1

    outcome: AdmissionOutcome[object] = reserve_with_eviction_backpressure(
        attempt=lambda: AdmissionAttempt.failure(AdmissionFailure.CAPACITY),
        get_generation=lambda: 0,
        request_eviction=lambda: None,
        wait_for_change=lambda _generation, _timeout: (
            AdmissionWaitResult.TIMEOUT,
            0,
        ),
        timeout_seconds=0.0,
        on_timeout=on_timeout,
    )

    assert outcome.failure is AdmissionFailure.TIMEOUT
    assert outcome.retries == 0
    assert outcome.waited is True
    assert timed_out == 1


def test_shutdown_stops_capacity_retry() -> None:
    """Shutdown wakes admission without another allocation attempt."""
    attempts = 0

    def attempt() -> AdmissionAttempt[object]:
        nonlocal attempts
        attempts += 1
        return AdmissionAttempt.failure(AdmissionFailure.CAPACITY)

    outcome = reserve_with_eviction_backpressure(
        attempt=attempt,
        get_generation=lambda: 0,
        request_eviction=lambda: None,
        wait_for_change=lambda _generation, _timeout: (
            AdmissionWaitResult.SHUTDOWN,
            0,
        ),
        timeout_seconds=1.0,
    )

    assert outcome.failure is AdmissionFailure.SHUTDOWN
    assert attempts == 1


def test_invalid_timeouts_are_rejected() -> None:
    """Configuration errors fail before attempting a reservation."""

    def invoke(timeout: float, retry_wait: float) -> None:
        reserve_with_eviction_backpressure(
            attempt=lambda: AdmissionAttempt.success(None),
            get_generation=lambda: 0,
            request_eviction=lambda: None,
            wait_for_change=lambda _generation, _timeout: (
                AdmissionWaitResult.TIMEOUT,
                0,
            ),
            timeout_seconds=timeout,
            retry_wait_seconds=retry_wait,
        )

    with pytest.raises(ValueError, match="timeout_seconds"):
        invoke(-1.0, 0.5)
    with pytest.raises(ValueError, match="retry_wait_seconds"):
        invoke(1.0, 0.0)
