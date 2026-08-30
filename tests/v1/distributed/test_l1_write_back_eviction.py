# SPDX-License-Identifier: Apache-2.0
"""Fail-closed L1 writeback eviction tests.

The core writeback contract is deliberately independent of periodic backup and
emergency prefetch pressure: when enabled, an L1 eviction batch is deleted only
after one synchronous L2 adapter reports every readable key durable.
"""

# Standard
from collections.abc import Iterator
from typing import Literal, cast
import argparse
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchMode,
    PrefetchRequestSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _object_key_to_filename,
)
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    InFlightPrefetchRequest,
    PrefetchController,
    PrefetchPhase,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.config import add_observability_args

POOL_BYTES = 8 * 1024 * 1024
OBJECT_LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size([256, 1024])],
    dtypes=[torch.float32],
)


class _SyncStoreAdapter:
    def __init__(self) -> None:
        self.stored_batches: list[list[ObjectKey]] = []
        self.result: tuple[bool, int, int] | None = None

    def store_objects_sync(
        self,
        keys: list[ObjectKey],
        objects: list,
        timeout: float | None = None,
    ) -> tuple[bool, int, int]:
        del timeout
        self.stored_batches.append(list(keys))
        if self.result is not None:
            return self.result
        return True, len(keys), sum(obj.get_size() for obj in objects)


class _BlockingSyncStoreAdapter(_SyncStoreAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def store_objects_sync(
        self,
        keys: list[ObjectKey],
        objects: list,
        timeout: float | None = None,
    ) -> tuple[bool, int, int]:
        self.entered.set()
        assert self.release.wait(timeout=timeout or 5.0)
        return super().store_objects_sync(keys, objects, timeout=timeout)


@pytest.fixture
def l1_manager() -> Iterator[L1Manager]:
    manager = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=POOL_BYTES,
                use_lazy=False,
                init_size_in_bytes=POOL_BYTES,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        )
    )
    yield manager
    manager.close()


def _make_keys(count: int) -> list[ObjectKey]:
    return [
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(index),
            model_name="writeback-test",
            kv_rank=0,
        )
        for index in range(count)
    ]


def _store_keys(manager: L1Manager, keys: list[ObjectKey]) -> None:
    result = manager.reserve_write(keys, [False] * len(keys), OBJECT_LAYOUT)
    assert all(result[key][0] == L1Error.SUCCESS for key in keys)
    manager.finish_write(keys)


def _readable_keys(manager: L1Manager, keys: list[ObjectKey]) -> list[ObjectKey]:
    result = manager.reserve_read(keys)
    readable = [key for key in keys if result[key][0] == L1Error.SUCCESS]
    if readable:
        manager.finish_read(readable)
    return readable


def _controller(
    manager: L1Manager,
    adapters: dict[int, object],
    *,
    enabled: bool = True,
    periodic_flush_interval: float = 0.0,
    eviction_policy: Literal["LRU", "IsolatedLRU", "noop"] = "LRU",
) -> L1EvictionController:
    return L1EvictionController(
        l1_manager=manager,
        eviction_config=EvictionConfig(
            eviction_policy=eviction_policy,
            write_back_on_evict=enabled,
            periodic_flush_interval=periodic_flush_interval,
        ),
        l2_adapters=adapters,
    )


def test_writeback_defaults_off() -> None:
    assert EvictionConfig(eviction_policy="LRU").write_back_on_evict is False


def test_writeback_cli_flag_is_plumbed() -> None:
    parser = argparse.ArgumentParser()
    add_storage_manager_args(parser)
    add_observability_args(parser)
    args = parser.parse_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--write-back-on-evict",
        ]
    )

    assert parse_args_to_config(args).eviction_config.write_back_on_evict is True


def test_periodic_backup_defaults_off_and_cli_is_plumbed() -> None:
    assert EvictionConfig(eviction_policy="LRU").periodic_flush_interval == 0.0
    parser = argparse.ArgumentParser()
    add_storage_manager_args(parser)
    add_observability_args(parser)
    args = parser.parse_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--periodic-flush-interval",
            "30.0",
        ]
    )

    assert parse_args_to_config(args).eviction_config.periodic_flush_interval == 30.0


def test_negative_periodic_backup_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="periodic_flush_interval"):
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=POOL_BYTES,
                    use_lazy=False,
                )
            ),
            eviction_config=EvictionConfig(
                eviction_policy="LRU",
                periodic_flush_interval=-1.0,
            ),
        )


def test_emergency_prefetch_defaults_off_and_cli_is_plumbed() -> None:
    assert EvictionConfig(eviction_policy="LRU").emergency_evict_for_prefetch is False
    parser = argparse.ArgumentParser()
    add_storage_manager_args(parser)
    add_observability_args(parser)
    args = parser.parse_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--write-back-on-evict",
            "--emergency-evict-for-prefetch",
        ]
    )

    config = parse_args_to_config(args).eviction_config
    assert config.write_back_on_evict is True
    assert config.emergency_evict_for_prefetch is True


def test_emergency_prefetch_requires_writeback() -> None:
    with pytest.raises(ValueError, match="requires write_back_on_evict"):
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=POOL_BYTES,
                    use_lazy=False,
                )
            ),
            eviction_config=EvictionConfig(
                eviction_policy="LRU",
                emergency_evict_for_prefetch=True,
            ),
        )


def test_disabled_controller_keeps_discard_behavior(l1_manager: L1Manager) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(l1_manager, {0: adapter}, enabled=False)
    keys = _make_keys(2)
    _store_keys(l1_manager, keys)

    actions = controller._eviction_policy.get_eviction_actions(
        1.0,
        key_eligible_filter=l1_manager.is_key_evictable,
    )
    assert len(actions) == 1
    assert actions[0].destination is EvictionDestination.DISCARD
    controller.execute_eviction_action(actions[0])

    assert adapter.stored_batches == []
    assert _readable_keys(l1_manager, keys) == []


def test_disabled_controller_preserves_legacy_l2_action_fallback(
    l1_manager: L1Manager,
) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(l1_manager, {0: adapter}, enabled=False)
    keys = _make_keys(2)
    _store_keys(l1_manager, keys)

    controller.execute_eviction_action(
        EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
    )

    assert controller.has_l2_flush_adapter() is False
    assert adapter.stored_batches == []
    assert _readable_keys(l1_manager, keys) == []


def test_missing_sync_adapter_fails_closed(l1_manager: L1Manager) -> None:
    controller = _controller(l1_manager, {0: object()})
    keys = _make_keys(2)
    _store_keys(l1_manager, keys)

    actions = controller._eviction_policy.get_eviction_actions(
        1.0,
        key_eligible_filter=l1_manager.is_key_evictable,
    )
    assert len(actions) == 1
    assert actions[0].destination is EvictionDestination.L2_CACHE
    controller.execute_eviction_action(actions[0])

    assert controller.has_l2_flush_adapter() is False
    assert _readable_keys(l1_manager, keys) == keys


def test_successful_flush_deletes_only_after_durability(
    l1_manager: L1Manager,
) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(l1_manager, {0: adapter})
    keys = _make_keys(3)
    _store_keys(l1_manager, keys)

    controller.execute_eviction_action(
        EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
    )

    assert adapter.stored_batches == [keys]
    assert _readable_keys(l1_manager, keys) == []


@pytest.mark.parametrize("result", [(False, 0, 0), (False, 2, 1024)])
def test_failed_or_partial_flush_preserves_entire_readable_batch(
    l1_manager: L1Manager,
    result: tuple[bool, int, int],
) -> None:
    adapter = _SyncStoreAdapter()
    adapter.result = result
    controller = _controller(l1_manager, {0: adapter})
    keys = _make_keys(3)
    _store_keys(l1_manager, keys)

    controller.execute_eviction_action(
        EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
    )

    assert _readable_keys(l1_manager, keys) == keys
    status = controller.report_status()
    assert status["sync_flush_failures"] == 1
    assert status["sync_flush_backoff_seconds"] > 0.0


def test_later_adapter_can_make_batch_durable(l1_manager: L1Manager) -> None:
    failed = _SyncStoreAdapter()
    failed.result = (False, 0, 0)
    durable = _SyncStoreAdapter()
    controller = _controller(l1_manager, {3: failed, 7: durable})
    keys = _make_keys(2)
    _store_keys(l1_manager, keys)

    controller.execute_eviction_action(
        EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
    )

    assert failed.stored_batches == [keys]
    assert durable.stored_batches == [keys]
    assert _readable_keys(l1_manager, keys) == []


def test_reserve_read_race_never_deletes_unreadable_key(
    l1_manager: L1Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(l1_manager, {0: adapter})
    keys = _make_keys(1)
    _store_keys(l1_manager, keys)
    original_reserve_read = l1_manager.reserve_read
    monkeypatch.setattr(
        l1_manager,
        "reserve_read",
        lambda _keys: {keys[0]: (L1Error.KEY_IS_LOCKED, None)},
    )

    controller.execute_eviction_action(
        EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
    )

    monkeypatch.setattr(l1_manager, "reserve_read", original_reserve_read)
    assert adapter.stored_batches == []
    assert _readable_keys(l1_manager, keys) == keys


def test_adapter_replacement_waits_for_active_flush(l1_manager: L1Manager) -> None:
    adapter = _BlockingSyncStoreAdapter()
    controller = _controller(l1_manager, {0: adapter})
    keys = _make_keys(1)
    _store_keys(l1_manager, keys)
    flush = threading.Thread(
        target=lambda: controller.execute_eviction_action(
            EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
        )
    )
    replaced = threading.Event()

    def replace_adapters() -> None:
        controller.set_l2_adapters({})
        replaced.set()

    replacement = threading.Thread(target=replace_adapters)
    flush.start()
    assert adapter.entered.wait(timeout=5.0)
    replacement.start()
    time.sleep(0.05)
    assert not replaced.is_set()

    adapter.release.set()
    flush.join(timeout=5.0)
    replacement.join(timeout=5.0)

    assert replaced.is_set()
    assert not flush.is_alive()
    assert not replacement.is_alive()


def test_adapter_removal_waits_for_active_flush(l1_manager: L1Manager) -> None:
    adapter = _BlockingSyncStoreAdapter()
    controller = _controller(l1_manager, {0: adapter})
    keys = _make_keys(1)
    _store_keys(l1_manager, keys)
    flush = threading.Thread(
        target=lambda: controller.execute_eviction_action(
            EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
        )
    )
    removed = threading.Event()

    def remove_adapter() -> None:
        controller.remove_l2_adapter(0)
        removed.set()

    removal = threading.Thread(target=remove_adapter)
    flush.start()
    assert adapter.entered.wait(timeout=5.0)
    removal.start()
    time.sleep(0.05)
    assert not removed.is_set()

    adapter.release.set()
    flush.join(timeout=5.0)
    removal.join(timeout=5.0)

    assert removed.is_set()
    assert controller.has_l2_flush_adapter() is False
    assert not flush.is_alive()
    assert not removal.is_alive()


def test_backoff_skips_repeated_flush(l1_manager: L1Manager) -> None:
    adapter = _SyncStoreAdapter()
    adapter.result = (False, 0, 0)
    controller = _controller(l1_manager, {0: adapter})
    keys = _make_keys(2)
    _store_keys(l1_manager, keys)
    action = EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)

    controller.execute_eviction_action(action)
    controller.execute_eviction_action(action)

    assert adapter.stored_batches == [keys]
    assert _readable_keys(l1_manager, keys) == keys


def test_storage_manager_wires_native_filesystem_writeback(tmp_path) -> None:
    pytest.importorskip("lmcache.lmcache_fs")
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=POOL_BYTES,
                use_lazy=False,
                init_size_in_bytes=POOL_BYTES,
                align_bytes=0x1000,
            )
        ),
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
            write_back_on_evict=True,
        ),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[
                FSNativeL2AdapterConfig(
                    base_path=str(tmp_path),
                    num_workers=1,
                    relative_tmp_dir="pending",
                )
            ]
        ),
    )
    manager = StorageManager(config)
    keys = _make_keys(2)
    try:
        # Keep the native adapter owned by StorageManager/writeback, but detach
        # ordinary async store routing so it cannot hold an independent read
        # lock while this test triggers eviction directly.
        assert manager._store_controller.request_remove_adapter(0).wait(timeout=5.0)
        reserved = manager.reserve_write(keys, OBJECT_LAYOUT, mode="new")
        assert list(reserved) == keys
        # Finish directly in L1 so this test exercises eviction writeback,
        # rather than the ordinary asynchronous StoreController path.
        manager._l1_manager.finish_write(keys)

        manager._eviction_controller.execute_eviction_action(
            EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
        )

        assert manager._eviction_controller.has_l2_flush_adapter() is True
        assert _readable_keys(manager._l1_manager, keys) == []
        assert all((tmp_path / _object_key_to_filename(key)).is_file() for key in keys)
    finally:
        manager.close()


def test_storage_manager_wires_native_filesystem_periodic_backup(tmp_path) -> None:
    pytest.importorskip("lmcache.lmcache_fs")
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=POOL_BYTES,
                use_lazy=False,
                init_size_in_bytes=POOL_BYTES,
                align_bytes=0x1000,
            )
        ),
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
            periodic_flush_interval=30.0,
        ),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[
                FSNativeL2AdapterConfig(
                    base_path=str(tmp_path),
                    num_workers=1,
                    relative_tmp_dir="pending",
                )
            ]
        ),
    )
    manager = StorageManager(config)
    keys = _make_keys(2)
    try:
        assert manager._store_controller.request_remove_adapter(0).wait(timeout=5.0)
        reserved = manager.reserve_write(keys, OBJECT_LAYOUT, mode="new")
        assert list(reserved) == keys
        manager._l1_manager.finish_write(keys)

        manager._eviction_controller._backup_to_l2_no_delete(batch_limit=2)

        assert manager._eviction_controller.has_periodic_flush_adapter() is True
        assert _readable_keys(manager._l1_manager, keys) == keys
        assert all((tmp_path / _object_key_to_filename(key)).is_file() for key in keys)
    finally:
        manager.close()


def test_periodic_backup_keeps_l1_copy(l1_manager: L1Manager) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(
        l1_manager,
        {0: adapter},
        enabled=False,
        periodic_flush_interval=30.0,
    )
    keys = _make_keys(3)
    _store_keys(l1_manager, keys)

    controller._backup_to_l2_no_delete(batch_limit=3)

    assert adapter.stored_batches == [keys]
    assert _readable_keys(l1_manager, keys) == keys


def test_periodic_backup_rotates_a_bounded_scan(l1_manager: L1Manager) -> None:
    keys = _make_keys(5)
    _store_keys(l1_manager, keys)
    assert l1_manager.reserve_read([keys[0]])[keys[0]][0] == L1Error.SUCCESS
    try:
        first, cursor = l1_manager.get_evictable_keys(
            limit=2,
            cursor=0,
            scan_limit=2,
        )
        second, cursor = l1_manager.get_evictable_keys(
            limit=2,
            cursor=cursor,
            scan_limit=2,
        )
        third, _ = l1_manager.get_evictable_keys(
            limit=2,
            cursor=cursor,
            scan_limit=2,
        )
    finally:
        l1_manager.finish_read([keys[0]])

    assert first == [keys[1]]
    assert second == keys[2:4]
    assert third == [keys[4]]


def test_periodic_backup_scan_limit_is_a_hard_bound(l1_manager: L1Manager) -> None:
    keys = _make_keys(3)
    _store_keys(l1_manager, keys)

    batch, cursor = l1_manager.get_evictable_keys(
        limit=3,
        cursor=1,
        scan_limit=0,
    )

    assert batch == []
    assert cursor == 1


def test_periodic_backup_cursor_visits_the_keyspace(l1_manager: L1Manager) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(
        l1_manager,
        {0: adapter},
        enabled=False,
        periodic_flush_interval=30.0,
    )
    keys = _make_keys(6)
    _store_keys(l1_manager, keys)

    for _ in range(3):
        controller._backup_to_l2_no_delete(batch_limit=2)

    assert [len(batch) for batch in adapter.stored_batches] == [2, 2, 2]
    assert {key for batch in adapter.stored_batches for key in batch} == set(keys)
    assert _readable_keys(l1_manager, keys) == keys


def test_periodic_backup_has_bounded_store_wait(l1_manager: L1Manager) -> None:
    adapter = _BlockingSyncStoreAdapter()
    controller = _controller(
        l1_manager,
        {0: adapter},
        enabled=False,
        periodic_flush_interval=30.0,
    )
    controller._PERIODIC_FLUSH_TIMEOUT_SECONDS = 0.05
    keys = _make_keys(2)
    _store_keys(l1_manager, keys)

    start = time.monotonic()
    controller._backup_to_l2_no_delete(batch_limit=2)
    elapsed = time.monotonic() - start

    assert adapter.entered.is_set()
    assert elapsed < 0.5
    assert _readable_keys(l1_manager, keys) == keys


class _NoTimeoutSyncStoreAdapter:
    def __init__(self) -> None:
        self.stored_batches: list[list[ObjectKey]] = []

    def store_objects_sync(
        self,
        keys: list[ObjectKey],
        objects: list,
    ) -> tuple[bool, int, int]:
        self.stored_batches.append(list(keys))
        return True, len(keys), sum(obj.get_size() for obj in objects)


class _PendingBufferSyncStoreAdapter(_SyncStoreAdapter):
    def __init__(self, pending: set[ObjectKey]) -> None:
        super().__init__()
        self.pending = pending

    def has_inflight_store_for_keys(self, keys: list[ObjectKey]) -> bool:
        return any(key in self.pending for key in keys)


def test_periodic_backup_skips_adapter_without_timeout_contract(
    l1_manager: L1Manager,
) -> None:
    adapter = _NoTimeoutSyncStoreAdapter()
    controller = _controller(
        l1_manager,
        {0: adapter},
        enabled=False,
        periodic_flush_interval=30.0,
    )
    keys = _make_keys(2)
    _store_keys(l1_manager, keys)

    controller._backup_to_l2_no_delete(batch_limit=2)

    assert controller.has_l2_flush_adapter() is True
    assert controller.has_periodic_flush_adapter() is False
    assert adapter.stored_batches == []
    assert _readable_keys(l1_manager, keys) == keys


def test_periodic_backup_loop_runs_below_watermark(l1_manager: L1Manager) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(
        l1_manager,
        {0: adapter},
        enabled=False,
        periodic_flush_interval=0.01,
    )
    keys = _make_keys(2)
    _store_keys(l1_manager, keys)

    controller.start()
    try:
        deadline = time.monotonic() + 5.0
        while not adapter.stored_batches and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        controller.stop()

    assert adapter.stored_batches
    assert _readable_keys(l1_manager, keys) == keys


def test_emergency_evict_frees_l1_through_writeback(
    l1_manager: L1Manager,
) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(l1_manager, {0: adapter})
    keys = _make_keys(6)
    _store_keys(l1_manager, keys)
    used_before, total = l1_manager.get_memory_usage()
    free_before = total - used_before

    free_after = controller.emergency_evict_bytes(
        free_before + 2 * 1024 * 1024,
        requester="test",
    )

    assert free_after > free_before
    evicted = [key for batch in adapter.stored_batches for key in batch]
    assert evicted
    assert set(evicted) <= set(keys)
    assert len(evicted) < len(keys)
    assert not set(evicted) & set(_readable_keys(l1_manager, keys))


def test_emergency_evict_is_noop_with_enough_free(l1_manager: L1Manager) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(l1_manager, {0: adapter})
    keys = _make_keys(2)
    _store_keys(l1_manager, keys)

    controller.emergency_evict_bytes(1024, requester="test")

    assert adapter.stored_batches == []
    assert _readable_keys(l1_manager, keys) == keys


def test_emergency_evict_respects_isolated_lru_cache_salt(
    l1_manager: L1Manager,
) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(
        l1_manager,
        {0: adapter},
        eviction_policy="IsolatedLRU",
    )
    salt_a = [
        ObjectKey(
            ObjectKey.IntHash2Bytes(index),
            "writeback-test",
            0,
            cache_salt="tenant-a",
        )
        for index in range(3)
    ]
    salt_b = [
        ObjectKey(
            ObjectKey.IntHash2Bytes(index + 10),
            "writeback-test",
            0,
            cache_salt="tenant-b",
        )
        for index in range(3)
    ]
    _store_keys(l1_manager, salt_a + salt_b)
    used, total = l1_manager.get_memory_usage()

    controller.emergency_evict_bytes(
        total - used + 1024 * 1024,
        requester="tenant-a-restore",
        cache_salt="tenant-a",
    )

    evicted = [key for batch in adapter.stored_batches for key in batch]
    assert evicted
    assert all(key.cache_salt == "tenant-a" for key in evicted)
    assert _readable_keys(l1_manager, salt_b) == salt_b


def test_emergency_evict_fails_closed_without_isolated_lru_cache_salt(
    l1_manager: L1Manager,
) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(
        l1_manager,
        {0: adapter},
        eviction_policy="IsolatedLRU",
    )
    keys = [
        ObjectKey(
            ObjectKey.IntHash2Bytes(index),
            "writeback-test",
            0,
            cache_salt="tenant-a",
        )
        for index in range(4)
    ]
    _store_keys(l1_manager, keys)
    used, total = l1_manager.get_memory_usage()
    free_before = total - used

    assert controller.emergency_evict_bytes(free_before + 1024) == free_before
    assert adapter.stored_batches == []
    assert _readable_keys(l1_manager, keys) == keys


def test_emergency_evict_skips_policy_during_backoff(
    l1_manager: L1Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _SyncStoreAdapter()
    controller = _controller(l1_manager, {0: adapter})
    keys = _make_keys(4)
    _store_keys(l1_manager, keys)
    controller._sync_flush_backoff_until = time.monotonic() + 60
    monkeypatch.setattr(
        controller._eviction_policy,
        "get_eviction_actions",
        lambda *args, **kwargs: pytest.fail("eviction scan should be skipped"),
    )
    used, total = l1_manager.get_memory_usage()
    free = total - used

    assert controller.emergency_evict_bytes(free + 1024) == free


def test_emergency_evict_preserves_inflight_native_store_buffers(
    l1_manager: L1Manager,
) -> None:
    keys = _make_keys(4)
    adapter = _PendingBufferSyncStoreAdapter(set(keys))
    controller = _controller(l1_manager, {0: adapter})
    _store_keys(l1_manager, keys)
    used, total = l1_manager.get_memory_usage()
    free_before = total - used

    assert controller.emergency_evict_bytes(free_before + 1024) == free_before
    assert adapter.stored_batches == []
    assert _readable_keys(l1_manager, keys) == keys

    controller.execute_eviction_action(
        EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
    )
    assert adapter.stored_batches == []
    assert _readable_keys(l1_manager, keys) == keys


def test_emergency_deadline_does_not_trip_backend_backoff(
    l1_manager: L1Manager,
) -> None:
    adapter = _BlockingSyncStoreAdapter()
    controller = _controller(l1_manager, {0: adapter})
    controller._EMERGENCY_FLUSH_TIMEOUT_SECONDS = 0.05
    keys = _make_keys(4)
    _store_keys(l1_manager, keys)
    used, total = l1_manager.get_memory_usage()
    free_before = total - used

    start = time.monotonic()
    free_after = controller.emergency_evict_bytes(free_before + 1024)
    elapsed = time.monotonic() - start

    assert adapter.entered.is_set()
    assert elapsed < 0.5
    assert free_after == free_before
    assert _readable_keys(l1_manager, keys) == keys
    status = controller.report_status()
    assert status["sync_flush_failures"] == 0
    assert status["sync_flush_backoff_seconds"] == 0.0


class _RecordingEvictor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str | None]] = []

    def emergency_evict_bytes(
        self,
        target_free_bytes: int,
        requester: str = "",
        cache_salt: str | None = None,
    ) -> int:
        self.calls.append((target_free_bytes, requester, cache_salt))
        return target_free_bytes


@pytest.fixture
def prefetch_controller(l1_manager: L1Manager) -> Iterator[PrefetchController]:
    controller = PrefetchController(
        l1_manager=l1_manager,
        l2_adapters=[],
        adapter_descriptors=[],
        policy=DefaultPrefetchPolicy(),
    )
    yield controller
    controller._submission_efd.close()
    controller._adapter_ctrl_efd.close()


def _prefetch_request(
    keys: list[ObjectKey],
    layouts: dict[int, MemoryLayoutDesc],
    *,
    mode: PrefetchMode = PrefetchMode.LOOKUP,
) -> InFlightPrefetchRequest:
    return InFlightPrefetchRequest(
        request_id=7,
        keys=keys,
        phase=PrefetchPhase.PLAN_AND_LOAD,
        mode=mode,
        group_layout_descs=layouts,
    )


def test_prefetch_make_room_uses_each_group_layout(
    l1_manager: L1Manager,
    prefetch_controller: PrefetchController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small = MemoryLayoutDesc(shapes=[torch.Size([2])], dtypes=[torch.float32])
    large = MemoryLayoutDesc(shapes=[torch.Size([100])], dtypes=[torch.float32])
    keys = [
        ObjectKey(ObjectKey.IntHash2Bytes(100), "writeback-test", 0, 0),
        ObjectKey(ObjectKey.IntHash2Bytes(101), "writeback-test", 0, 1),
    ]
    evictor = _RecordingEvictor()
    prefetch_controller.set_l1_eviction_controller(cast(L1EvictionController, evictor))
    monkeypatch.setattr(prefetch_controller, "_PER_OBJECT_HEADROOM_BYTES", 0)
    monkeypatch.setattr(l1_manager, "get_memory_usage", lambda: (999_900, 1_000_000))

    prefetch_controller._make_room_for_restore(
        _prefetch_request(keys, {0: small, 1: large}),
        keys,
    )

    assert evictor.calls == [(408, "prefetch_request=7", "")]


def test_warm_prefetch_never_requests_emergency_eviction(
    prefetch_controller: PrefetchController,
) -> None:
    keys = _make_keys(1)
    evictor = _RecordingEvictor()
    prefetch_controller.set_l1_eviction_controller(cast(L1EvictionController, evictor))

    prefetch_controller._make_room_for_restore(
        _prefetch_request(keys, {0: OBJECT_LAYOUT}, mode=PrefetchMode.WARM),
        keys,
    )

    assert evictor.calls == []


def test_prefetch_retry_oom_reservations_by_group(
    l1_manager: L1Manager,
    prefetch_controller: PrefetchController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small = MemoryLayoutDesc(shapes=[torch.Size([2])], dtypes=[torch.float32])
    large = MemoryLayoutDesc(shapes=[torch.Size([100])], dtypes=[torch.float32])
    keys = [
        ObjectKey(ObjectKey.IntHash2Bytes(110), "writeback-test", 0, 0),
        ObjectKey(ObjectKey.IntHash2Bytes(111), "writeback-test", 0, 1),
    ]
    request = _prefetch_request(keys, {0: small, 1: large})
    evictor = _RecordingEvictor()
    prefetch_controller.set_l1_eviction_controller(cast(L1EvictionController, evictor))
    monkeypatch.setattr(l1_manager, "get_memory_usage", lambda: (999_900, 1_000_000))

    merged = prefetch_controller._retry_oom_reservations(
        request,
        keys,
        {key: True for key in keys},
        {key: (L1Error.OUT_OF_MEMORY, None) for key in keys},
    )

    assert len(evictor.calls) == 1
    assert all(merged[key][0] == L1Error.SUCCESS for key in keys)
    assert merged[keys[0]][1].get_size() < merged[keys[1]][1].get_size()  # type: ignore[union-attr]
    l1_manager.finish_write(keys)


def test_storage_manager_reconnects_emergency_evictor_on_adapter_changes(
    tmp_path,
) -> None:
    pytest.importorskip("lmcache.lmcache_fs")
    initial = FSNativeL2AdapterConfig(
        base_path=str(tmp_path / "initial"),
        num_workers=1,
        relative_tmp_dir="pending",
    )
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=POOL_BYTES,
                use_lazy=False,
                init_size_in_bytes=POOL_BYTES,
                align_bytes=0x1000,
            )
        ),
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
            trigger_watermark=1.0,
            write_back_on_evict=True,
            emergency_evict_for_prefetch=True,
        ),
        l2_adapter_config=L2AdaptersConfig(adapters=[initial]),
    )
    manager = StorageManager(config)
    try:
        assert (
            manager._prefetch_controller._l1_eviction_controller
            is manager._eviction_controller
        )

        manager.delete_l2_adapter(0)
        assert manager._prefetch_controller._l1_eviction_controller is None

        manager.add_l2_adapter(
            FSNativeL2AdapterConfig(
                base_path=str(tmp_path / "runtime"),
                num_workers=1,
                relative_tmp_dir="pending",
            )
        )
        assert (
            manager._prefetch_controller._l1_eviction_controller
            is manager._eviction_controller
        )
    finally:
        manager.close()


def test_storage_manager_emergency_prefetch_restores_under_l1_pressure(
    tmp_path,
) -> None:
    pytest.importorskip("lmcache.lmcache_fs")
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=POOL_BYTES,
                use_lazy=False,
                init_size_in_bytes=POOL_BYTES,
                align_bytes=0x1000,
            )
        ),
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
            trigger_watermark=1.0,
            write_back_on_evict=True,
            emergency_evict_for_prefetch=True,
        ),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[
                FSNativeL2AdapterConfig(
                    base_path=str(tmp_path),
                    num_workers=1,
                    relative_tmp_dir="pending",
                )
            ]
        ),
    )
    manager = StorageManager(config)
    restore_keys = _make_keys(2)
    resident_keys = _make_keys(9)[2:]
    try:
        # Keep this test specific to the emergency path. Normal StoreController
        # routing would otherwise race the manual L2 seed with its own read
        # locks and duplicate asynchronous store.
        store_detached = manager._store_controller.request_remove_adapter(0)
        assert store_detached.wait(timeout=5.0)

        restore = manager.reserve_write(restore_keys, OBJECT_LAYOUT, mode="new")
        assert list(restore) == restore_keys
        manager._l1_manager.finish_write(restore_keys)
        read = manager._l1_manager.reserve_read(restore_keys)
        restore_objs = [read[key][1] for key in restore_keys]
        assert all(obj is not None for obj in restore_objs)
        adapter = manager._l2_adapters[0]
        sync_store = adapter.store_objects_sync  # type: ignore[attr-defined]
        ok, persisted, _ = sync_store(
            restore_keys,
            restore_objs,  # type: ignore[arg-type]
            timeout=5.0,
        )
        assert ok and persisted == len(restore_keys)
        manager._l1_manager.finish_read(restore_keys)
        assert all(
            error == L1Error.SUCCESS
            for error in manager._l1_manager.delete(restore_keys).values()
        )

        resident = manager.reserve_write(resident_keys, OBJECT_LAYOUT, mode="new")
        assert list(resident) == resident_keys
        manager._l1_manager.finish_write(resident_keys)
        resident_before = set(_readable_keys(manager._l1_manager, resident_keys))

        handle = manager.submit_prefetch_task(
            PrefetchRequestSpec(restore_keys, {0: OBJECT_LAYOUT})
        )
        assert manager.wait_prefetch_status(handle, timeout=5.0)
        result = manager.query_prefetch_status(handle)

        assert result is not None and result.count_leading_ones() == 2
        assert all(
            entry[0] == L1Error.SUCCESS
            for entry in manager._l1_manager.unsafe_read(restore_keys).values()
        )
        assert set(_readable_keys(manager._l1_manager, resident_keys)) < resident_before
        manager.finish_read_prefetched(restore_keys)
    finally:
        manager.close()
