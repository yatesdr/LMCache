# SPDX-License-Identifier: Apache-2.0
"""Fail-closed L1 writeback eviction tests.

The core writeback contract is deliberately independent of periodic backup and
emergency prefetch pressure: when enabled, an L1 eviction batch is deleted only
after one synchronous L2 adapter reports every readable key durable.
"""

# Standard
from collections.abc import Iterator
import argparse
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
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
) -> L1EvictionController:
    return L1EvictionController(
        l1_manager=manager,
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
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
