# SPDX-License-Identifier: Apache-2.0
"""Transfer strategy implementations for non-GPU transport paths."""

# Standard
from _thread import LockType
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
import abc
import pickle

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.admission import (
    AdmissionAttempt,
    AdmissionFailure,
    AdmissionOutcome,
    reserve_with_eviction_backpressure,
)
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.multiprocess.custom_types import (
    ENGINE_DRIVEN_ABORT_STORE_PAYLOAD,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.shm import ShmSlotDescriptor

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

logger = init_logger(__name__)

ObjectKeyGroups = list[list[ObjectKey]]
ResolveObjectKeyGroups = Callable[[IPCCacheServerKey], ObjectKeyGroups]


def _dtype_to_name(dtype: torch.dtype) -> str:
    """Return a stable torch dtype name without module prefix."""
    return str(dtype).split(".")[-1]


def _attempt_group_writes(
    storage_manager: "StorageManager",
    obj_key_groups: ObjectKeyGroups,
    context: EngineDrivenContextMetadata,
) -> AdmissionAttempt[list[dict[ObjectKey, Any]]]:
    """Reserve every new group object or abort the entire attempt."""
    layouts = context.effective_group_layouts
    if len(layouts) != len(obj_key_groups):
        return AdmissionAttempt.failure(AdmissionFailure.INVALID_LAYOUT)

    reserved_by_group: list[dict[ObjectKey, Any]] = []
    all_reserved_keys: list[ObjectKey] = []
    for layout, obj_keys in zip(layouts, obj_key_groups, strict=True):
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size(layout.shape)],
            dtypes=[getattr(torch, layout.dtype_str)],
        )
        detailed = storage_manager.reserve_write_detailed(obj_keys, layout_desc, "new")
        reserved = {
            obj_key: memory_obj
            for obj_key, (_error, memory_obj) in detailed.items()
            if memory_obj is not None
        }
        reserved_by_group.append(reserved)
        all_reserved_keys.extend(reserved)

        missing = [obj_key for obj_key in obj_keys if obj_key not in reserved]
        readable = set(storage_manager.get_readable_keys(missing))
        unresolved = [obj_key for obj_key in missing if obj_key not in readable]
        if unresolved:
            storage_manager.abort_write(all_reserved_keys)
            capacity_only = all(
                detailed.get(obj_key, (L1Error.KEY_NOT_WRITABLE, None))[0]
                is L1Error.OUT_OF_MEMORY
                for obj_key in unresolved
            )
            return AdmissionAttempt.failure(
                AdmissionFailure.CAPACITY
                if capacity_only
                else AdmissionFailure.CONFLICT
            )
    return AdmissionAttempt.success(reserved_by_group)


def _reserve_group_writes(
    storage_manager: "StorageManager",
    obj_key_groups: ObjectKeyGroups,
    context: EngineDrivenContextMetadata,
) -> AdmissionOutcome[list[dict[ObjectKey, Any]]]:
    """Reserve all groups with bounded capacity-only eviction backpressure."""
    outcome = reserve_with_eviction_backpressure(
        attempt=lambda: _attempt_group_writes(storage_manager, obj_key_groups, context),
        get_generation=storage_manager.get_capacity_generation,
        request_eviction=storage_manager.request_immediate_eviction,
        wait_for_change=storage_manager.wait_for_capacity_change,
        timeout_seconds=storage_manager.store_admission_timeout_seconds,
        on_wait=storage_manager.record_admission_wait,
        on_retry=storage_manager.record_admission_retry,
        on_success_after_eviction=(
            storage_manager.record_admission_success_after_eviction
        ),
        on_timeout=storage_manager.record_admission_timeout,
    )
    if outcome.failure is not None:
        logger.warning(
            "Atomic engine-driven store admission failed: reason=%s retries=%d",
            outcome.failure.value,
            outcome.retries,
        )
    return outcome


def create_transfer_strategy(
    storage_manager: "StorageManager",
    *,
    shm_name: str,
    pool_size: int,
    pending_writes: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
    pending_reads: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
    pending_lock: LockType,
    transfer_key_factory: Callable[
        [IPCCacheServerKey, int], tuple[int, IPCCacheServerKey]
    ],
) -> "TransferStrategy":
    """Create the non-GPU transfer strategy for a registered context.

    Args:
        storage_manager: Storage manager used by the selected strategy.
        shm_name: Shared-memory pool name advertised to workers.
        pool_size: Shared-memory pool size in bytes.
        pending_writes: Map of pending SHM write reservations keyed by transfer key.
        pending_reads: Map of pending SHM read reservations keyed by transfer key.
        pending_lock: Lock guarding shared pending SHM reservation state.
        transfer_key_factory: Factory that builds the `(instance_id, key)` lookup key
            used in the pending SHM reservation maps.

    Returns:
        ``ShmTransferStrategy`` when SHM is configured with a non-empty pool name and
        positive pool size, otherwise ``PickleTransferStrategy``.
    """
    if shm_name and pool_size > 0:
        logger.info("Using shm non-GPU transfer strategy")
        return ShmTransferStrategy(
            storage_manager=storage_manager,
            pending_writes=pending_writes,
            pending_reads=pending_reads,
            pending_lock=pending_lock,
            transfer_key_factory=transfer_key_factory,
            fallback_strategy=PickleTransferStrategy(storage_manager),
        )

    logger.info("Using pickle non-GPU transfer strategy")
    return PickleTransferStrategy(storage_manager)


class TransferStrategy(abc.ABC):
    """Contract for non-GPU transport backends used by the server.

    Implementations encapsulate the transport-specific prepare/commit lifecycle for
    store and retrieve operations, allowing the server to use either pickle-based or
    shared-memory-based transfers behind a common interface.
    """

    @abc.abstractmethod
    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjectKeyGroups,
    ) -> PrepareStoreResponse:
        """Prepare destination resources for a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            Transport-specific store preparation response.
        """

    @abc.abstractmethod
    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjectKeyGroups,
    ) -> bool:
        """Finalize a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            cpu_data: Serialized payload from the worker.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            ``True`` when the strategy successfully commits the store request.
        """

    @abc.abstractmethod
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: ResolveObjectKeyGroups,
    ) -> PrepareRetrieveResponse:
        """Prepare source resources for a retrieve request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            Transport-specific retrieve preparation response.
        """

    @abc.abstractmethod
    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.

        Returns:
            ``True`` when retrieve finalization succeeds.
        """


class PickleTransferStrategy(TransferStrategy):
    """Pickle-based transport for non-GPU transfer requests.

    This is the default transport when SHM is unavailable, and it is also used as a
    fallback by the SHM strategy when the worker sends an inline serialized payload.
    ``prepare_store`` returns an empty context, while ``commit_store`` deserializes
    the pickle payload and writes the resulting tensors into reserved objects.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
    ) -> None:
        """Initialize pickle transfer strategy.

        Args:
            storage_manager: Storage manager used for reserve/read/finish calls.
        """
        self._storage_manager = storage_manager

    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjectKeyGroups,
    ) -> PrepareStoreResponse:
        """Return empty store context for pickle mode.

        Pickle transport does not pre-allocate SHM slots during prepare.
        """
        return PrepareStoreResponse(context={})

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjectKeyGroups,
    ) -> bool:
        """Deserialize and write pickled chunks into reserved objects.

        Returns:
            ``True`` when every reserved object is written successfully.
        """
        obj_key_groups = resolve_obj_keys(key)
        try:
            decoded = pickle.loads(cpu_data)
        except (pickle.PickleError, EOFError, AttributeError, ValueError, TypeError):
            logger.exception("Invalid engine-driven pickle store payload")
            return False
        grouped = len(obj_key_groups) > 1
        if not isinstance(decoded, list):
            return False
        chunk_groups: list[list[torch.Tensor]] = decoded if grouped else [decoded]
        if len(chunk_groups) != len(obj_key_groups):
            return False

        layouts = context.effective_group_layouts
        if len(layouts) != len(obj_key_groups):
            return False
        for obj_keys, chunks, layout in zip(
            obj_key_groups, chunk_groups, layouts, strict=True
        ):
            if not isinstance(chunks, list) or len(chunks) != len(obj_keys):
                return False
            expected_shape = torch.Size(layout.shape)
            expected_dtype = getattr(torch, layout.dtype_str, None)
            if not isinstance(expected_dtype, torch.dtype) or any(
                not isinstance(chunk, torch.Tensor)
                or chunk.device.type != "cpu"
                or chunk.shape != expected_shape
                or chunk.dtype != expected_dtype
                for chunk in chunks
            ):
                return False

        admission = _reserve_group_writes(
            self._storage_manager, obj_key_groups, context
        )
        if admission.failure is not None or admission.value is None:
            return False
        reserved_by_group = admission.value
        reserved_keys = [
            obj_key for reserved in reserved_by_group for obj_key in reserved
        ]
        try:
            for obj_keys, chunks, reserved_dict in zip(
                obj_key_groups, chunk_groups, reserved_by_group, strict=True
            ):
                for idx, obj_key in enumerate(obj_keys):
                    memory_obj = reserved_dict.get(obj_key)
                    if memory_obj is None:
                        continue
                    if memory_obj.tensor is None:
                        raise RuntimeError(
                            f"reserved object {obj_key} has no tensor storage"
                        )
                    chunk_cpu = chunks[idx]
                    if (
                        chunk_cpu.shape != memory_obj.tensor.shape
                        or chunk_cpu.dtype != memory_obj.tensor.dtype
                    ):
                        raise RuntimeError(
                            f"reserved object {obj_key} layout changed after "
                            "payload validation"
                        )
                    memory_obj.tensor.copy_(chunk_cpu)
        except (RuntimeError, ValueError, TypeError):
            logger.exception("Failed to copy engine-driven pickle store payload")
            if reserved_keys:
                self._storage_manager.abort_write(reserved_keys)
            return False

        if reserved_keys:
            self._storage_manager.finish_write(reserved_keys)
        return True

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: ResolveObjectKeyGroups,
    ) -> PrepareRetrieveResponse:
        """Read prefetched objects and return serialized pickle payload."""
        obj_key_groups = resolve_obj_keys(key)
        prefetched_keys: list[ObjectKey] = []
        try:
            chunk_groups: list[list[torch.Tensor]] = []
            for obj_keys in obj_key_groups:
                read_ctx = self._storage_manager.read_prefetched_results(obj_keys)
                with read_ctx as maybe_memory_objs:
                    if not maybe_memory_objs or len(maybe_memory_objs) != len(obj_keys):
                        return PrepareRetrieveResponse(
                            success=False, data=b"", context={}
                        )
                    prefetched_keys.extend(obj_keys)
                    chunks = []
                    for memory_obj in maybe_memory_objs:
                        if memory_obj.tensor is None:
                            return PrepareRetrieveResponse(
                                success=False, data=b"", context={}
                            )
                        chunks.append(memory_obj.tensor.cpu().clone())
                    chunk_groups.append(chunks)
            payload: object = (
                chunk_groups if len(obj_key_groups) > 1 else chunk_groups[0]
            )
            return PrepareRetrieveResponse(
                success=True, data=pickle.dumps(payload), context={}
            )
        finally:
            if prefetched_keys:
                self._storage_manager.finish_read_prefetched(prefetched_keys)

    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """No-op for pickle mode; data was already copied during prepare."""
        return True


class ShmTransferStrategy(TransferStrategy):
    """Shared-memory transport for non-GPU transfer requests.

    This strategy exposes SHM slot descriptors during ``prepare_store`` and
    ``prepare_retrieve`` so workers can access storage buffers directly. It tracks
    pending SHM reservations until the matching commit step releases them, and it
    falls back to pickle-based commit handling when ``cpu_data`` is non-empty.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
        pending_writes: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
        pending_reads: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
        pending_lock: LockType,
        transfer_key_factory: Callable[
            [IPCCacheServerKey, int], tuple[int, IPCCacheServerKey]
        ],
        fallback_strategy: PickleTransferStrategy,
    ) -> None:
        """Initialize SHM transfer strategy.

        Args:
            storage_manager: Storage manager used for reserve/read/finish calls.
            pending_writes: Shared pending SHM write reservations map.
            pending_reads: Shared pending SHM read reservations map.
            pending_lock: Lock guarding shared pending SHM maps.
            transfer_key_factory: Factory to build `(instance_id, key)` transfer keys.
            fallback_strategy: Pickle fallback for non-empty ``cpu_data`` payloads.
        """
        self._storage_manager = storage_manager
        self._pending_writes = pending_writes
        self._pending_reads = pending_reads
        self._pending_lock = pending_lock
        self._transfer_key_factory = transfer_key_factory
        self._fallback_strategy = fallback_strategy

    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjectKeyGroups,
    ) -> PrepareStoreResponse:
        """Reserve SHM-backed objects and return slot descriptors.

        Returns:
            Context with ``slots`` and ``chunk_indices``.
        """
        obj_key_groups = resolve_obj_keys(key)
        grouped = len(obj_key_groups) > 1
        slots: list[dict[str, Any]] = []
        chunk_indices: list[int] = []
        group_ids: list[int] = []
        admission = _reserve_group_writes(
            self._storage_manager, obj_key_groups, context
        )
        if admission.failure is not None or admission.value is None:
            reason = (
                admission.failure.value
                if admission.failure is not None
                else AdmissionFailure.CONFLICT.value
            )
            return PrepareStoreResponse(
                context={"success": False, "failure_reason": reason}
            )
        reserved_by_group = admission.value
        reserved_keys = [
            obj_key for reserved in reserved_by_group for obj_key in reserved
        ]
        for group_idx, (obj_keys, reserved) in enumerate(
            zip(obj_key_groups, reserved_by_group, strict=True)
        ):
            for idx, obj_key in enumerate(obj_keys):
                memory_obj = reserved.get(obj_key)
                if memory_obj is None:
                    continue
                if memory_obj.tensor is None:
                    self._storage_manager.abort_write(reserved_keys)
                    return PrepareStoreResponse(
                        context={
                            "success": False,
                            "failure_reason": AdmissionFailure.INVALID_LAYOUT.value,
                        }
                    )
                slots.append(
                    ShmSlotDescriptor(
                        offset=memory_obj.shm_offset,
                        length=memory_obj.shm_byte_length,
                        shape=list(memory_obj.tensor.shape),
                        dtype=_dtype_to_name(memory_obj.tensor.dtype),
                    ).to_dict()
                )
                chunk_indices.append(idx)
                group_ids.append(group_idx)
        if not reserved_keys:
            response_context: dict[str, Any] = {
                "slots": [],
                "chunk_indices": [],
            }
            if grouped:
                response_context["group_ids"] = []
            return PrepareStoreResponse(context=response_context)
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            duplicate = transfer_key in self._pending_writes
            if not duplicate:
                self._pending_writes[transfer_key] = reserved_keys
        if duplicate:
            self._storage_manager.abort_write(reserved_keys)
            raise RuntimeError("duplicate pending engine-driven SHM store")
        response_context = {"slots": slots, "chunk_indices": chunk_indices}
        if grouped:
            response_context["group_ids"] = group_ids
        return PrepareStoreResponse(context=response_context)

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjectKeyGroups,
    ) -> bool:
        """Finalize SHM store write locks or fallback to pickle commit.

        Returns:
            ``True`` when pending SHM reservation is committed successfully.
        """
        transfer_key = self._transfer_key_factory(key, instance_id)
        if cpu_data == ENGINE_DRIVEN_ABORT_STORE_PAYLOAD:
            with self._pending_lock:
                reserved_keys = self._pending_writes.pop(transfer_key, None)
            if reserved_keys is None:
                return False
            self._storage_manager.abort_write(reserved_keys)
            return True
        if cpu_data != b"":
            with self._pending_lock:
                reserved_keys = self._pending_writes.pop(transfer_key, None)
            if reserved_keys:
                self._storage_manager.abort_write(reserved_keys)
            return self._fallback_strategy.commit_store(
                key=key,
                instance_id=instance_id,
                cpu_data=cpu_data,
                context=context,
                resolve_obj_keys=resolve_obj_keys,
            )
        with self._pending_lock:
            reserved_keys = self._pending_writes.pop(transfer_key, None)
        if reserved_keys is None:
            return False
        if reserved_keys:
            self._storage_manager.finish_write(reserved_keys)
        return True

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: ResolveObjectKeyGroups,
    ) -> PrepareRetrieveResponse:
        """Read SHM objects and return slot descriptors for worker access."""
        obj_key_groups = resolve_obj_keys(key)
        grouped = len(obj_key_groups) > 1
        slots: list[dict[str, Any]] = []
        group_ids: list[int] = []
        all_prefetched_keys: list[ObjectKey] = []
        for group_idx, obj_keys in enumerate(obj_key_groups):
            prefetched_keys, memory_objs = self._storage_manager.unsafe_read(obj_keys)
            if (
                not memory_objs
                or len(prefetched_keys) != len(obj_keys)
                or len(memory_objs) != len(obj_keys)
            ):
                if prefetched_keys:
                    self._storage_manager.finish_read_prefetched(prefetched_keys)
                if all_prefetched_keys:
                    self._storage_manager.finish_read_prefetched(all_prefetched_keys)
                return PrepareRetrieveResponse(success=False, data=b"", context={})
            all_prefetched_keys.extend(prefetched_keys)
            for memory_obj in memory_objs:
                if memory_obj.tensor is None:
                    self._storage_manager.finish_read_prefetched(all_prefetched_keys)
                    return PrepareRetrieveResponse(success=False, data=b"", context={})
                slots.append(
                    ShmSlotDescriptor(
                        offset=memory_obj.shm_offset,
                        length=memory_obj.shm_byte_length,
                        shape=list(memory_obj.tensor.shape),
                        dtype=_dtype_to_name(memory_obj.tensor.dtype),
                    ).to_dict()
                )
                group_ids.append(group_idx)
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            duplicate = transfer_key in self._pending_reads
            if not duplicate:
                self._pending_reads[transfer_key] = all_prefetched_keys
        if duplicate:
            self._storage_manager.finish_read_prefetched(all_prefetched_keys)
            raise RuntimeError("duplicate pending engine-driven SHM retrieve")
        response_context: dict[str, Any] = {"slots": slots}
        if grouped:
            response_context["group_ids"] = group_ids
        return PrepareRetrieveResponse(success=True, data=b"", context=response_context)

    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Release pending SHM read locks for the completed retrieve request."""
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            prefetched_keys = self._pending_reads.pop(transfer_key, [])
        if prefetched_keys:
            self._storage_manager.finish_read_prefetched(prefetched_keys)
        return True
