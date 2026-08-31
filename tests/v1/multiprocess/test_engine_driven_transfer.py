# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast
from unittest.mock import MagicMock, PropertyMock, patch
import os
import pickle
import sys

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.posix_shm import (
    shm_create_readwrite,
    shm_munmap,
    shm_open_pool_as_mmap,
    shm_unlink,
)
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContextMetadata,
    create_engine_driven_context,
)
from lmcache.v1.multiprocess.transfer_context.pickle import EngineDrivenContextPickle
from lmcache.v1.multiprocess.transfer_context.shm import EngineDrivenContextShm
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.config import StorageManagerConfig
    from lmcache.v1.gpu_connector.utils import LayoutHints
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheServerKey,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
        EngineDrivenTransferModule,
    )


class ServerModuleFactory(Protocol):
    """Typed callable contract for creating patched server test modules.

    Args:
        storage_manager_config: Optional engine storage config override.
        chunk_size: Engine chunk size used to initialize the context.
        object_keys: Object keys returned by ``ipc_key_to_object_keys``.
        mock_storage: Optional storage mock; defaults to a new ``MagicMock``.
        mock_session: Optional session mock; defaults to a new ``MagicMock``.

    Returns a tuple of ``(EngineDrivenTransferModule, storage MagicMock,
    session MagicMock, MPCacheServerContext)``.
    """

    def __call__(
        self,
        *,
        storage_manager_config: "StorageManagerConfig | None" = None,
        chunk_size: int = 8,
        object_keys: list[str] | None = None,
        mock_storage: MagicMock | None = None,
        mock_session: MagicMock | None = None,
    ) -> tuple[
        "EngineDrivenTransferModule", MagicMock, MagicMock, "MPCacheServerContext"
    ]: ...


def _make_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Build per-layer NHD KV tensors for device-agnostic data transfer tests."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            2, num_blocks, block_size, num_heads, head_size
        )
    return kv_caches


def _make_mla_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    hidden_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Build per-layer MLA KV tensors for device-agnostic data transfer tests.

    Args:
        num_layers: Number of KV layers to generate.
        num_blocks: Number of paged blocks per layer.
        block_size: Number of tokens per block.
        hidden_size: Hidden size per token.

    Returns:
        Mapping from layer name to MLA KV tensor with shape
        ``[num_blocks, block_size, hidden_size]``.
    """
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(num_blocks, block_size, hidden_size)
    return kv_caches


def _make_hnd_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Build per-layer HND KV tensors for device-agnostic data transfer tests."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            2, num_blocks, num_heads, block_size, head_size
        )
    return kv_caches


def _make_hnd_flashinfer_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Build per-layer HND flash-infer KV tensors for
    device-agnostic data transfer tests.
    """
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            num_blocks, 2, num_heads, block_size, head_size
        )
    return kv_caches


def _make_fused_hnd_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Build per-layer blocks-first fused-K/V HND tensors ([NB, NH, BS, 2*HS])."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            num_blocks, num_heads, block_size, 2 * head_size
        )
    return kv_caches


def _make_fused_nhd_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Build per-layer blocks-first fused-K/V NHD tensors ([NB, BS, NH, 2*HS])."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            num_blocks, block_size, num_heads, 2 * head_size
        )
    return kv_caches


def _make_storage_manager_config(
    *,
    shm_name: str = "",
    pool_size: int = 4096,
    use_lazy: bool = False,
) -> Any:
    """Build a StorageManagerConfig for multiprocess engine-context tests."""
    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )

    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=pool_size,
                use_lazy=use_lazy,
                shm_name=shm_name,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )


def _default_register_payload(
    instance_id: int = 1,
) -> "RegisterEngineDrivenContextPayload":
    """Build a default non-GPU registration payload for server-side tests.

    Args:
        instance_id: Worker instance id to register. Defaults to ``1``.

    Uses fixed values ``model_name="m"``, ``world_size=1``, ``block_size=4``,
    ``num_layers=2``, ``hidden_dim_size=16``, ``dtype_str="float32"``, and
    ``use_mla=False`` for a compact baseline scenario used by most tests.
    """
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload

    return RegisterEngineDrivenContextPayload(
        instance_id=instance_id,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
    )


def _grouped_register_payload(
    instance_id: int = 20,
) -> "RegisterEngineDrivenContextPayload":
    """Build a two-group registration with different planes and windows."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        EngineDrivenGroupLayout,
        RegisterEngineDrivenContextPayload,
    )

    return RegisterEngineDrivenContextPayload(
        instance_id=instance_id,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=1,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        group_layouts=(
            EngineDrivenGroupLayout(
                object_group_id=0,
                engine_group_idx=1,
                layer_indices=(0,),
                tokens_per_block=4,
                blocks_per_chunk=2,
                shape=(2, 1, 8, 16),
                dtype_str="float32",
                blocks_per_window=2,
                group_kind="attention",
                num_chunks_in_window=-1,
            ),
            EngineDrivenGroupLayout(
                object_group_id=1,
                engine_group_idx=0,
                layer_indices=(1,),
                tokens_per_block=2,
                blocks_per_chunk=4,
                shape=(1, 2, 8),
                dtype_str="float32",
                blocks_per_window=1,
                group_kind="recurrent",
                num_chunks_in_window=1,
            ),
        ),
    )


def test_engine_driven_registration_payload_decodes_legacy_wire_shape() -> None:
    """The trailing group field must not break an old eight-field payload."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload

    encoded = msgspec.msgpack.encode(
        {
            "instance_id": 1,
            "model_name": "m",
            "world_size": 1,
            "block_size": 4,
            "num_layers": 2,
            "hidden_dim_size": 16,
            "dtype_str": "float32",
            "use_mla": False,
        }
    )
    payload = msgspec.msgpack.decode(encoded, type=RegisterEngineDrivenContextPayload)

    assert payload.group_layouts == ()


def test_engine_driven_registration_response_defaults_capability_off() -> None:
    """Omitted server capabilities default to fail-closed values."""
    response = RegisterEngineDrivenContextResponse(shm_name="pool", pool_size=8)

    assert response.accepts_group_layouts is False
    assert response.accepts_store_abort is False


def test_engine_driven_registration_response_is_wire_compatible() -> None:
    """Old and new response maps decode safely in both directions."""
    old_wire = msgspec.msgpack.encode({"shm_name": "pool", "pool_size": 8})
    current = msgspec.msgpack.decode(old_wire, type=RegisterEngineDrivenContextResponse)
    assert current.accepts_group_layouts is False
    assert current.accepts_store_abort is False

    legacy_type = msgspec.defstruct(
        "LegacyRegisterEngineDrivenContextResponse",
        [("shm_name", str, ""), ("pool_size", int, 0)],
    )
    current_wire = msgspec.msgpack.encode(
        RegisterEngineDrivenContextResponse(
            shm_name="pool",
            pool_size=8,
            accepts_group_layouts=True,
            accepts_store_abort=True,
        )
    )
    legacy = msgspec.msgpack.decode(current_wire, type=legacy_type)
    legacy_any = cast(Any, legacy)
    assert legacy_any.shm_name == "pool"
    assert legacy_any.pool_size == 8


def _default_key(tokens: int = 8) -> "IPCCacheServerKey":
    """Build a default IPC cache key with ``tokens`` contiguous token IDs.

    Args:
        tokens: Total token count and key end offset. Defaults to ``8``.

    Uses fixed values ``model_name="m"``, ``world_size=1``, ``rank=0``,
    token IDs of ``[1] * tokens``, ``start=0``, ``end=tokens``,
    and ``request_id="req"``.
    """
    # First Party
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

    return IPCCacheServerKey.from_token_ids(
        "m",
        1,
        0,
        [1] * tokens,
        start=0,
        end=tokens,
        request_id="req",
    )


def test_wrap_kv_caches_wraps_all_tensors() -> None:
    """Verify wrap_kv_caches wraps all provided KV tensors."""
    # First Party
    from lmcache.v1.platform import get_device_spec
    from lmcache.v1.platform.kv_wrap import wrap_kv_caches

    kv_caches = _make_kv_caches()

    # ``wrap_kv_caches`` dispatches through
    # :func:`resolve_kv_wrapper_factory`, which reads
    # ``DeviceSpec.ipc_wrapper_cls`` for each device. Substitute a fake
    # wrapper class per relevant spec so the test doesn't require the
    # real IPC-backed factories to be usable in the harness.
    class _FakeWrapper:
        @classmethod
        def wrap(cls, tensor: Any) -> tuple[str, Any]:
            return ("wrapped", tensor)

    with ExitStack() as stack:
        for device_type in {t.device.type for t in kv_caches.values()}:
            spec = get_device_spec(device_type)
            assert spec is not None, "no DeviceSpec registered for %r" % device_type
            stack.enter_context(
                patch.object(
                    type(spec),
                    "ipc_wrapper_cls",
                    new_callable=PropertyMock,
                    return_value=_FakeWrapper,
                )
            )
        wrapped = wrap_kv_caches(kv_caches)

    assert len(wrapped) == len(kv_caches)


def test_create_transfer_context_uses_default_context_on_cpu() -> None:
    """Ensure factory returns EngineDrivenTransferContext for CPU KV."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        EngineDrivenTransferContext,
        create_transfer_context,
    )

    context = create_transfer_context({"layer_0": torch.randn(2, 2)})
    assert isinstance(context, EngineDrivenTransferContext)


def test_resolve_extra_config_default_mp_transfer_mode_is_auto() -> None:
    """Without override the resolved mp_transfer_mode must be ``auto``."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        ExtraConfigDefault,
        _resolve_extra_config,
    )

    cfg = _resolve_extra_config(None)
    assert cfg[ExtraConfigDefault.mp_transfer_mode.name] == "auto"


def test_resolve_extra_config_overrides_mp_transfer_mode() -> None:
    """``lmcache.mp.mp_transfer_mode`` override flows through unchanged."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        ExtraConfigDefault,
        _resolve_extra_config,
    )

    cfg = _resolve_extra_config({"lmcache.mp.mp_transfer_mode": "lmcache_driven"})
    assert cfg[ExtraConfigDefault.mp_transfer_mode.name] == "lmcache_driven"


def test_extra_config_default_lets_env_var_select_mp_transfer_mode(
    monkeypatch: Any,
) -> None:
    """When extra_config omits mp_transfer_mode, env var must still win.

    The adapter detects the absence of ``lmcache.mp.mp_transfer_mode`` and
    passes ``mode=None`` to ``create_transfer_context``, which then reads
    the ``LMCACHE_MP_TRANSFER_MODE`` env var. Regression test for
    buildkite k3-multiprocess CI ``cpu_e2e_validation (server-side copy)``.
    """
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        _EXTRA_CONFIG_KEY_PREFIX,
        ExtraConfigDefault,
    )
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        create_transfer_context,
    )
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        ENV_MP_TRANSFER_MODE,
    )

    mp_mode_key = _EXTRA_CONFIG_KEY_PREFIX + ExtraConfigDefault.mp_transfer_mode.name
    # Simulate adapter init: extra_config omits the mp_transfer_mode key.
    extra_config: dict[str, Any] = {"lmcache.mp.mq_timeout": "1"}
    resolved_mode = extra_config[mp_mode_key] if mp_mode_key in extra_config else None
    assert resolved_mode is None

    # With env=engine_driven and mode=None, CPU KV must pick
    # EngineDrivenTransferContext.
    monkeypatch.setenv(ENV_MP_TRANSFER_MODE, "engine_driven")
    context = create_transfer_context(
        {"layer_0": torch.randn(2, 2)}, mode=resolved_mode
    )
    assert isinstance(context, EngineDrivenTransferContext)


def test_create_transfer_context_force_lmcache_driven_mode() -> None:
    """``mode='lmcache_driven'`` must always pick
    LMCacheDrivenTransferContext (handle path); CPU also works because the
    CPU SHM wrapper factory is registered on import."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        LMCacheDrivenTransferContext,
        MPTransferMode,
        create_transfer_context,
    )

    # Importing the CPU sub-package self-registers its KV-wrapper factory.
    import lmcache.v1.platform.cpu  # noqa: F401

    context = create_transfer_context(
        {"layer_0": torch.randn(2, 2)}, mode=MPTransferMode.LMCACHE_DRIVEN
    )
    assert isinstance(context, LMCacheDrivenTransferContext)


def test_create_transfer_context_force_engine_driven_mode_on_cpu() -> None:
    """``mode='engine_driven'`` on CPU returns EngineDrivenTransferContext
    (data path; no wrapper-factory capability check is performed)."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        create_transfer_context,
    )

    context = create_transfer_context(
        {"layer_0": torch.randn(2, 2)}, mode="engine_driven"
    )
    assert isinstance(context, EngineDrivenTransferContext)


def test_create_transfer_context_invalid_mode_raises() -> None:
    """Unknown mode strings must raise a clear ValueError."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import create_transfer_context

    with pytest.raises(ValueError, match="Invalid MP transfer mode"):
        create_transfer_context({"layer_0": torch.randn(2, 2)}, mode="bogus")


def test_create_transfer_context_handle_mode_unsupported_device_raises(
    monkeypatch: Any,
) -> None:
    """``mode='lmcache_driven'`` must raise when no wrapper factory exists
    for the device."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import create_transfer_context
    from lmcache.v1.platform import get_device_spec

    cpu_spec = get_device_spec("cpu")
    assert cpu_spec is not None
    # Strip the wrapper binding so ``resolve_kv_wrapper_factory('cpu')``
    # raises, mirroring the historical "empty registry" fixture.
    monkeypatch.setattr(
        type(cpu_spec),
        "ipc_wrapper_cls",
        property(lambda self: None),
    )
    with pytest.raises(ValueError, match="not supported for device type"):
        create_transfer_context({"layer_0": torch.randn(2, 2)}, mode="lmcache_driven")


@pytest.mark.musa
def test_musa_data_context_keeps_layout_validation_device_agnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUSA MP data path must not put device layout gates in transfer context."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )

    def _fake_compute_kv_layout(
        *_args: Any, **_kwargs: Any
    ) -> tuple[int, int, int, str, Any, int]:
        return (
            4,
            2,
            16,
            "float32",
            lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
            2,
        )

    monkeypatch.setattr(worker_transfer, "compute_kv_layout", _fake_compute_kv_layout)
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *_args, **_kwargs: MagicMock(),
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    ctx = EngineDrivenTransferContext()

    ctx.register(
        instance_id=1,
        kv_caches=_make_hnd_kv_caches(),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
    )


def test_engine_driven_hybrid_registration_requires_server_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new hybrid worker must fail closed against an old server response."""
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )

    monkeypatch.setattr(
        worker_transfer,
        "compute_kv_layout",
        lambda *_args, **_kwargs: (
            4,
            1,
            16,
            "float32",
            lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            2,
        ),
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()

    with pytest.raises(RuntimeError, match="did not acknowledge"):
        EngineDrivenTransferContext().register(
            instance_id=1,
            kv_caches=_make_kv_caches(num_layers=2),
            model_name="m",
            world_size=1,
            blocks_in_chunk=2,
            mq_client=MagicMock(),
            mq_timeout=1.0,
            send_request=MagicMock(return_value=future),
            engine_group_infos=(
                EngineGroupInfo(engine_group_id=1, layer_indices=(0,)),
                EngineGroupInfo(engine_group_id=0, layer_indices=(1,)),
            ),
        )


def test_engine_driven_shm_registration_requires_abort_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new worker must not use SHM that cannot cancel failed gathers."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )

    monkeypatch.setattr(
        worker_transfer,
        "compute_kv_layout",
        lambda *_args, **_kwargs: (
            4,
            2,
            16,
            "float32",
            lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            2,
        ),
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse(
        shm_name="pool", pool_size=8
    )

    with pytest.raises(RuntimeError, match="canceling failed SHM stores"):
        EngineDrivenTransferContext().register(
            instance_id=1,
            kv_caches=_make_kv_caches(),
            model_name="m",
            world_size=1,
            blocks_in_chunk=2,
            mq_client=MagicMock(),
            mq_timeout=1.0,
            send_request=MagicMock(return_value=future),
        )


def test_engine_driven_hybrid_registration_preserves_group_order_and_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid wire layouts use exact per-group planes, windows, and order."""
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )

    split_format = lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    fused_format = lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_TWO_HS

    def _fake_layout(
        caches: dict[str, torch.Tensor], **_kwargs: Any
    ) -> tuple[int, int, int, str, Any, int]:
        name = next(iter(caches))
        if name == "layer_1":
            return 2, 1, 8, "float32", fused_format, 1
        return 4, 1, 16, "float32", split_format, 2

    monkeypatch.setattr(worker_transfer, "compute_kv_layout", _fake_layout)
    engine_context = MagicMock()
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        MagicMock(return_value=engine_context),
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse(
        accepts_group_layouts=True
    )
    send_request = MagicMock(return_value=future)
    ctx = EngineDrivenTransferContext()

    ctx.register(
        instance_id=1,
        kv_caches=_make_kv_caches(num_layers=2),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=send_request,
        engine_group_infos=(
            EngineGroupInfo(
                engine_group_id=1,
                layer_indices=(0,),
                tokens_per_block=4,
            ),
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(1,),
                tokens_per_block=2,
                sw_size_tokens=2,
                recurrent_state=True,
            ),
        ),
    )

    payload = send_request.call_args.args[2][0]
    assert [group.engine_group_idx for group in payload.group_layouts] == [1, 0]
    assert [group.blocks_per_chunk for group in payload.group_layouts] == [2, 4]
    assert [group.blocks_per_window for group in payload.group_layouts] == [2, 1]
    assert [group.shape for group in payload.group_layouts] == [
        (2, 1, 8, 16),
        (1, 2, 8),
    ]
    assert [group.group_kind for group in payload.group_layouts] == [
        "attention",
        "recurrent",
    ]

    gathered_ids: list[list[int]] = []

    def _capture_gather(
        _caches: dict[str, torch.Tensor], ids: list[int], *_args: Any, **_kwargs: Any
    ) -> list[torch.Tensor]:
        gathered_ids.append(ids)
        return [torch.zeros(1)]

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _capture_gather)
    ctx._gather_group_payloads(  # noqa: SLF001 - focused protocol-order test
        _make_kv_caches(num_layers=2),
        [[10, 11], [20, 21, 22, 23]],
    )
    assert gathered_ids == [[10, 11], [20, 21, 22, 23]]


def test_engine_driven_hybrid_registration_uses_explicit_chunk_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Physical page size cannot inflate a hybrid LMCache chunk."""
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )

    def _fake_layout(
        caches: dict[str, torch.Tensor], **_kwargs: Any
    ) -> tuple[int, int, int, str, Any, int]:
        block_size = 8 if "layer_0" in caches else 1
        return (
            block_size,
            1,
            4,
            "float32",
            lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
            1,
        )

    monkeypatch.setattr(worker_transfer, "compute_kv_layout", _fake_layout)
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        MagicMock(return_value=MagicMock()),
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse(
        accepts_group_layouts=True
    )
    send_request = MagicMock(return_value=future)

    EngineDrivenTransferContext().register(
        instance_id=1,
        kv_caches={
            "layer_0": torch.zeros(4, 8, 4),
            "layer_1": torch.zeros(4, 1, 4),
        },
        model_name="m",
        world_size=1,
        blocks_in_chunk=8,
        tokens_per_chunk=8,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=send_request,
        engine_group_infos=(
            EngineGroupInfo(0, (0,), tokens_per_block=8),
            EngineGroupInfo(1, (1,), tokens_per_block=8),
        ),
    )

    payload = send_request.call_args.args[2][0]
    assert [group.blocks_per_chunk for group in payload.group_layouts] == [1, 1]
    assert [group.shape[-2] for group in payload.group_layouts] == [8, 1]


@pytest.mark.musa
def test_musa_data_context_store_uses_device_agnostic_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage3 store keeps MUSA native details behind block-transfer entry."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )

    class _FakeEngineDrivenContext:
        def prepare_store(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def commit_store(self, *_args: Any, **_kwargs: Any) -> bool:
            return True

        def close(self) -> None:
            return None

    captured_kwargs: dict[str, Any] = {}
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    monkeypatch.setattr(
        worker_transfer,
        "compute_kv_layout",
        lambda *_args, **_kwargs: (
            4,
            2,
            16,
            "float32",
            lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            2,
        ),
    )
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *_args, **_kwargs: _FakeEngineDrivenContext(),
    )

    def _fake_gather(*_args: Any, **kwargs: Any) -> list[torch.Tensor]:
        captured_kwargs.update(kwargs)
        return [torch.zeros(2, 2, 8, 16)]

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _fake_gather)
    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=_make_kv_caches(),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
    )

    result = ctx.submit_store(
        "req",
        _default_key(),
        1,
        _make_kv_caches(),
        [[0, 1]],
        MagicMock(),
        2,
    ).result()

    assert result is True
    assert "prefer_musa_native" not in captured_kwargs


@pytest.mark.musa
def test_musa_data_context_retrieve_uses_device_agnostic_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage3 retrieve keeps MUSA native details behind block-transfer entry."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )

    class _FakeEngineDrivenContext:
        def prepare_retrieve(self, *_args: Any, **_kwargs: Any) -> list[torch.Tensor]:
            return [torch.zeros(2, 2, 8, 16)]

        def commit_retrieve(self, *_args: Any, **_kwargs: Any) -> bool:
            return True

        def close(self) -> None:
            return None

    captured_kwargs: dict[str, Any] = {}
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    monkeypatch.setattr(
        worker_transfer,
        "compute_kv_layout",
        lambda *_args, **_kwargs: (
            4,
            2,
            16,
            "float32",
            lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            2,
        ),
    )
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *_args, **_kwargs: _FakeEngineDrivenContext(),
    )

    def _fake_scatter(*_args: Any, **kwargs: Any) -> None:
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(worker_transfer, "scatter_cpu_to_paged_kv", _fake_scatter)
    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=_make_kv_caches(),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
    )

    result = ctx.submit_retrieve(
        "req",
        _default_key(),
        1,
        _make_kv_caches(),
        [[0, 1]],
        MagicMock(),
        2,
    ).result()

    assert result is True
    assert "prefer_musa_native" not in captured_kwargs


def test_create_transfer_context_env_var_overrides_default(
    monkeypatch: Any,
) -> None:
    """``LMCACHE_MP_TRANSFER_MODE=lmcache_driven`` must force the
    LMCache-driven path."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        LMCacheDrivenTransferContext,
        create_transfer_context,
    )
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        ENV_MP_TRANSFER_MODE,
    )

    # Importing the CPU sub-package self-registers its KV-wrapper factory,
    # which is required by the lmcache-driven (handle) path.
    import lmcache.v1.platform.cpu  # noqa: F401

    monkeypatch.setenv(ENV_MP_TRANSFER_MODE, "lmcache_driven")
    context = create_transfer_context({"layer_0": torch.randn(2, 2)})
    assert isinstance(context, LMCacheDrivenTransferContext)


@pytest.mark.parametrize(
    ("builder_fn", "expected_block_size", "expected_hidden_dim", "layout_hints"),
    [
        pytest.param(
            lambda: _make_kv_caches(
                num_layers=2,
                num_blocks=8,
                block_size=4,
                num_heads=4,
                head_size=4,
            ),
            4,
            16,
            None,
            id="nhd",
        ),
        pytest.param(
            lambda: _make_mla_kv_caches(
                num_layers=2, num_blocks=8, block_size=4, hidden_size=16
            ),
            4,
            16,
            None,
            id="mla",
        ),
        pytest.param(
            lambda: _make_fused_hnd_kv_caches(
                num_layers=2, num_blocks=8, block_size=4, num_heads=2, head_size=8
            ),
            4,
            32,
            {"kv_layout": "HND"},
            id="fused_hnd",
        ),
        pytest.param(
            lambda: _make_fused_nhd_kv_caches(
                num_layers=2, num_blocks=8, block_size=4, num_heads=2, head_size=8
            ),
            4,
            32,
            {"kv_layout": "NHD"},
            id="fused_nhd",
        ),
    ],
)
def test_compute_kv_layout_and_gather_scatter_roundtrip(
    builder_fn: Callable[[], dict[str, torch.Tensor]],
    expected_block_size: int,
    expected_hidden_dim: int,
    layout_hints: "LayoutHints | None",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate layout extraction and gather/scatter round-trip on CPU tensors."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import (
        compute_kv_layout,
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    def _vllm_detector_device_type() -> str:
        """Keep the detector on the active accelerator, but bypass CPU hosts."""

        return torch_device_type if torch_device_type != "cpu" else "cuda"

    # Bypass the CPU-host HND safeguard so the layout hint drives detection
    # regardless of the host running the test.
    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type",
        _vllm_detector_device_type(),
    )

    source = {k: v.to(torch_device_type) for k, v in builder_fn().items()}
    (
        block_size,
        num_layers,
        hidden_dim,
        dtype_str,
        detected_kv_format,
        kv_size,
    ) = compute_kv_layout(source, layout_hints=layout_hints)
    assert block_size == expected_block_size
    assert num_layers == 2
    assert hidden_dim == expected_hidden_dim
    assert dtype_str == "float32"
    assert detected_kv_format is not None

    blocks_per_chunk = 2
    gathered = gather_paged_kv_to_cpu(
        source, [0, 1], blocks_per_chunk, layout_hints=layout_hints
    )
    # The gathered chunk shape must equal the layout the worker registers with
    # the server (register() builds it from kv_size and hidden_dim), or the
    # server-side commit_store shape check rejects every chunk.
    expected_chunk_shape = (
        (num_layers, blocks_per_chunk * block_size, hidden_dim)
        if kv_size == 1
        else (2, num_layers, blocks_per_chunk * block_size, hidden_dim)
    )
    assert tuple(gathered[0].shape) == expected_chunk_shape
    destination = {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    scatter_cpu_to_paged_kv(
        destination, [4, 5], gathered, blocks_per_chunk, layout_hints=layout_hints
    )

    for name in source:
        if source[name].dim() == 5:
            assert torch.allclose(source[name][:, 0], destination[name][:, 4])
            assert torch.allclose(source[name][:, 1], destination[name][:, 5])
        else:
            assert torch.allclose(source[name][0], destination[name][4])
            assert torch.allclose(source[name][1], destination[name][5])


@pytest.mark.parametrize(
    ("hnd_builder", "expected_format"),
    [
        (_make_hnd_kv_caches, "NL_X_TWO_NB_NH_BS_HS"),
        (_make_hnd_flashinfer_kv_caches, "NL_X_NB_TWO_NH_BS_HS"),
    ],
)
def test_gather_scatter_roundtrip_hnd_layout(
    hnd_builder: Callable[[int, int, int, int, int], dict[str, torch.Tensor]],
    expected_format: str,
) -> None:
    """Validate gather/scatter round-trip for HND vLLM KV layout."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import (
        compute_kv_layout,
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = {k: v.to(torch_device_type) for k, v in hnd_builder(2, 8, 4, 2, 8).items()}
    layout_hints: LayoutHints = {"kv_layout": "HND"}
    (
        block_size,
        num_layers,
        hidden_dim,
        dtype_str,
        detected_kv_format,
        _kv_size,
    ) = compute_kv_layout(source, layout_hints=layout_hints)
    assert block_size == 4
    assert num_layers == 2
    assert hidden_dim == 16
    assert dtype_str == "float32"
    assert detected_kv_format == getattr(lmcache_native.EngineKVFormat, expected_format)

    blocks_per_chunk = 2
    gathered = gather_paged_kv_to_cpu(
        source,
        [0, 1],
        blocks_per_chunk,
        layout_hints=layout_hints,
        engine_kv_format=detected_kv_format,
    )
    destination = {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    scatter_cpu_to_paged_kv(
        destination,
        [4, 5],
        gathered,
        blocks_per_chunk,
        layout_hints=layout_hints,
        engine_kv_format=detected_kv_format,
    )

    for name in source:
        if detected_kv_format == lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS:
            assert torch.allclose(source[name][:, 0], destination[name][:, 4])
            assert torch.allclose(source[name][:, 1], destination[name][:, 5])
        else:
            assert torch.allclose(source[name][0], destination[name][4])
            assert torch.allclose(source[name][1], destination[name][5])


def test_compute_kv_layout_empty_raises_value_error() -> None:
    """Ensure compute_kv_layout rejects empty KV cache input."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import compute_kv_layout

    with pytest.raises(ValueError, match="kv_caches is empty"):
        compute_kv_layout({})


@pytest.mark.parametrize(
    (
        "builder_fn",
        "skip_tokens",
        "expected_unchanged_blocks",
        "expected_copied_blocks",
    ),
    [
        pytest.param(
            lambda: _make_kv_caches(
                num_layers=2,
                num_blocks=8,
                block_size=4,
                num_heads=4,
                head_size=4,
            ),
            8,
            [0, 1],
            [2, 3],
            id="nhd-skip-two-blocks",
        ),
        pytest.param(
            lambda: _make_mla_kv_caches(
                num_layers=2, num_blocks=8, block_size=4, hidden_size=16
            ),
            8,
            [0, 1],
            [2, 3],
            id="mla-skip-two-blocks",
        ),
        pytest.param(
            lambda: _make_mla_kv_caches(
                num_layers=2, num_blocks=8, block_size=4, hidden_size=16
            ),
            40,
            [0, 1, 2, 3],
            [],
            id="mla-skip-past-chunk",
        ),
    ],
)
def test_scatter_respects_skip_first_n_tokens(
    builder_fn: Callable[[], dict[str, torch.Tensor]],
    skip_tokens: int,
    expected_unchanged_blocks: list[int],
    expected_copied_blocks: list[int],
) -> None:
    """Ensure scatter honors skip_first_n_tokens and preserves skipped blocks."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = {k: v.to(torch_device_type) for k, v in builder_fn().items()}
    destination = {
        name: torch.full_like(tensor, 999.0) for name, tensor in source.items()
    }
    gathered = gather_paged_kv_to_cpu(source, [0, 1, 2, 3], blocks_per_chunk=4)
    scatter_cpu_to_paged_kv(
        destination,
        [0, 1, 2, 3],
        gathered,
        blocks_per_chunk=4,
        skip_first_n_tokens=skip_tokens,
    )

    for name in destination:
        for block_idx in expected_unchanged_blocks:
            if destination[name].dim() == 5:
                assert torch.all(destination[name][:, block_idx] == 999.0)
            else:
                assert torch.all(destination[name][block_idx] == 999.0)
        for block_idx in expected_copied_blocks:
            if destination[name].dim() == 5:
                assert torch.allclose(
                    destination[name][:, block_idx], source[name][:, block_idx]
                )
            else:
                assert torch.allclose(
                    destination[name][block_idx],
                    source[name][block_idx],
                )


@pytest.mark.parametrize(
    ("builder_fn", "layout_hints"),
    [
        pytest.param(
            lambda: _make_hnd_kv_caches(num_layers=2, num_blocks=4, block_size=4),
            {"kv_layout": "HND"},
            id="hnd",
        ),
        pytest.param(
            lambda: _make_mla_kv_caches(
                num_layers=2, num_blocks=4, block_size=4, hidden_size=16
            ),
            None,
            id="mla",
        ),
    ],
)
def test_scatter_rounds_down_partial_block_skip_first_n_tokens(
    builder_fn: Callable[[], dict[str, torch.Tensor]],
    layout_hints: "LayoutHints | None",
) -> None:
    """Scatter rounds non-block-aligned prefix skips down to whole blocks."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = {k: v.to(torch_device_type) for k, v in builder_fn().items()}
    destination = {
        name: torch.full_like(tensor, 999.0) for name, tensor in source.items()
    }
    gathered = gather_paged_kv_to_cpu(
        source,
        [0, 1],
        blocks_per_chunk=2,
        layout_hints=layout_hints,
    )
    scatter_cpu_to_paged_kv(
        destination,
        [0, 1],
        gathered,
        blocks_per_chunk=2,
        skip_first_n_tokens=2,
        layout_hints=layout_hints,
    )

    for name in destination:
        for block_idx in (0, 1):
            if destination[name].dim() == 5:
                assert torch.allclose(
                    destination[name][:, block_idx],
                    source[name][:, block_idx],
                )
            else:
                assert torch.allclose(
                    destination[name][block_idx],
                    source[name][block_idx],
                )
        for block_idx in (2, 3):
            if destination[name].dim() == 5:
                assert torch.all(destination[name][:, block_idx] == 999.0)
            else:
                assert torch.all(destination[name][block_idx] == 999.0)


@pytest.fixture
def stub_lmcache_native() -> Any:
    """Stub native modules so server imports work in source-only test runs."""
    module = type(sys)("lmcache.lmcache_native")
    module.PageBufferShapeDesc = type("PageBufferShapeDesc", (), {})  # type: ignore[attr-defined]
    module.KernelGroupSpec = type(  # type: ignore[attr-defined]
        "KernelGroupSpec",
        (),
        {"__init__": lambda self, *args, **kwargs: None},
    )
    module.TTLLock = type("TTLLock", (), {})  # type: ignore[attr-defined]
    module.Bitmap = type("Bitmap", (), {})  # type: ignore[attr-defined]
    module.PeriodicEventNotifier = type(  # type: ignore[attr-defined]
        "PeriodicEventNotifier", (), {}
    )
    with patch.dict(
        sys.modules,
        {
            "lmcache.lmcache_native": module,
            "cupy": MagicMock(),
        },
    ):
        yield


@pytest.fixture
def server_module_factory(
    stub_lmcache_native: Any,
) -> Iterator[ServerModuleFactory]:
    """Create a patched server module/context with configurable mocks."""
    # Standard
    from contextlib import ExitStack

    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
        EngineDrivenTransferModule,
    )

    stack = ExitStack()

    def _create(
        *,
        storage_manager_config: "StorageManagerConfig | None" = None,
        chunk_size: int = 8,
        object_keys: list[str] | None = None,
        mock_storage: MagicMock | None = None,
        mock_session: MagicMock | None = None,
    ) -> tuple[
        "EngineDrivenTransferModule", MagicMock, MagicMock, "MPCacheServerContext"
    ]:
        """Create a patched module/context plus storage/session mocks.

        Args:
            storage_manager_config: Optional engine storage config override.
            chunk_size: Engine chunk size passed to context construction.
            object_keys: Keys returned from ``ipc_key_to_object_keys`` patch.
            mock_storage: Optional storage mock instance to inject.
            mock_session: Optional session mock instance to inject.

        Returns ``(module, mock_storage, mock_session, ctx)``.
        """
        mock_storage = mock_storage or MagicMock()
        # First Party
        from lmcache.v1.distributed.admission import AdmissionWaitResult
        from lmcache.v1.distributed.error import L1Error

        def _reserve_write_detailed(
            obj_keys: list[str], *args: Any, **kwargs: Any
        ) -> dict[str, tuple[Any, Any]]:
            reserved = mock_storage.reserve_write(obj_keys, *args, **kwargs)
            return {
                obj_key: (
                    (L1Error.SUCCESS, reserved[obj_key])
                    if obj_key in reserved
                    else (L1Error.KEY_NOT_WRITABLE, None)
                )
                for obj_key in obj_keys
            }

        mock_storage.reserve_write_detailed.side_effect = _reserve_write_detailed
        mock_storage.get_readable_keys.side_effect = lambda keys: list(keys)
        mock_storage.get_capacity_generation.return_value = 0
        mock_storage.wait_for_capacity_change.return_value = (
            AdmissionWaitResult.TIMEOUT,
            0,
        )
        mock_storage.store_admission_timeout_seconds = 0.0
        if mock_session is None:
            mock_session = MagicMock()
            mock_session.get_hashes.return_value = [b"h"]

        stack.enter_context(
            patch(
                "lmcache.v1.multiprocess.engine_context.StorageManager",
                return_value=mock_storage,
            )
        )
        stack.enter_context(patch("lmcache.v1.multiprocess.engine_context.TokenHasher"))
        session_cls = stack.enter_context(
            patch("lmcache.v1.multiprocess.engine_context.SessionManager")
        )
        stack.enter_context(
            patch("lmcache.v1.multiprocess.engine_context.get_event_bus")
        )
        stack.enter_context(
            patch(
                "lmcache.v1.multiprocess.engine_context.ipc_key_to_object_keys",
                return_value=[object_keys or ["obj"]],
            )
        )

        session_cls.return_value.get_or_create.return_value = mock_session
        if storage_manager_config is None:
            storage_manager_config = MagicMock()
            # GDS L1 is off in these tests. A bare MagicMock would auto-vivify
            # gds_l1_config to a truthy mock, making MPCacheServerContext attempt
            # real cuFile init; pin it to None so GDS init stays a no-op.
            storage_manager_config.l1_manager_config.gds_l1_config = None
        ctx = MPCacheServerContext(
            storage_manager_config=storage_manager_config,
            chunk_size=chunk_size,
        )
        module = EngineDrivenTransferModule(ctx)

        return module, mock_storage, mock_session, ctx

    yield _create  # type: ignore[misc]
    stack.close()


@pytest.mark.parametrize(
    ("config_kwargs", "expected_pool_info"),
    [
        pytest.param(
            {"shm_name": "/test_pool", "pool_size": 1024},
            {"shm_name": "lmcache_l1_pool_test_pool", "pool_size": 1024},
            id="non-lazy",
        ),
        pytest.param(
            {
                "shm_name": "lmcache_l1_pool_existing",
                "pool_size": 2048,
                "use_lazy": True,
            },
            {"shm_name": "", "pool_size": 0},
            id="lazy",
        ),
    ],
)
def test_engine_context_shm_pool_info(
    stub_lmcache_native: Any,
    config_kwargs: dict[str, Any],
    expected_pool_info: dict[str, Any],
) -> None:
    """Ensure engine context computes SHM pool metadata for lazy and non-lazy modes."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext

    with patch(
        "lmcache.v1.distributed.config.current_device_spec",
        MagicMock(is_pin_supported=True),
    ):
        config = _make_storage_manager_config(**config_kwargs)

    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = MPCacheServerContext(storage_manager_config=config, chunk_size=16)

    assert ctx.shm_pool_info == expected_pool_info


def test_server_register_and_find_non_cuda_context_layout(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Ensure backend-agnostic registration stores metadata and lookup finds layout."""
    module, _, _, ctx = server_module_factory(chunk_size=16)
    response = module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=1)
    )
    assert response.shm_name == ""
    assert response.pool_size == 0

    layout = ctx.layout_desc_registry.find("m", 1)
    assert layout is not None
    assert layout.shapes[0] == torch.Size([2, 2, 16, 16])


def test_server_registers_exact_hybrid_group_layouts_and_windows(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Grouped registration populates current-dev layout and lookup metadata."""
    module, _, _, ctx = server_module_factory(chunk_size=8)

    response = module.register_kv_cache_engine_driven_context(
        _grouped_register_payload()
    )

    assert response.accepts_group_layouts is True
    assert response.accepts_store_abort is True
    layouts = ctx.layout_desc_registry.find_group_layout_descs("m", 1)
    assert layouts is not None
    assert layouts[0].shapes == [torch.Size([2, 1, 8, 16])]
    assert layouts[1].shapes == [torch.Size([1, 2, 8])]
    attn_desc = ctx.layout_desc_registry.find_attn_desc("m", 1)
    assert attn_desc.num_chunks_in_sw == [-1, 1]
    assert attn_desc.group_kinds == ("attention", "recurrent")


def test_server_rejects_group_geometry_that_disagrees_with_chunk_size(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Registration must fail before mismatched chunks can share cache keys."""
    module, _, _, _ = server_module_factory(chunk_size=16)

    with pytest.raises(ValueError, match="server uses 16"):
        module.register_kv_cache_engine_driven_context(_grouped_register_payload())


def test_server_rejects_group_shape_that_disagrees_with_layers(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Wire shapes must retain a supported plane and layer structure."""
    payload = _grouped_register_payload()
    payload = msgspec.structs.replace(
        payload,
        group_layouts=(
            msgspec.structs.replace(payload.group_layouts[0], shape=(3, 1, 8, 16)),
            payload.group_layouts[1],
        ),
    )
    module, _, _, _ = server_module_factory(chunk_size=8)

    with pytest.raises(ValueError, match="Invalid engine-driven group layout"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_allows_multiple_layouts_from_one_engine_group(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Physical transfer splits may legitimately share one block-id space."""
    payload = _grouped_register_payload()
    payload = msgspec.structs.replace(
        payload,
        group_layouts=tuple(
            msgspec.structs.replace(group, engine_group_idx=0)
            for group in payload.group_layouts
        ),
    )
    module, _, _, ctx = server_module_factory(chunk_size=8)

    response = module.register_kv_cache_engine_driven_context(payload)

    assert response.accepts_group_layouts is True
    layouts = ctx.layout_desc_registry.find_group_layout_descs("m", 1)
    assert layouts is not None
    assert set(layouts) == {0, 1}


def test_server_pickle_roundtrip_requires_every_hybrid_group(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Pickle transport stores and retrieves one complete payload per group."""
    mock_storage = MagicMock()
    group_tensors = {
        "g0": torch.zeros(2, 1, 8, 16),
        "g1": torch.zeros(1, 2, 8),
    }
    memory_objs = {}
    for object_key, tensor in group_tensors.items():
        memory_obj = MagicMock()
        memory_obj.tensor = tensor
        memory_objs[object_key] = memory_obj

    def _reserve(keys: list[str], *_args: Any) -> dict[str, Any]:
        return {key: memory_objs[key] for key in keys}

    @contextmanager
    def _read(keys: list[str]) -> Any:
        yield [memory_objs[key] for key in keys]

    mock_storage.reserve_write.side_effect = _reserve
    mock_storage.read_prefetched_results.side_effect = _read
    module, _, _, ctx = server_module_factory(chunk_size=8, mock_storage=mock_storage)
    module.register_kv_cache_engine_driven_context(_grouped_register_payload())
    cast(Any, ctx).resolve_obj_keys = MagicMock(return_value=[["g0"], ["g1"]])
    source = [
        [torch.ones_like(group_tensors["g0"])],
        [torch.full_like(group_tensors["g1"], 2.0)],
    ]
    key = _default_key()

    assert module.commit_store(key, 20, pickle.dumps(source)) is True
    assert torch.equal(group_tensors["g0"], source[0][0])
    assert torch.equal(group_tensors["g1"], source[1][0])

    response = module.prepare_retrieve(key, 20)
    assert response.success is True
    recovered = pickle.loads(response.data)
    assert len(recovered) == 2
    assert torch.equal(recovered[0][0], source[0][0])
    assert torch.equal(recovered[1][0], source[1][0])


def test_server_pickle_rejects_invalid_group_before_reserving(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Malformed group payloads cannot allocate or publish cache objects."""
    mock_storage = MagicMock()
    module, _, _, ctx = server_module_factory(chunk_size=8, mock_storage=mock_storage)
    module.register_kv_cache_engine_driven_context(_grouped_register_payload())
    cast(Any, ctx).resolve_obj_keys = MagicMock(return_value=[["g0"], ["g1"]])
    invalid = [
        [torch.zeros(1)],
        [torch.zeros(1, 2, 8)],
    ]

    assert module.commit_store(_default_key(), 20, pickle.dumps(invalid)) is False
    mock_storage.reserve_write.assert_not_called()
    mock_storage.finish_write.assert_not_called()


def test_server_pickle_aborts_every_reservation_when_copy_fails(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """A failed CPU copy leaves no write reservation readable."""
    mock_storage = MagicMock()
    target = MagicMock()
    target.shape = torch.Size([2, 1, 8, 16])
    target.dtype = torch.float32
    target.copy_.side_effect = RuntimeError("copy failed")
    memory_obj = MagicMock()
    memory_obj.tensor = target
    mock_storage.reserve_write.return_value = {"g0": memory_obj}
    module, _, _, ctx = server_module_factory(chunk_size=8, mock_storage=mock_storage)
    module.register_kv_cache_engine_driven_context(_grouped_register_payload())
    cast(Any, ctx).resolve_obj_keys = MagicMock(return_value=[["g0"], ["g1"]])
    source = [
        [torch.zeros(2, 1, 8, 16)],
        [torch.zeros(1, 2, 8)],
    ]

    assert module.commit_store(_default_key(), 20, pickle.dumps(source)) is False
    mock_storage.abort_write.assert_called_once_with(["g0"])
    mock_storage.finish_write.assert_not_called()


def test_server_store_and_retrieve_cpu_chunks(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Validate mocked server-side CPU chunk store and retrieve behavior."""
    mock_storage = MagicMock()
    target_tensor = torch.zeros(2, 2, 8, 16)
    mock_memory_obj = MagicMock()
    mock_memory_obj.tensor = target_tensor
    mock_storage.reserve_write.return_value = {"obj": mock_memory_obj}

    @contextmanager
    def _read_prefetched_results(_keys: Any) -> Any:
        yield [mock_memory_obj]

    mock_storage.read_prefetched_results.side_effect = _read_prefetched_results
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]
    module, _, _, _ = server_module_factory(
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=2)
    )
    payload = torch.ones(2, 2, 8, 16)
    key = _default_key()
    store_ok = module.commit_store(key, 2, pickle.dumps([payload]))
    response = module.prepare_retrieve(key, 2)
    success = response.success
    cpu_data = response.data

    assert isinstance(store_ok, bool)
    assert torch.allclose(mock_memory_obj.tensor, payload)

    assert success is True
    recovered_chunks: list[torch.Tensor] = pickle.loads(cpu_data)
    assert len(recovered_chunks) == 1
    assert torch.allclose(recovered_chunks[0], payload)


def test_server_shm_commit_store_allows_noop_when_all_keys_exist(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Regression: repeated prompt after worker restart should no-op-store cleanly.

    When all object keys already exist in cache, SHM ``prepare_store`` reserves
    no new objects and returns empty slots (``{"slots": [], "chunk_indices": []}``).
    The worker sees an empty chunk_indices list, skips gather and commit entirely,
    so no entry leaks in ``_pending_shm_writes`` and no spurious error is logged.
    """
    mock_storage = MagicMock()
    # Empty reserve_write indicates all object keys already exist in cache.
    mock_storage.reserve_write.return_value = {}
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]

    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=1024
        ),
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=3)
    )
    key = _default_key()
    prepare_response = module.prepare_store(key, 3)
    # Server signals all-cached via empty slots list (not missing "slots" key).
    assert prepare_response.context == {"slots": [], "chunk_indices": []}

    # commit_store without a matching prepare must fail (no entry leaked).
    assert module.commit_store(key, 3, b"") is False


def test_server_prepare_store_rejects_reserved_object_without_tensor(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """SHM admission aborts a reserved object with no writable tensor."""
    # First Party
    from lmcache.v1.multiprocess.protocols.engine import PrepareStoreResponse

    mock_storage = MagicMock()
    memory_obj = MagicMock()
    memory_obj.tensor = None
    mock_storage.reserve_write.side_effect = lambda obj_keys, *_args, **_kwargs: {
        obj_key: memory_obj for obj_key in obj_keys
    }
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]

    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=1024
        ),
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=5)
    )
    key = _default_key()
    prepare_response = module.prepare_store(key, 5)
    assert isinstance(prepare_response, PrepareStoreResponse)
    assert prepare_response.context == {
        "success": False,
        "failure_reason": "invalid_layout",
    }
    reserved_keys = mock_storage.reserve_write.call_args[0][0]
    mock_storage.abort_write.assert_called_once_with(reserved_keys)
    mock_storage.finish_write.assert_not_called()


def test_server_shm_transport_uses_engine_level_config(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Ensure all instances share the same engine-level SHM transport setting."""
    mock_storage = MagicMock()
    mock_memory_obj = MagicMock()
    mock_memory_obj.tensor = torch.zeros(2, 2, 8, 16)
    mock_memory_obj.shm_offset = 0
    mock_memory_obj.shm_byte_length = 2048
    mock_storage.reserve_write.side_effect = lambda obj_keys, *_args, **_kwargs: {
        obj_key: mock_memory_obj for obj_key in obj_keys
    }
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]

    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=1024
        ),
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=6)
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=7)
    )
    key = _default_key()
    assert module.prepare_store(key, 6).context.get("slots")
    assert module.prepare_store(key, 7).context.get("slots")
    assert mock_storage.reserve_write.call_count == 2


def test_server_engine_driven_reregister_returns_existing_shm_response(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Ensure duplicate non-GPU registration returns existing SHM response."""
    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=2048
        ),
    )
    payload = _default_register_payload(instance_id=8)
    first_response = module.register_kv_cache_engine_driven_context(payload)
    second_response = module.register_kv_cache_engine_driven_context(payload)

    assert first_response.shm_name == "lmcache_l1_pool_lmcache_test_pool"
    assert first_response.pool_size == 2048
    assert second_response.shm_name == "lmcache_l1_pool_lmcache_test_pool"
    assert second_response.pool_size == 2048


def test_server_unregister_engine_driven_context_releases_pending_shm_locks(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Ensure unregister releases pending SHM read/write reservations."""
    mock_storage = MagicMock()
    mock_memory_obj = MagicMock()
    mock_memory_obj.tensor = torch.zeros(2, 2, 8, 16)
    mock_memory_obj.shm_offset = 0
    mock_memory_obj.shm_byte_length = 2048
    mock_storage.reserve_write.side_effect = lambda obj_keys, *_args, **_kwargs: {
        obj_key: mock_memory_obj for obj_key in obj_keys
    }
    mock_storage.unsafe_read.side_effect = lambda obj_keys: (
        obj_keys,
        [mock_memory_obj for _ in obj_keys],
    )
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]

    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=4096
        ),
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=4)
    )
    key = _default_key()
    assert module.prepare_store(key, 4).context.get("slots")
    assert module.prepare_retrieve(key, 4).success is True

    module.unregister_kv_cache(4)

    mock_storage.abort_write.assert_called_once()
    mock_storage.finish_read_prefetched.assert_called_once()


def test_server_close_releases_pending_shm_locks(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Server shutdown aborts orphaned writes and releases reads."""
    mock_storage = MagicMock()
    memory_obj = MagicMock()
    memory_obj.tensor = torch.zeros(2, 2, 8, 16)
    memory_obj.shm_offset = 0
    memory_obj.shm_byte_length = 2048
    mock_storage.reserve_write.side_effect = lambda obj_keys, *_args, **_kwargs: {
        obj_key: memory_obj for obj_key in obj_keys
    }
    mock_storage.unsafe_read.side_effect = lambda obj_keys: (
        obj_keys,
        [memory_obj for _ in obj_keys],
    )
    module, _, _, ctx = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=4096
        ),
        mock_storage=mock_storage,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=14)
    )
    key = _default_key()
    assert module.prepare_store(key, 14).context["slots"]
    assert module.prepare_retrieve(key, 14).success is True

    module.close()

    mock_storage.abort_write.assert_called_once()
    mock_storage.finish_read_prefetched.assert_called_once()
    assert ctx.layout_desc_registry.find("m", 1) is None


def test_gather_paged_kv_with_chunk_indices_subset() -> None:
    """gather_paged_kv_to_cpu with chunk_indices only gathers the specified chunks.

    This tests the fix for the IndexError that occurred when SHM prepare_store
    returned fewer slots than total chunks because some chunks already existed
    in cache.
    """
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu

    # 3 chunks (6 blocks, 2 blocks per chunk), but we only want chunks 0 and 2
    source = {
        k: v.to(torch_device_type)
        for k, v in _make_kv_caches(
            num_layers=2,
            num_blocks=6,
            block_size=4,
            num_heads=4,
            head_size=4,
        ).items()
    }
    blocks_per_chunk = 2
    # Pre-allocate output buffers for chunks 0 and 2 only (2 tensors, not 3).
    # Shape: [2, num_layers, chunk_tokens, hidden_dim] where
    # chunk_tokens = blocks_per_chunk * block_size = 2 * 4 = 8.
    out0 = torch.zeros(2, 2, 8, 16)
    out2 = torch.zeros(2, 2, 8, 16)
    out_buffers = [out0, out2]

    # With chunk_indices=[0, 2], gather only chunks at positions 0 and 2
    # block_ids has 6 entries: [0,1] for chunk 0, [2,3] for chunk 1, [4,5] for chunk 2
    result = gather_paged_kv_to_cpu(
        source,
        block_ids=[0, 1, 2, 3, 4, 5],
        blocks_per_chunk=blocks_per_chunk,
        out=out_buffers,
        chunk_indices=[0, 2],
    )
    torch_dev.synchronize()
    # Result should be the same list as out_buffers (in-place fill)
    assert result is out_buffers

    # out_buffers[0] should contain chunk 0 (blocks 0,1) data
    # out_buffers[1] should contain chunk 2 (blocks 4,5) data
    # Verify by independently gathering all chunks and comparing
    all_chunks = gather_paged_kv_to_cpu(source, [0, 1, 2, 3, 4, 5], blocks_per_chunk)
    torch_dev.synchronize()

    assert torch.allclose(out_buffers[0], all_chunks[0])
    assert torch.allclose(out_buffers[1], all_chunks[2])


def test_gather_scatter_uses_trailing_subchunk_window() -> None:
    """A sub-chunk group transfers only each logical chunk's trailing blocks."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = {
        key: value.to(torch_device_type)
        for key, value in _make_kv_caches(
            num_layers=1,
            num_blocks=8,
            block_size=4,
            num_heads=2,
            head_size=4,
        ).items()
    }
    chunks = gather_paged_kv_to_cpu(
        source,
        [0, 1, 2, 3],
        blocks_per_chunk=2,
        blocks_per_window=1,
    )
    full_chunks = gather_paged_kv_to_cpu(
        source,
        [0, 1, 2, 3],
        blocks_per_chunk=2,
    )
    assert chunks[0].numel() * 2 == full_chunks[0].numel()
    destination = {
        name: torch.full_like(tensor, 999.0) for name, tensor in source.items()
    }

    scatter_cpu_to_paged_kv(
        destination,
        [4, 5, 6, 7],
        chunks,
        blocks_per_chunk=2,
        blocks_per_window=1,
    )

    for name in source:
        assert torch.all(destination[name][:, 4] == 999.0)
        assert torch.all(destination[name][:, 6] == 999.0)
        assert torch.allclose(destination[name][:, 5], source[name][:, 1])
        assert torch.allclose(destination[name][:, 7], source[name][:, 3])


def test_recurrent_alias_collapse_keeps_only_newest_snapshot() -> None:
    """Repeated destination IDs must not race old and new recurrent states."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        _collapse_chunks_for_single_destination,
    )

    chunks = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]

    compact_chunks, compact_ids = _collapse_chunks_for_single_destination(
        chunks,
        [7, 7, 7],
        blocks_per_chunk=1,
        blocks_per_window=1,
    )

    assert compact_chunks == [chunks[-1]]
    assert compact_ids == [7]


def test_server_prepare_store_includes_chunk_indices(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """prepare_store response context includes chunk_indices for SHM mode.

    Regression test: the server must return the positional indices of the
    reserved chunks so the client only gathers KV data for those chunks.
    """
    mock_storage = MagicMock()
    obj1 = "obj1"
    obj2 = "obj2"
    mock_memory_obj = MagicMock()
    mock_memory_obj.tensor = torch.zeros(2, 2, 8, 16)
    mock_memory_obj.shm_offset = 0
    mock_memory_obj.shm_byte_length = 2048
    # Only obj2 (index 1) is reserved; obj1 (index 0) already exists in cache.
    mock_storage.reserve_write.return_value = {obj2: mock_memory_obj}
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h1", b"h2"]

    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=4096
        ),
        object_keys=[obj1, obj2],
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=10)
    )
    key = _default_key(tokens=16)
    response = module.prepare_store(key, 10)
    response_context = response.context

    # slots should have 1 entry (only obj2 reserved)
    assert len(response_context.get("slots", [])) == 1
    # chunk_indices should be [1] (position of obj2 in [obj1, obj2])
    assert response_context.get("chunk_indices") == [1]


def test_server_grouped_shm_labels_slots_and_releases_all_group_miss(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """SHM descriptors retain group identity and a later miss unlocks earlier groups."""
    mock_storage = MagicMock()
    memory_objs = []
    for offset, shape in ((0, (2, 1, 8, 16)), (4096, (1, 2, 8))):
        memory_obj = MagicMock()
        memory_obj.tensor = torch.zeros(shape)
        memory_obj.shm_offset = offset
        memory_obj.shm_byte_length = memory_obj.tensor.numel() * 4
        memory_objs.append(memory_obj)
    mock_storage.reserve_write.side_effect = [
        {"g0": memory_objs[0]},
        {"g1": memory_objs[1]},
    ]
    module, _, _, ctx = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_group_pool", pool_size=16384
        ),
        chunk_size=8,
        mock_storage=mock_storage,
    )
    module.register_kv_cache_engine_driven_context(_grouped_register_payload())
    cast(Any, ctx).resolve_obj_keys = MagicMock(return_value=[["g0"], ["g1"]])
    key = _default_key()

    response = module.prepare_store(key, 20)
    assert response.context["group_ids"] == [0, 1]
    assert response.context["chunk_indices"] == [0, 0]
    assert len(response.context["slots"]) == 2

    mock_storage.unsafe_read.side_effect = [(["g0"], [memory_objs[0]]), ([], [])]
    miss = module.prepare_retrieve(key, 20)
    assert miss.success is False
    mock_storage.finish_read_prefetched.assert_called_once_with(["g0"])


def test_server_grouped_shm_capacity_failure_aborts_every_group(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """A partial multi-group allocation is never exposed as a cache store."""
    # First Party
    from lmcache.v1.distributed.error import L1Error

    mock_storage = MagicMock()
    memory_obj = MagicMock()
    memory_obj.tensor = torch.zeros(2, 1, 8, 16)
    memory_obj.shm_offset = 0
    memory_obj.shm_byte_length = memory_obj.tensor.numel() * 4
    module, _, _, ctx = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_group_pool", pool_size=16384
        ),
        chunk_size=8,
        mock_storage=mock_storage,
    )
    mock_storage.reserve_write_detailed.side_effect = [
        {"g0": (L1Error.SUCCESS, memory_obj)},
        {"g1": (L1Error.OUT_OF_MEMORY, None)},
    ]
    mock_storage.get_readable_keys.return_value = []
    mock_storage.get_readable_keys.side_effect = None
    mock_storage.store_admission_timeout_seconds = 0.0
    module.register_kv_cache_engine_driven_context(_grouped_register_payload())
    cast(Any, ctx).resolve_obj_keys = MagicMock(return_value=[["g0"], ["g1"]])

    response = module.prepare_store(_default_key(), 20)

    assert response.context == {
        "success": False,
        "failure_reason": "capacity_timeout",
    }
    mock_storage.abort_write.assert_called_once_with(["g0"])
    mock_storage.finish_write.assert_not_called()


def test_server_grouped_shm_abort_discards_pending_reservations(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """The abort sentinel removes SHM reservations without publishing them."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        ENGINE_DRIVEN_ABORT_STORE_PAYLOAD,
    )

    mock_storage = MagicMock()
    memory_objs = []
    for offset, shape in ((0, (2, 1, 8, 16)), (4096, (1, 2, 8))):
        memory_obj = MagicMock()
        memory_obj.tensor = torch.zeros(shape)
        memory_obj.shm_offset = offset
        memory_obj.shm_byte_length = memory_obj.tensor.numel() * 4
        memory_objs.append(memory_obj)
    mock_storage.reserve_write.side_effect = [
        {"g0": memory_objs[0]},
        {"g1": memory_objs[1]},
    ]
    module, _, _, ctx = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_group_pool", pool_size=16384
        ),
        chunk_size=8,
        mock_storage=mock_storage,
    )
    module.register_kv_cache_engine_driven_context(_grouped_register_payload())
    cast(Any, ctx).resolve_obj_keys = MagicMock(return_value=[["g0"], ["g1"]])
    key = _default_key()

    assert module.prepare_store(key, 20).context["slots"]
    assert module.commit_store(key, 20, ENGINE_DRIVEN_ABORT_STORE_PAYLOAD) is True
    mock_storage.abort_write.assert_called_once_with(["g0", "g1"])
    mock_storage.finish_write.assert_not_called()


def test_worker_grouped_gather_failure_aborts_prepared_shm_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker gather error cannot strand or publish its prepared SHM slots."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )

    engine_context = MagicMock()
    engine_context.prepare_store_grouped.return_value = (
        [[torch.zeros(1)], [torch.zeros(1)]],
        [[0], [0]],
    )
    context = EngineDrivenTransferContext()
    context._engine_driven_context = engine_context  # noqa: SLF001
    context._worker_groups = (MagicMock(), MagicMock())  # noqa: SLF001
    monkeypatch.setattr(
        context,
        "_gather_group_payloads",
        MagicMock(side_effect=RuntimeError("gather failed")),
    )
    monkeypatch.setattr(worker_transfer.torch_dev, "synchronize", MagicMock())
    key = _default_key()

    result = context.submit_store(
        "req",
        key,
        20,
        _make_kv_caches(),
        [[0], [1]],
        MagicMock(),
        1,
    ).result()

    assert result is False
    engine_context.abort_store.assert_called_once_with(key, 20)
    engine_context.commit_store_grouped.assert_not_called()


def test_worker_grouped_scatter_failure_releases_retrieve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed destination write still releases every server read lock."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import EngineDrivenTransferContext

    engine_context = MagicMock()
    engine_context.prepare_retrieve_grouped.return_value = [
        [torch.zeros(1)],
        [torch.zeros(1)],
    ]
    engine_context.commit_retrieve.return_value = True
    context = EngineDrivenTransferContext()
    context._engine_driven_context = engine_context  # noqa: SLF001
    context._worker_groups = (MagicMock(), MagicMock())  # noqa: SLF001
    monkeypatch.setattr(
        context,
        "_scatter_group_payloads",
        MagicMock(side_effect=RuntimeError("scatter failed")),
    )
    key = _default_key()

    result = context.submit_retrieve(
        "req",
        key,
        20,
        _make_kv_caches(),
        [[0], [1]],
        MagicMock(),
        1,
    ).result()

    assert result is False
    engine_context.commit_retrieve.assert_called_once_with(key, 20)


class _CompletedFuture:
    def __init__(self, value):
        self._value = value

    def wait(self, timeout=None):  # noqa: ARG002
        return True

    def result(self, timeout=None):  # noqa: ARG002
        return self._value


def _create_shm_segment(shm_name: str, size: int) -> int:
    """Create a POSIX SHM segment via the project facade.

    Returns the owner mmap address so the test can release the segment
    with ``shm_munmap`` + ``shm_unlink`` regardless of platform
    (Linux/macOS), instead of hard-coding ``/dev/shm`` paths.
    """
    return shm_create_readwrite(shm_name, size)


def test_engine_driven_context_shm_tensor_view_from_buffer() -> None:
    shm_name = f"lmcache_test_view_{os.getpid()}"
    addr = _create_shm_segment(shm_name, 4096)
    try:
        mm = shm_open_pool_as_mmap(shm_name, 4096)
        try:
            src = torch.arange(8, dtype=torch.float32).reshape(2, 4)
            mm[: src.numel() * src.element_size()] = src.numpy().tobytes()
        finally:
            mm.close()

        context = EngineDrivenContextShm(
            metadata=EngineDrivenContextMetadata(
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([2, 4])],
                    dtypes=[torch.float32],
                ),
                block_size=1,
                use_mla=False,
            ),
            mq_client=MagicMock(),
            mq_timeout=1.0,
            shm_name=shm_name,
            pool_size=4096,
        )
        try:
            view = context._make_tensor_view(
                offset=0,
                length=src.numel() * src.element_size(),
                shape=[2, 4],
                dtype_str="float32",
            )
            assert torch.equal(view, src)
        finally:
            context.close()
    finally:
        shm_munmap(addr, 4096)
        shm_unlink(shm_name)


def test_engine_driven_context_shm_store_retrieve_flow_with_mocked_mq() -> None:
    shm_name = f"lmcache_test_flow_{os.getpid()}"
    addr = _create_shm_segment(shm_name, 4096)
    slots = [
        {
            "offset": 0,
            "length": 16,
            "shape": [2, 2],
            "dtype": "float32",
        }
    ]

    mq_client = MagicMock()

    def _submit_request(req_type, payload, response_cls):  # noqa: ARG001
        if req_type == RequestType.PREPARE_STORE:
            return _CompletedFuture(
                PrepareStoreResponse(context={"slots": slots, "chunk_indices": [0]})
            )
        if req_type == RequestType.COMMIT_STORE:
            _, _, commit_cpu_data = payload
            assert commit_cpu_data == b""
            return _CompletedFuture(True)
        if req_type == RequestType.PREPARE_RETRIEVE:
            return _CompletedFuture(
                PrepareRetrieveResponse(
                    success=True, data=b"", context={"slots": slots}
                )
            )
        if req_type == RequestType.COMMIT_RETRIEVE:
            return _CompletedFuture(True)
        raise AssertionError(f"Unexpected request type: {req_type}")

    mq_client.submit_request.side_effect = _submit_request

    context = EngineDrivenContextShm(
        metadata=EngineDrivenContextMetadata(
            layout_desc=MemoryLayoutDesc(
                shapes=[torch.Size([2, 2])],
                dtypes=[torch.float32],
            ),
            block_size=1,
            use_mla=False,
        ),
        mq_client=mq_client,
        mq_timeout=1.0,
        shm_name=shm_name,
        pool_size=4096,
    )
    try:
        key = _default_key()
        store_result = context.prepare_store(key=key, instance_id=1)
        assert store_result is not None
        store_views, _ = store_result
        store_views[0].copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        )
        assert context.commit_store(key, 1, store_views)

        retrieve_views = context.prepare_retrieve(key=key, instance_id=1)
        assert retrieve_views is not None
        assert torch.equal(
            retrieve_views[0],
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        )
        assert context.commit_retrieve(key, 1)
    finally:
        context.close()
        shm_munmap(addr, 4096)
        shm_unlink(shm_name)


def test_engine_driven_context_shm_init_raises_when_segment_missing() -> None:
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        EngineDrivenContextShm(
            metadata=EngineDrivenContextMetadata(
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([2, 2])],
                    dtypes=[torch.float32],
                ),
                block_size=1,
                use_mla=False,
            ),
            mq_client=MagicMock(),
            mq_timeout=1.0,
            shm_name="lmcache_missing_shm_segment",
            pool_size=4096,
        )


def test_create_engine_driven_context_falls_back_to_pickle_without_shm_info() -> None:
    context = create_engine_driven_context(
        metadata=EngineDrivenContextMetadata(
            layout_desc=MemoryLayoutDesc(
                shapes=[torch.Size([2, 2])],
                dtypes=[torch.float32],
            ),
            block_size=1,
            use_mla=False,
        ),
        mq_client=MagicMock(),
        mq_timeout=1.0,
        shm_name="",
        pool_size=0,
    )
    assert isinstance(context, EngineDrivenContextPickle)


def test_create_engine_driven_context_use_pickle_ignores_valid_shm_info() -> None:
    context = create_engine_driven_context(
        metadata=EngineDrivenContextMetadata(
            layout_desc=MemoryLayoutDesc(
                shapes=[torch.Size([2, 2])],
                dtypes=[torch.float32],
            ),
            block_size=1,
            use_mla=False,
        ),
        mq_client=MagicMock(),
        mq_timeout=1.0,
        shm_name="lmcache_valid_shm",
        pool_size=4096,
        use_pickle=True,
    )
    assert isinstance(context, EngineDrivenContextPickle)


def test_engine_driven_context_shm_close_is_idempotent() -> None:
    shm_name = f"lmcache_test_close_{os.getpid()}"
    addr = _create_shm_segment(shm_name, 4096)
    try:
        context = EngineDrivenContextShm(
            metadata=EngineDrivenContextMetadata(
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([2, 2])],
                    dtypes=[torch.float32],
                ),
                block_size=1,
                use_mla=False,
            ),
            mq_client=MagicMock(),
            mq_timeout=1.0,
            shm_name=shm_name,
            pool_size=4096,
        )
        context.close()
        context.close()
    finally:
        shm_munmap(addr, 4096)
        shm_unlink(shm_name)
