# SPDX-License-Identifier: Apache-2.0
"""
Filesystem native L2 adapter config and factory.

Backed by the native C++ filesystem connector wrapped with
``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Optional
import os

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _filename_to_object_key,
)

logger = init_logger(__name__)

_FILE_EXT = ".data"
_IGNORED_FILE_SAMPLE_LIMIT = 5


def _scan_existing_key_sizes(base_path: str) -> dict[ObjectKey, int]:
    """Inventory complete native-FS objects before the client starts.

    Only direct, regular, positive-size ``.data`` files with filenames
    reversible by the current ObjectKey codec are counted. Entries are returned
    oldest-to-newest by ``(mtime_ns, filename)`` so an LRU policy can reconstruct
    a deterministic best-effort order. The cache directory is expected to be
    quiescent (or exclusively owned by this server) during startup: a file that
    disappears during the scan is skipped, while every other I/O error fails
    construction instead of silently undercounting capacity.
    """
    records: list[tuple[int, str, ObjectKey, int]] = []
    ignored_data_files: list[str] = []
    ignored_data_count = 0

    try:
        entries = os.scandir(base_path)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise RuntimeError(
            f"Failed to scan native FS cache directory {base_path!r}: {exc}"
        ) from exc

    try:
        with entries:
            for entry in entries:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to inspect native FS cache entry {entry.path!r}: {exc}"
                    ) from exc

                key = _filename_to_object_key(entry.name)
                if key is None:
                    if entry.name.endswith(_FILE_EXT):
                        ignored_data_count += 1
                        if len(ignored_data_files) < _IGNORED_FILE_SAMPLE_LIMIT:
                            ignored_data_files.append(entry.name)
                    continue

                try:
                    stat = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # A concurrent cleanup can remove an entry between listing
                    # and stat. No controller is connected to this adapter yet.
                    continue
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to stat native FS cache entry {entry.path!r}: {exc}"
                    ) from exc

                if stat.st_size <= 0:
                    ignored_data_count += 1
                    if len(ignored_data_files) < _IGNORED_FILE_SAMPLE_LIMIT:
                        ignored_data_files.append(entry.name)
                    continue
                records.append((stat.st_mtime_ns, entry.name, key, stat.st_size))
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(
            f"Failed while scanning native FS cache directory {base_path!r}: {exc}"
        ) from exc

    if ignored_data_count:
        logger.warning(
            "Ignored %d unrecognized or empty native FS .data files during "
            "restart inventory (sample=%s)",
            ignored_data_count,
            ignored_data_files,
        )

    records.sort(key=lambda record: (record[0], record[1]))
    key_sizes: dict[ObjectKey, int] = {}
    for _mtime_ns, filename, key, size in records:
        if key in key_sizes:
            raise RuntimeError(
                "Native FS restart inventory found multiple filenames for the "
                f"same ObjectKey (latest={filename!r})"
            )
        key_sizes[key] = size
    return key_sizes


class FSNativeL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++
    filesystem connector.

    Fields:
    - base_path: directory for storing KV cache files.
    - num_workers: C++ worker threads for I/O (default 4).
    - relative_tmp_dir: relative sub-dir for temp files.
    - use_odirect: bypass page cache via O_DIRECT.
    - read_ahead_size: trigger filesystem readahead by
      reading this many bytes first (optional).
    """

    def __init__(
        self,
        base_path: str,
        num_workers: int = 4,
        relative_tmp_dir: str = "",
        use_odirect: bool = False,
        read_ahead_size: Optional[int] = None,
        max_capacity_gb: float = 0,
    ):
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "FSNativeL2AdapterConfig":
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        relative_tmp_dir = d.get("relative_tmp_dir", "")
        if not isinstance(relative_tmp_dir, str):
            raise ValueError("relative_tmp_dir must be a string")

        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")

        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")

        return cls(
            base_path=base_path,
            num_workers=num_workers,
            relative_tmp_dir=str(relative_tmp_dir),
            use_odirect=use_odirect,
            read_ahead_size=read_ahead_size,
            max_capacity_gb=float(max_capacity_gb),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "FS native L2 adapter config fields:\n"
            "- base_path (str): directory for KV "
            "cache files (required)\n"
            "- num_workers (int): C++ worker threads "
            "for I/O (default 4, >0)\n"
            "- relative_tmp_dir (str): relative "
            "sub-dir for temp files (default empty)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (default false)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes "
            "first (optional)\n"
            "- max_capacity_gb (float): max L2 capacity "
            "in GB for usage tracking / eviction "
            "(default 0 = disabled)"
        )


def _create_fs_native_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ filesystem connector."""
    try:
        # First Party
        from lmcache.lmcache_fs import (
            LMCacheFSClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "FS native L2 adapter requires the C++ FS "
            "extension. Build with: pip install -e ."
        ) from e

    # Lazy import to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (  # noqa: E501
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, FSNativeL2AdapterConfig)
    initial_key_sizes = _scan_existing_key_sizes(config.base_path)
    native_client = LMCacheFSClient(
        config.base_path,
        config.num_workers,
        config.relative_tmp_dir,
        config.use_odirect,
        config.read_ahead_size or 0,
    )
    try:
        adapter = NativeConnectorL2Adapter(
            native_client,
            max_capacity_gb=config.max_capacity_gb,
            type_name="FSNativeL2Adapter",
            extra_status={
                "base_path": config.base_path,
                "use_odirect": config.use_odirect,
                "num_workers": config.num_workers,
                "read_ahead_size": config.read_ahead_size,
            },
            initial_key_sizes=initial_key_sizes,
        )
    except Exception:
        native_client.close()
        raise
    logger.info(
        "Created FS native L2 adapter: %s (workers=%d, odirect=%s, "
        "read_ahead=%s, restored_keys=%d, restored_bytes=%d)",
        config.base_path,
        config.num_workers,
        config.use_odirect,
        config.read_ahead_size,
        len(initial_key_sizes),
        sum(initial_key_sizes.values()),
    )
    return adapter


register_l2_adapter_type("fs_native", FSNativeL2AdapterConfig)
register_l2_adapter_factory("fs_native", _create_fs_native_l2_adapter)
