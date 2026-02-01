# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import functools
import inspect
import math
import os
import queue
import re
import shutil
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.ft import FTManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Checkpoint as CheckpointConfig, TORCH_DTYPE_MAP
from torchtitan.protocols import BaseStateDictAdapter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection


MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cache_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }
        return state_dict

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
        # `set_model_state_dict()` does change the keys of the input state_dict,
        # we will need to reinitialize the cache_state_dict.
        self.cache_state_dict = self._get_state_dict()


class Terminate:
    pass


class SaveDone:
    pass


def purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)
            logger.info(
                "Checkpointer deleted %s in %.2f seconds.",
                path,
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the purge thread.")


def _should_retry_with_quantized_reader(exc: BaseException) -> bool:
    message = str(exc)
    if "Size mismatch between saved" not in message:
        return False
    return "down_proj_blocks" in message or "gate_up_proj_blocks" in message


def _fix_quantized_block_metadata(
    metadata: Any, expected_shapes: dict[str, torch.Size] | None = None
) -> Any:
    for fqn, tensor_metadata in metadata.state_dict_metadata.items():
        if not isinstance(tensor_metadata, TensorStorageMetadata):
            continue
        if not fqn.endswith("_blocks"):
            continue
        if expected_shapes and fqn in expected_shapes:
            tensor_metadata.size = expected_shapes[fqn]
            continue
        if len(tensor_metadata.size) < 4:
            continue
        *prefix_shape, groups, block = tensor_metadata.size
        dequantized_size = torch.Size([*prefix_shape, groups * block * 2])
        if expected_shapes and fqn in expected_shapes:
            expected_size = expected_shapes[fqn]
            if expected_size.numel() == dequantized_size.numel():
                tensor_metadata.size = expected_size
                continue
        tensor_metadata.size = dequantized_size
    return metadata


def _ensure_quantized_scale_mapping(reader: Any) -> None:
    weight_map = getattr(reader, "_weight_map", None)
    weight_scale_mapping = getattr(reader, "_weight_scale_mapping", None)
    if not isinstance(weight_map, dict) or not isinstance(weight_scale_mapping, dict):
        return
    for tensor_name in list(weight_map.keys()):
        if not tensor_name.endswith("_blocks"):
            continue
        scale_name = tensor_name.replace("_blocks", "_scales")
        if scale_name in weight_map and tensor_name not in weight_scale_mapping:
            weight_scale_mapping[tensor_name] = scale_name


def _patch_quantized_hf_reader(reader: Any) -> None:
    calculate_scale_shape = getattr(reader, "_calculate_scale_shape", None)
    if not callable(calculate_scale_shape):
        return
    if getattr(reader, "_torchtitan_scale_shape_patch", False):
        return

    def _calculate_scale_shape(weight: torch.Tensor, block_size: int) -> torch.Size:
        try:
            return calculate_scale_shape(weight, block_size)
        except ValueError as exc:
            if "too many values to unpack" not in str(exc):
                raise
            shape = tuple(weight.shape)
            if len(shape) <= 2:
                raise
            *prefix, rows, cols = shape
            if isinstance(block_size, (tuple, list)):
                block = int(block_size[-1])
            else:
                block = int(block_size)
            if block <= 0:
                block = cols
            scale_cols = int(math.ceil(cols / block))
            total_rows = rows * math.prod(prefix) if prefix else rows
            return torch.Size([total_rows, scale_cols])

    reader._calculate_scale_shape = _calculate_scale_shape
    reader._torchtitan_scale_shape_patch = True

    dequantize_tensor = getattr(reader, "_dequantize_tensor", None)
    if callable(dequantize_tensor) and not getattr(
        reader, "_torchtitan_dequantize_patch", False
    ):

        signature = inspect.signature(dequantize_tensor)
        expects_full_shape = len(signature.parameters) >= 3

        def _dequantize_tensor(
            weight: torch.Tensor,
            scale_inv: torch.Tensor,
            full_tensor_shape: torch.Size | None = None,
            slice_info: Any | None = None,
        ) -> torch.Tensor:
            if (
                expects_full_shape
                and full_tensor_shape is not None
                and scale_inv.ndim > 2
                and len(full_tensor_shape) == 2
            ):
                expected_block_rows = math.ceil(
                    full_tensor_shape[0] / reader.block_size
                )
                expected_block_cols = math.ceil(
                    full_tensor_shape[1] / reader.block_size
                )
                expected_numel = expected_block_rows * expected_block_cols
                if scale_inv.numel() == expected_numel:
                    scale_inv = scale_inv.reshape(
                        expected_block_rows, expected_block_cols
                    )
                elif scale_inv.shape[-1] == expected_block_cols:
                    prefix = math.prod(scale_inv.shape[:-1])
                    if prefix == expected_block_rows:
                        scale_inv = scale_inv.reshape(
                            expected_block_rows, expected_block_cols
                        )
            if expects_full_shape:
                return dequantize_tensor(
                    weight, scale_inv, full_tensor_shape, slice_info
                )
            return dequantize_tensor(weight, scale_inv)

        reader._dequantize_tensor = _dequantize_tensor
        reader._torchtitan_dequantize_patch = True

    if getattr(reader, "_torchtitan_quantized_reader_patch", False):
        return

    if not hasattr(reader, "_tensor_full_shapes"):
        reader._tensor_full_shapes = {}

    if not callable(getattr(reader, "_dequantize_tensor_mxfp4", None)):

        def _dequantize_tensor_mxfp4(
            blocks: torch.Tensor,
            scales: torch.Tensor,
            req: Any,
            group_start: int,
            offset_in_first_group: int,
        ) -> torch.Tensor:
            # Adapted from PyTorch 2.10 implementation.
            fp4_values = [
                +0.0,
                +0.5,
                +1.0,
                +1.5,
                +2.0,
                +3.0,
                +4.0,
                +6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]

            dim0_start = req.storage_offsets[0]
            dim0_end = dim0_start + req.lengths[0]
            dim1_start = req.storage_offsets[1]
            dim1_end = dim1_start + req.lengths[1]
            num_groups = blocks.shape[2]
            scales = scales[
                dim0_start:dim0_end,
                dim1_start:dim1_end,
                group_start : group_start + num_groups,
            ]

            scales = scales.to(torch.int32) - 127

            if blocks.shape[:-1] != scales.shape:
                raise ValueError(
                    f"{blocks.shape=} does not match {scales.shape=} for MXFP4"
                )

            lut = torch.tensor(
                fp4_values, dtype=reader.target_dtype, device=blocks.device
            )

            *prefix_shape, groups, block = blocks.shape
            rows_total = math.prod(prefix_shape) * groups

            blocks = blocks.reshape(rows_total, block)
            scales = scales.reshape(rows_total, 1)

            out = torch.empty(
                rows_total, block * 2, dtype=reader.target_dtype, device=blocks.device
            )

            rows_per_chunk = 16384 * 512

            for r0 in range(0, rows_total, rows_per_chunk):
                r1 = min(r0 + rows_per_chunk, rows_total)

                blk = blocks[r0:r1]
                exp = scales[r0:r1]

                idx_lo = (blk & 0x0F).to(torch.long)
                idx_hi = (blk >> 4).to(torch.long)

                sub = out[r0:r1]
                sub[:, 0::2] = lut[idx_lo]
                sub[:, 1::2] = lut[idx_hi]

                torch.ldexp(sub, exp, out=sub)

                del idx_lo, idx_hi, blk, exp

            result = out.reshape(*prefix_shape, groups, block * 2).view(
                *prefix_shape, groups * block * 2
            )

            if offset_in_first_group > 0 or result.shape[-1] > req.lengths[2]:
                end_offset = offset_in_first_group + req.lengths[2]
                result = result[..., offset_in_first_group:end_offset]

            return result

        reader._dequantize_tensor_mxfp4 = _dequantize_tensor_mxfp4

    read_quantized_tensor = getattr(reader, "_read_quantized_tensor_with_block_alignment")
    if callable(read_quantized_tensor) and not getattr(
        reader, "_torchtitan_read_quantized_patch", False
    ):

        def _read_quantized_tensor_with_block_alignment(
            req: Any, safetensor_file: Any
        ) -> torch.Tensor:
            tensor_fqn = req.storage_index.fqn
            scale_fqn = reader._weight_scale_mapping[tensor_fqn]

            try:
                group_start = 0
                offset_in_first_group = 0
                if tensor_fqn.endswith("_blocks"):
                    # Determine quantized shape to infer block width.
                    quantized_shape = safetensor_file.get_slice(tensor_fqn).shape
                    block = int(quantized_shape[-1])
                    values_per_group = block * 2

                    assert len(req.storage_offsets) == 3
                    dim2_start_deq = req.storage_offsets[2]
                    dim2_length_deq = req.lengths[2]
                    dim2_end_deq = dim2_start_deq + dim2_length_deq

                    group_start = dim2_start_deq // values_per_group
                    group_end = (dim2_end_deq + values_per_group - 1) // values_per_group

                    weight_slices_4d = (
                        slice(
                            req.storage_offsets[0],
                            req.storage_offsets[0] + req.lengths[0],
                        ),
                        slice(
                            req.storage_offsets[1],
                            req.storage_offsets[1] + req.lengths[1],
                        ),
                        slice(group_start, group_end),
                        slice(None),
                    )
                    quantized_tensor = safetensor_file.get_slice(tensor_fqn)[
                        weight_slices_4d
                    ]

                    offset_in_first_group = dim2_start_deq - (
                        group_start * values_per_group
                    )
                else:
                    weight_slices = tuple(
                        slice(offset, offset + length)
                        for offset, length in zip(req.storage_offsets, req.lengths)
                    )
                    quantized_tensor = safetensor_file.get_slice(tensor_fqn)[weight_slices]

                scale_file_name = reader._weight_map.get(scale_fqn)
                if scale_file_name is None:
                    raise ValueError(
                        f"Scale tensor {scale_fqn} not found in weight_map"
                    )

                weight_file_name = reader._weight_map.get(tensor_fqn)
                if scale_file_name == weight_file_name:
                    scale_inv = safetensor_file.get_tensor(scale_fqn)
                else:
                    from safetensors import safe_open  # type: ignore[import]

                    scale_file_path = Path(reader.path) / scale_file_name
                    with safe_open(
                        scale_file_path, framework="pt", device="cpu"
                    ) as scale_file:
                        scale_inv = scale_file.get_tensor(scale_fqn)

                if tensor_fqn.endswith("_blocks"):
                    return reader._dequantize_tensor_mxfp4(
                        blocks=quantized_tensor,
                        scales=scale_inv,
                        req=req,
                        group_start=group_start,
                        offset_in_first_group=offset_in_first_group,
                    )

                return reader._dequantize_tensor(weight=quantized_tensor, scale_inv=scale_inv)

            except Exception as exc:
                logger.error("Failed to read the quantized tensor!!")
                raise exc

        reader._read_quantized_tensor_with_block_alignment = (
            _read_quantized_tensor_with_block_alignment
        )
        reader._torchtitan_read_quantized_patch = True

    is_tensor_quantized = getattr(reader, "_is_tensor_quantized", None)
    if callable(is_tensor_quantized) and not getattr(
        reader, "_torchtitan_is_quantized_patch", False
    ):

        def _is_tensor_quantized(tensor_fqn: str) -> bool:
            if tensor_fqn.endswith((".weight_scale_inv", "_scales")):
                return False
            return tensor_fqn in reader._weight_scale_mapping

        reader._is_tensor_quantized = _is_tensor_quantized
        reader._torchtitan_is_quantized_patch = True

    reader._torchtitan_quantized_reader_patch = True


class _MetadataFixingStorageReader:
    def __init__(
        self, base_reader: Any, expected_shapes: dict[str, torch.Size] | None = None
    ) -> None:
        self._base_reader = base_reader
        self._expected_shapes = expected_shapes
        self._cached_metadata: Any | None = None

    def read_metadata(self, *args: Any, **kwargs: Any) -> Any:
        if self._cached_metadata is None:
            metadata = self._base_reader.read_metadata(*args, **kwargs)
            _ensure_quantized_scale_mapping(self._base_reader)
            self._cached_metadata = _fix_quantized_block_metadata(
                metadata, self._expected_shapes
            )
            tensor_full_shapes = getattr(self._base_reader, "_tensor_full_shapes", None)
            if isinstance(tensor_full_shapes, dict):
                for fqn, tensor_metadata in metadata.state_dict_metadata.items():
                    if not fqn.endswith("_blocks"):
                        continue
                    if isinstance(tensor_metadata, TensorStorageMetadata):
                        tensor_full_shapes[fqn] = tensor_metadata.size
        return self._cached_metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_reader, name)


def _wrap_storage_reader_for_quantized_blocks(
    storage_reader: Any, expected_shapes: dict[str, torch.Size] | None = None
) -> Any:
    _patch_quantized_hf_reader(storage_reader)
    return _MetadataFixingStorageReader(storage_reader, expected_shapes)


def _collect_expected_quantized_shapes(
    state_dict: dict[str, Any],
) -> dict[str, torch.Size]:
    expected_shapes: dict[str, torch.Size] = {}
    for key, value in state_dict.items():
        if not key.endswith("_blocks"):
            continue
        expected_shapes[key] = torch.Size(value.shape)
    return expected_shapes


def _has_quantized_blocks(state_dict: dict[str, Any]) -> bool:
    return any(key.endswith("_blocks") for key in state_dict)


class CheckpointManager:
    """This class manages the checkpointing logic for the TorchTitan trainer.


    Note: Pipeline Parallelism and Virtual Stages

    1. even for simple PP schedules, there is a separate optimizer each PP rank.
    rank0's optimizer would have a param_group[0] which refers to layers.0 in the original
    model.  rank1's would _also_ have a param_group[0], since it's index based, but
    referring to layers.1.  When saving, these collide and one of them is lost.  Then when
    reloading, only one stage can restore its optimizer states, others will error.

        The solution to this problem is optimizer flattening: it landed in #127071 and is
        enabled in TorchTitan by passing the 'flatten_optimizer_state_dict' kwarg to DCP
        functions called in the OptimizerContainer.
        See PR #127071 (https://github.com/pytorch/pytorch/pull/127071) for the example of
        a flattening state_dict.

    2. With complex PP schedules, we have multiple model chunks per pp rank. This compounds
    challenge (1) by also requiring us to reason about multiple 'optim' objects locally.

        We solve this in the Model and Optimizer wrapper classes by flattening the state dicts
        from each object into one state dict before saving/loading. We rely on the individual
        state_dicts to not collide, which is guaranteed for the model by correct pipeline
        splitting and for the optimizer by the flattening support described in (1).

    3. LR schedulers also index model states like optimizers. Here we flatten the lr_schedulers
    with the assumption that all lr_schedulers have the same state_dict.

    Note: TorchFT checkpointing flow

    There are two types of checkpoints: when TorchFT is enabled: 1) the full persistent
    checkpoint, 2) the per-replica checkpoint.

    The full persistent checkpoint is saved by the replica with
    ``ft_manager.participating_rank() == 0``. It contains everything including the model,
    optimizer, lr_scheduler, dataloader, and train_state. Right now the full persistent
    checkpoint is loaded by all replicas. However, we can optimize it to only load if
    there are no other alive replicas.

    The per-replica checkpoint contains only the dataloader and is saved/loaded by all
    replicas to/from the its own folder. The folder name is prefixed with the ft_replica_id.

    Args:
        dataloader (DataLoader): The dataloader used to load the data.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        checkpoint_config (Checkpoint): The config used to configure the checkpointing.
        base_folder (str): The base folder to save the checkpoint. Will be concatenated
            with checkpoint_config.folder
        sd_adapter (Optional[type[BaseStateDictAdapter]]): The adapter used to convert model state
            dicts between native format and other formats.
        ft_manager (Optional[ft.Manager]): The FTManager from TorchFT.

    """

    mp_queue_send: queue.Queue
    pg: dist.ProcessGroup
    purge_thread: threading.Thread | None

    def __init__(
        self,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        checkpoint_config: CheckpointConfig,
        sd_adapter: BaseStateDictAdapter | None,
        base_folder: str = "",
        ft_manager: FTManager | None = None,
    ) -> None:
        self.enable = checkpoint_config.enable
        self.load_only = checkpoint_config.load_only

        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )

        self.ft_manager = (
            ft_manager.manager if ft_manager and ft_manager.enabled else None
        )

        self.enable_ft_dataloader_checkpoints = (
            self.ft_manager and checkpoint_config.enable_ft_dataloader_checkpoints
        )

        if self.ft_manager and not self.enable_ft_dataloader_checkpoints:
            logger.warning(
                "Fault tolerance is enabled but enable_ft_dataloader_checkpoints is False. "
                "This means replicas can retrain over the same data multiple times, which can result in overfitting."
            )

        if self.ft_manager:
            optimizers.init_cache_state_dict()

            def state_dict():
                ret = {}
                for k, v in self.states.items():
                    if k in {
                        MODEL,
                        OPTIMIZER,
                        LR_SCHEDULER,
                        TRAIN_STATE,
                    }:
                        ret[k] = v.state_dict()
                return ret

            def load_state_dict(state_dict):
                assert state_dict is not None
                for k, v in state_dict.items():
                    self.states[k].load_state_dict(v)

            # pyrefly: ignore [missing-attribute]
            self.ft_manager.set_state_dict_fns(load_state_dict, state_dict)
            assert ft_manager is not None
            self.ft_replica_id = ft_manager.replica_id

        async_mode = checkpoint_config.async_mode.lower()
        self.enable_staging = (
            self.enable and async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        ) or self.enable_ft_dataloader_checkpoints

        if not self.enable and not self.enable_ft_dataloader_checkpoints:
            return

        self.ft_states = {DATALOADER: dataloader}

        self.staging = False
        self.sending_to_checkpoint_mp = False
        self.staging_id = None
        self.cpu_offload_state_dict = None
        self.stager = None

        self.folder = os.path.join(base_folder, checkpoint_config.folder)

        # Checkpoint policy related fields.
        self.initial_load_model_only = checkpoint_config.initial_load_model_only
        self.initial_load_in_hf = checkpoint_config.initial_load_in_hf
        self.initial_load_path = checkpoint_config.initial_load_path
        self.initial_load_in_hf_quantized = (
            checkpoint_config.initial_load_in_hf_quantized
        )
        self.last_save_model_only = checkpoint_config.last_save_model_only
        self.last_save_in_hf = checkpoint_config.last_save_in_hf
        if self.last_save_in_hf:
            assert (
                sd_adapter is not None
            ), "job_config.checkpoint.last_save_in_hf is True, but sd_adapter is not provided."
        self.sd_adapter = sd_adapter
        self.export_dtype = TORCH_DTYPE_MAP[checkpoint_config.export_dtype]
        self.exclude_from_loading = checkpoint_config.exclude_from_loading
        self.interval = checkpoint_config.interval
        self.enable_first_step_checkpoint = (
            checkpoint_config.enable_first_step_checkpoint
        )

        # Async checkpoint related fields.
        async_mode = checkpoint_config.async_mode.lower()
        if (
            async_mode == AsyncMode.ASYNC
            or async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
            or self.enable_ft_dataloader_checkpoints
        ):
            self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))

        self.keep_latest_k = checkpoint_config.keep_latest_k
        if self.keep_latest_k > 0:
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()
        else:
            self.purge_thread = None

        self.mp = None
        self.staging_future = None
        self.save_future = None
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            raise ValueError(
                f"Unknown checkpoint async_mode {checkpoint_config.async_mode}"
            )

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "enable") and self.enable:
            if hasattr(self, "mp") and self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.mp.join()
            if (
                hasattr(self, "purge_thread")
                and self.purge_thread
                and self.purge_thread.is_alive()
            ):
                self.purge_queue.put(Terminate())
                self.purge_thread.join()

            if self.stager is not None:
                self.stager.close()

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Future | AsyncSaveResponse | None:
        """Save the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to save.
            checkpoint_id (str): The checkpoint id to save.
            async_mode (AsyncMode): Whether the checkpoint is async.
            enable_garbage_collection (bool): Whether to enable garbage collection after save.
            to_hf (bool): Whether to save in HF model definition and safetensors format.

        Returns:
            Future: The future object if the checkpoint is async, otherwise None.
        """

        ret: Future | AsyncSaveResponse | None = None

        storage_writer: HuggingFaceStorageWriter | None = None
        checkpoint_save_id: str | None = None
        fqn_to_index_mapping: dict[Any, int] | None = None
        if to_hf:
            assert (
                self.sd_adapter is not None
            ), "trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            state_dict = self.sd_adapter.to_hf(state_dict)

            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping
            if fqn_to_index_mapping:
                storage_writer = HuggingFaceStorageWriter(
                    path=os.path.join(checkpoint_id, "sharded"),
                    save_distributed=True,
                    fqn_to_index_mapping=fqn_to_index_mapping,
                    enable_consolidation=False,
                )
            else:
                # the reason for only enabling consolidation if there is
                # no mapping is because no mapping implies that we save all fqns
                # to one file. This means we only need one rank to consolidate.
                # Otherwise we should use consolidate_safetensors_files_on_every_rank
                storage_writer = HuggingFaceStorageWriter(
                    path=checkpoint_id,
                    save_distributed=True,
                    enable_consolidation=True,
                )

        else:
            checkpoint_save_id = checkpoint_id

        if async_mode == AsyncMode.ASYNC:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
            )
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self.stager,
            )
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
            )

        if to_hf and fqn_to_index_mapping:
            consolidate_safetensors_files_on_every_rank(
                input_dir=os.path.join(checkpoint_id, "sharded"),
                output_dir=checkpoint_id,
                fqn_to_index_mapping=fqn_to_index_mapping,
                num_threads=5,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        """Load the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to load.
            checkpoint_id (str): The checkpoint id to load.
            from_hf (bool): Whether to load from HuggingFace checkpoint with
                its own model definition and safetensors format.
        """

        if from_hf:
            assert (
                self.sd_adapter is not None
            ), "trying to load checkpoint in HF safetensors format, but sd_adapter is not provided."
            hf_state_dict = self.sd_adapter.to_hf(state_dict)
            if _has_quantized_blocks(hf_state_dict):
                from_quantized = True
            hf_storage_reader = self.sd_adapter.get_hf_storage_reader(
                checkpoint_id, from_quantized
            )
            if from_quantized:
                expected_shapes = _collect_expected_quantized_shapes(hf_state_dict)
                hf_storage_reader = _wrap_storage_reader_for_quantized_blocks(
                    hf_storage_reader, expected_shapes
                )

            try:
                dcp.load(
                    hf_state_dict,
                    storage_reader=hf_storage_reader,
                )
            except CheckpointException as exc:
                if from_quantized or not _should_retry_with_quantized_reader(exc):
                    raise
                logger.warning(
                    "Detected quantized GPT-OSS checkpoint layout. Retrying load with "
                    "QuantizedHuggingFaceStorageReader."
                )
                hf_storage_reader = self.sd_adapter.get_hf_storage_reader(
                    checkpoint_id, True
                )
                expected_shapes = _collect_expected_quantized_shapes(hf_state_dict)
                hf_storage_reader = _wrap_storage_reader_for_quantized_blocks(
                    hf_storage_reader, expected_shapes
                )
                dcp.load(
                    hf_state_dict,
                    storage_reader=hf_storage_reader,
                )

            state_dict = self.sd_adapter.from_hf(hf_state_dict)
            self.states[MODEL].load_state_dict(state_dict)
        else:
            dcp.load(state_dict, checkpoint_id=checkpoint_id)

            # TODO: Since we flatten the model states in state_dict, we need to
            # manually call load_state_dict() for the model. Need to fix this.
            if MODEL in self.states:
                self.states[MODEL].load_state_dict(state_dict)

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> None:
        """Save the checkpoint for the current step.

        This function will save the checkpoint for the current step. If ``last_step`` is
        true, it will save the checkpoint even if the interval has not been reached.
        This only happens when train_state.step == job_config.training.steps, or
        for initial seed checkpoint.

        Args:
            curr_step (int): The current step.
            last_step (bool, optional): Whether this is the last step of training.

        Returns:
            None
        """

        if self.enable_ft_dataloader_checkpoints:
            self._ft_save(curr_step)

        if not self._should_save(curr_step, last_step):
            return

        begin = time.monotonic()
        if not self.enable_ft_dataloader_checkpoints or (
            self.ft_manager
            # pyrefly: ignore [missing-attribute]
            and self.ft_manager.participating_rank() == 0
        ):
            logger.info("Saving the checkpoint (or staging if async is enabled).")
            checkpoint_id = self._create_checkpoint_id(curr_step)
            self._async_wait()
            # This GC is called for async checkpoint as it is useless to do
            # GC right after async_save -- the CPU memory is not able to be
            # freed until _async_wait()
            if last_step:
                self._save_last_step(curr_step)
                return

            states = self._flattened_model_states_sd()
            if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                if self.stager is None:
                    self.stager = DefaultStager(StagingOptions(True, True, True, True))
                result = self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=self.async_mode,
                )
                assert isinstance(result, AsyncSaveResponse)
                self.save_future = result.upload_completion
                self.staging_future = result.staging_completion
                self.staging = True
            elif self.async_mode == AsyncMode.ASYNC:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                self.save_future = self.dcp_save(
                    states, checkpoint_id=checkpoint_id, async_mode=self.async_mode
                )
                GarbageCollection.collect("GC collection invoked by checkpointer.")
            else:
                self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=AsyncMode.DISABLED,
                    enable_garbage_collection=True,
                )
            self._purge_stale_checkpoints()

            logger.info(
                "Finished saving the checkpoint (or staging if async is enabled)"
                f"in {time.monotonic() - begin:.2f} seconds."
            )
        elif self.enable_ft_dataloader_checkpoints:
            assert self.ft_manager is not None
            logger.info(
                "Replica %d doesn't save checkpoint.",
                # pyrefly: ignore [missing-attribute]
                self.ft_manager.participating_rank(),
            )

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load the checkpoint for the given step.

        This function will load the checkpoint for the given step. If ``step`` is -1, it
        will load the latest checkpoint. If the checkpoint does not exist, it will return
        False and load nothing.

        Args:
            step (int, optional): The step to load the checkpoint for. Defaults to -1.

        Returns:
            bool: Whether the checkpoint was loaded successfully.
        """

        if self.enable_ft_dataloader_checkpoints:
            self._ft_load()

        if not self.enable:
            return False

        model_only = False
        from_hf = False
        from_quantized = False
        if not os.path.exists(self.folder):
            model_only = self.initial_load_model_only
            from_hf = self.initial_load_in_hf
            from_quantized = self.initial_load_in_hf_quantized
            if from_hf:
                assert (
                    model_only
                ), "Only model can be loaded when loading from HF's safetensors checkpoint."

            if from_quantized:
                assert (
                    from_hf
                ), "Quantized checkpoint can only be loaded from HuggingFace format."

            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "checkpoint.initial_load_path is specified but the path is not valid."
                    )
                if from_hf:
                    logger.info(
                        f"loading from HF safetensors from --checkpoint.initial_load_path: {self.initial_load_path}"
                    )
            elif from_hf:
                assert (
                    self.sd_adapter is not None
                    and self.sd_adapter.hf_assets_path is not None
                ), "from_hf is True but sd_adapter or hf_assets_path is not provided."
                hf_assets_path = self.sd_adapter.hf_assets_path
                checkpoint_id = hf_assets_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "model.hf_assets_path is being used to load HF weights but the path is not valid. \
                        Either make sure hf_assets_path is correct or provide a valid checkpoint.initial_load_path"
                    )
                logger.info(
                    f"loading HF safetensors from --model.hf_assets_path: {hf_assets_path}"
                )
            else:
                return False
        else:
            if self.initial_load_path:
                logger.warning(
                    "checkpoint.initial_load_path is provided but the checkpoint.folder exists. "
                    f"Checkpointer will use the checkpoints from the checkpoint.folder {self.folder}."
                )
            if self.initial_load_in_hf:
                logger.warning(
                    "checkpoint.initial_load_in_hf is True but the checkpoint.folder exists. "
                    "Checkpointer will not load from HF safetensors"
                )
            step = self._find_load_step() if step == -1 else step
            if step == -1:
                return False
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)

            if not os.path.isdir(checkpoint_id):
                raise FileNotFoundError(
                    f"--checkpoint.load_step={step} but checkpoint {checkpoint_id} is not found."
                )

        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()
        states = self._states_to_load(model_only)
        self.dcp_load(
            states,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    def maybe_wait_for_staging(self) -> None:
        """Wait for the staging to finish if it is enabled.

        This function will wait for staging to finish. The staging is only enabled
        with ``async_checkpoint_with_pinned_memory``.
        """
        if self.enable_staging and self.staging:
            assert self.staging_future is not None
            self.staging_future.result()
            self.staging = False

    def _find_load_step(self, folder: str = "") -> int:
        """Find the step to load the checkpoint for.

        Args:
            folder (str, optional): The folder to find the checkpoint for. If ``folder``
            is "", then ``self.folder`` will be used.

        Returns:
            int: The step to load the checkpoint for.
        """
        folder = folder if folder else self.folder
        pattern = r"step-(\d+)"
        step_counts = []

        if not os.path.isdir(folder):
            return -1

        for filename in os.listdir(folder):
            match = re.search(pattern, filename)
            dcp_metadata_probe = os.path.join(folder, filename, ".metadata")
            safetensors_metadata_probe = os.path.join(
                folder, filename, "model.safetensors.index.json"
            )
            if match and os.path.isfile(dcp_metadata_probe):
                step_counts.append(int(match.group(1)))
            elif match and os.path.isfile(safetensors_metadata_probe):
                step_counts.append(int(match.group(1)))
        if not step_counts:
            return -1
        return max(step_counts)

    def _ft_folder(self) -> str:
        return os.path.join(self.folder, f"ft-replicat-{self.ft_replica_id}")

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        folder = folder if folder else self.folder
        return os.path.join(folder, f"step-{step}")

    def _ft_save(self, step: int) -> None:
        begin = time.monotonic()
        self._async_wait()
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        self.save_future = self.dcp_save(
            self.ft_states, checkpoint_id=checkpoint_id, async_mode=AsyncMode.ASYNC
        )
        logger.info(f"Staging ft checkpoint took {time.monotonic() - begin} secs.")

    def _ft_load(self) -> None:
        step = self._find_load_step(folder=self._ft_folder())
        if step == -1:
            return

        begin = time.monotonic()
        logger.info(f"Loading the FT checkpoint at step {step}.")
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        self.dcp_load(
            self.ft_states,
            checkpoint_id=checkpoint_id,
            # FT checkpoints are always DCP because FT checkpoint currently only save/load dataloader.
            from_hf=False,
            from_quantized=False,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the ft checkpoint in {time.monotonic() - begin:.2f} seconds."
        )

    def _flattened_model_states_sd(
        self, state_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Flatten the model states into a single dictionary.

        Note that other states, such as optimizer states, are not flattened.
        """
        states = state_dict if state_dict is not None else self.states
        sd = {k: v for k, v in states.items() if k != MODEL}
        if MODEL in states:
            sd.update(states[MODEL].state_dict())
        return sd

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        """Determines which states to load for the given step.

        This API is used to determine which states to load based on the
        configurations.

        Args:
            model_only (bool): Whether to load the model only.

        Returns:
            Dict[str, Any]: The states to load for the given step.
        """
        # For the first step, we will only load the model.
        if model_only:
            return self.states[MODEL].state_dict()

        for exclude_key in self.exclude_from_loading:
            if exclude_key not in self.states:
                raise ValueError(f"{exclude_key} not found in state_dict.")

        states_to_load = {
            k: v for k, v in self.states.items() if k not in self.exclude_from_loading
        }

        states_to_load = self._flattened_model_states_sd(states_to_load)

        if self.enable_ft_dataloader_checkpoints:
            states_to_load.pop(DATALOADER)

        return states_to_load

    def _save_last_step(self, curr_step: int) -> None:
        # We only consider saving model only at the end of the training. So this
        # won't affect preemption and training resume. We also only allow dtype
        # conversion when we are checkpointing model only and the current dtype
        # is not the same as the export dtype at the end of the training.

        if self.last_save_model_only:
            states = self.states[MODEL].state_dict()

            if self.export_dtype != torch.float32:
                states = {k: v.to(self.export_dtype) for k, v in states.items()}
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flattened_model_states_sd()

        if self.last_save_in_hf:
            assert (
                self.last_save_model_only
            ), "Only model can be saved when saving in HF safetensors format."

        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
        )

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        if not self.enable or self.load_only:
            return False

        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True

        if last_step:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.save_future is not None:
                self.save_future.result()
        elif (
            self.async_mode == AsyncMode.ASYNC or self.enable_ft_dataloader_checkpoints
        ):
            if self.save_future is not None:
                self.save_future.result()
                self.save_future = None
        elif self.save_future is not None:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is not enabled "
                "and fault tolerance is not active."
            )

    def _purge_stale_checkpoints(self):
        if (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and os.path.isdir(self.folder)
            and (
                not self.enable_ft_dataloader_checkpoints
                # pyrefly: ignore [missing-attribute]
                or (self.ft_manager and self.ft_manager.participating_rank() == 0)
            )
        ):
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):
                match = re.search(r"step-(\d+)", filename)
                if match:
                    path = os.path.join(self.folder, filename)
                    discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)
