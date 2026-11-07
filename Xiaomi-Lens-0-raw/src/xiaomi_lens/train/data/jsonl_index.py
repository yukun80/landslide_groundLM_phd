# Copyright 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 【中文导读｜阅读第 4 步】
# 本文件：只记住每条样本在文件中的位置，需要时再读取，避免一次载入全部数据。
# 怎么读：先看 scan_jsonl_offsets → JsonlIndexShard.get → JsonlIndexDataset.__getitem__。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""Lightweight JSONL offset-index dataset.

This is the ms-swift-like path for large training mixes: scan cheap line
offsets and lengths into memory, then parse/process one row at a time in
DataLoader workers. No dataset artifact is written to disk.
"""

from __future__ import annotations

import json
import logging
import math
import mmap
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .io import DatasetEntry, resolve_sample_images

logger = logging.getLogger(__name__)


# 【类的用途】保存非空样本行的字节起点与长度，以及源文件大小和修改时间。
@dataclass(frozen=True)
class JsonlIndex:
    """Byte offsets for every non-empty line in one JSONL file."""

    path: Path
    offsets: np.ndarray
    lengths: np.ndarray
    size: int
    mtime_ns: int

    # 【步骤说明】返回索引中非空样本行的数量，供 len(index) 使用。
    def __len__(self) -> int:
        return int(self.offsets.shape[0])


# 【步骤说明】逐行扫描样本文件，跳过空白行并记录每行起点和长度；返回索引，不把全部样本内容留在内存。
# 对照参数：path；下面保留原始英文说明与类型，方便对应调用处。
def scan_jsonl_offsets(path: str | Path) -> JsonlIndex:
    """Single-pass scan of non-empty JSONL line byte ranges."""
    p = Path(path).resolve()
    st = p.stat()
    offsets: list[int] = []
    lengths: list[int] = []
    cursor = 0
    with p.open("rb") as f:
        for line in f:
            line_len = len(line)
            stripped = line.rstrip(b"\r\n")
            if stripped.strip():
                offsets.append(cursor)
                lengths.append(len(stripped))
            # cursor 按原始字节前进，包括换行；只去掉换行后的长度用于读取 JSON。
            # 不能按中文字符个数当字节偏移，否则含中文样本会读错位置。
            cursor += line_len
    return JsonlIndex(
        path=p,
        offsets=np.asarray(offsets, dtype=np.uint64),
        lengths=np.asarray(lengths, dtype=np.uint32),
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
    )


# 【步骤说明】根据 all、first、end 或 random 规则挑选样本编号；返回选中编号，百分比向上取整，随机选择受 seed 控制。
# 对照参数：indices, strategy, seed；下面保留原始英文说明与类型，方便对应调用处。
def apply_sampling_strategy(
    indices: np.ndarray,
    strategy: str,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Apply the data_config sampling DSL to an index array."""
    if not strategy or strategy == "all":
        return indices
    if ":" not in strategy:
        raise ValueError(
            f"Unsupported sampling_strategy {strategy!r}; expected mode:count."
        )
    mode, count_str = strategy.split(":", 1)
    n = int(indices.shape[0])
    if count_str.endswith("%"):
        pct = int(count_str[:-1])
        count = math.ceil(n * pct / 100)
    else:
        count = int(count_str)
    count = max(0, min(count, n))

    if mode == "first":
        return indices[:count]
    if mode == "end":
        return indices[-count:] if count else indices[:0]
    if mode == "random":
        rng = np.random.default_rng(seed)
        return indices[rng.permutation(n)[:count]]
    raise ValueError(
        f"Unsupported sampling mode {mode!r}; valid: first | end | random."
    )


# 【步骤说明】按进程编号隔位领取样本，必要时裁掉不能平分的尾部；返回本进程编号列表，样本少于进程数时保留原列表。
# 对照参数：indices, rank, world_size；下面保留原始英文说明与类型，方便对应调用处。
def stride_for_rank(
    indices: np.ndarray,
    *,
    rank: int,
    world_size: int,
) -> np.ndarray:
    """Deterministically keep rank::world_size after trimming tails."""
    if world_size <= 1:
        return indices
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} not in [0, {world_size})")
    n = int(indices.shape[0])
    if n < world_size:
        return indices
    trimmed = (n // world_size) * world_size
    return indices[rank:trimmed:world_size]


# 【类的用途】管理一份样本文件的只读映射与索引，提供按行读取及关闭操作。
class JsonlIndexShard:
    """One JSONL file plus offset index and dataset metadata."""

    # 【步骤说明】保存索引与图片目录等来源信息，并打开当前进程的只读文件映射。 构造完成后通过对象保存这些状态，不返回预测结果。
    # 对照参数：index, image_folder, name, tag；下面保留原始英文说明与类型，方便对应调用处。
    def __init__(
        self,
        index: JsonlIndex,
        *,
        image_folder: str = "",
        name: str = "",
        tag: str = "",
    ) -> None:
        self.index = index
        self.image_folder = image_folder
        self.name = name or index.path.stem
        self.tag = tag
        self._fd: int | None = None
        self._mmap: mmap.mmap | None = None
        self._default_modality: bool | None = None
        self._open()

    # 【步骤说明】打开只读文件并建立按需读取的内存映射；更新对象状态，已经打开时直接返回。
    def _open(self) -> None:
        if self._mmap is not None:
            return
        fd = os.open(str(self.index.path), os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._mmap = mm

    # 【步骤说明】返回可传递给另一进程的索引与来源信息，不传递已打开的文件句柄。
    def __getstate__(self) -> dict:
        return {
            "index": self.index,
            "image_folder": self.image_folder,
            "name": self.name,
            "tag": self.tag,
            "default_modality": self._default_modality,
        }

    # 【步骤说明】接收另一进程恢复的状态，重建本进程自己的文件映射；无返回值，避免跨进程复用无效句柄。
    # 对照参数：state；下面保留原始英文说明与类型，方便对应调用处。
    def __setstate__(self, state: dict) -> None:
        self.index = state["index"]
        self.image_folder = state["image_folder"]
        self.name = state["name"]
        self.tag = state["tag"]
        self._default_modality = state.get("default_modality")
        self._fd = None
        self._mmap = None
        self._open()

    # 【步骤说明】返回这一份样本文件的有效行数，转交给底层索引统计。
    def __len__(self) -> int:
        return len(self.index)

    # 【步骤说明】按指定行号查索引，只解析这一行，再补齐图片路径与来源信息；返回一条样本。
    # 对照参数：i；下面保留原始英文说明与类型，方便对应调用处。
    def get(self, i: int) -> dict:
        self._open()
        assert self._mmap is not None
        off = int(self.index.offsets[i])
        ln = int(self.index.lengths[i])
        # 只从当前位置读这一行，不解析整个文件；图片此时仍是路径，实际像素留到拼批时再读。
        sample = json.loads(self._mmap[off:off + ln])
        sample = resolve_sample_images(sample, self.image_folder)
        sample.setdefault("_dataset_name", self.name)
        sample.setdefault("_dataset_tag", self.tag)
        return sample

    # 【步骤说明】读取首条样本来估计该来源是否含图片并缓存；返回布尔值，这是来源级估计，不逐条检查。
    @property
    def default_modality(self) -> bool:
        if self._default_modality is None:
            self._default_modality = (
                bool((self.get(0).get("images") or [])) if len(self) else False
            )
        return self._default_modality

    # 【步骤说明】释放只读映射与文件描述符并清空状态，可重复调用；不修改样本文件。
    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

    # 【步骤说明】对象销毁时调用 close 释放文件资源，不执行数据写入。
    def __del__(self) -> None:
        self.close()


# 【步骤说明】对一个样本文件先建索引，再创建带图片目录与来源名称的读取对象；返回可按行读取的分片。
# 对照参数：path, image_folder, name, tag；下面保留原始英文说明与类型，方便对应调用处。
def open_jsonl_shard(
    path: str | Path,
    *,
    image_folder: str = "",
    name: str = "",
    tag: str = "",
) -> JsonlIndexShard:
    idx = scan_jsonl_offsets(path)
    return JsonlIndexShard(idx, image_folder=image_folder, name=name, tag=tag)


# 【类的用途】把多个样本文件拼成统一编号空间，并为损坏记录提供有限重试。
class JsonlIndexDataset:
    """Concatenated JSONL shards with bounded retry on bad rows."""

    # 【步骤说明】建立总编号到分片行号的对应表，保存抽样结果，并生成遇到坏行时的备用读取顺序。 构造完成后通过对象保存这些状态，不返回预测结果。
    # 对照参数：shards, n_try_fetch, seed, is_rank_sharded, shard_world, shard_rank, global_indices；下面保留原始英文说明与类型，方便对应调用处。
    def __init__(
        self,
        shards: list[JsonlIndexShard],
        *,
        n_try_fetch: int = 10,
        seed: int = 0,
        is_rank_sharded: bool = False,
        shard_world: int = 1,
        shard_rank: int = 0,
        global_indices: np.ndarray | None = None,
    ) -> None:
        if not shards:
            raise ValueError("JsonlIndexDataset requires at least one shard.")
        self._shards = shards
        shard_ids: list[np.ndarray] = []
        local_idx: list[np.ndarray] = []
        for sid, shard in enumerate(shards):
            n = len(shard)
            shard_ids.append(np.full(n, sid, dtype=np.uint16))
            local_idx.append(np.arange(n, dtype=np.uint32))
        sid_flat = (
            np.concatenate(shard_ids) if shard_ids else np.zeros(0, dtype=np.uint16)
        )
        loc_flat = (
            np.concatenate(local_idx) if local_idx else np.zeros(0, dtype=np.uint32)
        )
        if global_indices is None:
            self._shard_ids = sid_flat
            self._local_idx = loc_flat
        else:
            self._shard_ids = sid_flat[global_indices]
            self._local_idx = loc_flat[global_indices]

        self.n_try_fetch = max(1, int(n_try_fetch))
        rng = np.random.default_rng(seed)
        n = len(self)
        self._fallback_perm = (
            rng.permutation(n).astype(np.int64) if n else np.zeros(0, dtype=np.int64)
        )
        self._fallback_cursor = 0
        self._warning_count = 0
        self.is_rank_sharded = is_rank_sharded
        self.shard_world = shard_world
        self.shard_rank = shard_rank

    # 【步骤说明】返回抽样和分片后当前数据集的样本数量，不一定等于原文件行数总和。
    def __len__(self) -> int:
        return int(self._shard_ids.shape[0])

    # 【步骤说明】根据总编号定位分片和行；坏行会按固定备用顺序有限次重试，成功返回样本，全部失败则报错。
    # 对照参数：idx；下面保留原始英文说明与类型，方便对应调用处。
    def __getitem__(self, idx: int) -> dict:
        n = len(self)
        if n == 0:
            raise IndexError("JsonlIndexDataset is empty")
        last_exc: Exception | None = None
        # 先试用户要求的样本，失败才取备用编号；有限重试防止一直困在坏数据上。
        # 这会改变实际送入的样本，所以警告和数据检查不能忽略。
        for attempt in range(self.n_try_fetch):
            if attempt == 0:
                try_idx = int(idx) % n
            else:
                try_idx = int(self._fallback_perm[self._fallback_cursor % n])
                self._fallback_cursor += 1
            try:
                sid = int(self._shard_ids[try_idx])
                loc = int(self._local_idx[try_idx])
                return self._shards[sid].get(loc)
            except Exception as exc:
                last_exc = exc
                if self._warning_count < 10:
                    logger.warning(
                        "[jsonl-index] failed idx=%d attempt=%d/%d: %s",
                        try_idx,
                        attempt + 1,
                        self.n_try_fetch,
                        exc,
                    )
                    self._warning_count += 1
        raise RuntimeError(
            f"JsonlIndexDataset exhausted {self.n_try_fetch} retries from "
            f"idx={idx}; last error: {last_exc!r}"
        )

    # 【步骤说明】按当前数据集编号逐个产生样本，实际读取与重试都交给 __getitem__。
    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    # 【步骤说明】返回数据对象对外提供的字段名称，供训练工具识别样本结构。
    @property
    def column_names(self) -> list[str]:
        return ["messages", "images", "segmentation", "_dataset_name", "_dataset_tag"]

    # 【步骤说明】返回每条所选样本的行字节长度，用作分组时的粗略长短估计；它不等于真正的文字单元数。
    @property
    def lengths(self) -> list[int]:
        return [
            int(self._shards[int(sid)].index.lengths[int(loc)])
            for sid, loc in zip(self._shard_ids, self._local_idx)
        ]

    # 【步骤说明】返回各样本所属来源的图文类型估计，辅助将纯文字与图文样本分组。
    @property
    def modalities(self) -> list[bool]:
        return [
            self._shards[int(sid)].default_modality for sid in self._shard_ids
        ]


# 【步骤说明】接收数据来源清单，逐文件建立索引、应用抽样规则并按进程切分；返回按需读取的数据集对象。
# 对照参数：entries, seed, rank, world_size, n_try_fetch；下面保留原始英文说明与类型，方便对应调用处。
def build_jsonl_index_dataset(
    entries: list[DatasetEntry],
    *,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
    n_try_fetch: int = 10,
) -> JsonlIndexDataset:
    """Build an indexed lazy dataset from DatasetEntry objects."""
    if not entries:
        raise ValueError("build_jsonl_index_dataset: `entries` is empty.")

    shards: list[JsonlIndexShard] = []
    per_entry_indices: list[np.ndarray] = []
    offset = 0
    for entry in entries:
        shard = open_jsonl_shard(
            entry.path,
            image_folder=entry.image_folder,
            name=entry.display_name,
            tag=entry.tag,
        )
        shards.append(shard)
        local = np.arange(len(shard), dtype=np.int64)
        local = apply_sampling_strategy(
            local,
            entry.sampling_strategy or "all",
            seed=seed,
        )
        per_entry_indices.append(local + offset)
        offset += len(shard)

    global_indices = (
        np.concatenate(per_entry_indices)
        if per_entry_indices else np.zeros(0, dtype=np.int64)
    )
    global_indices = stride_for_rank(
        global_indices,
        rank=rank,
        world_size=world_size,
    )
    return JsonlIndexDataset(
        shards,
        n_try_fetch=n_try_fetch,
        seed=seed,
        is_rank_sharded=(world_size > 1),
        shard_world=world_size,
        shard_rank=rank,
        global_indices=global_indices,
    )
