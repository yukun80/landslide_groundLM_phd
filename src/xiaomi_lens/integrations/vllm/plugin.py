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

# 【中文导读｜阅读第 7 步】
# 本文件：向 vLLM 登记本项目的自定义模型类，让服务能找到它。
# 怎么读：登记不是加载权重；实际加载与运行由服务框架调用。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""vLLM general-plugin entry point for Xiaomi-Lens-0.

Registered via `pyproject.toml` under `[project.entry-points."vllm.general_plugins"]`.
vLLM calls `load_general_plugins()` in every process it spawns — the API server AND
each EngineCore worker — so registering the custom model here (rather than with a
one-off `ModelRegistry.register_model(...)` in the launcher) is what makes it visible
across the v1 multiprocess boundary.

The model class is passed as a `"module:Class"` string so vLLM imports it lazily,
inside the worker, only when the architecture is actually requested — keeping this
entry point cheap to load.
"""

from __future__ import annotations


# 【步骤说明】向 vLLM 的模型注册表登记本项目模型名称与位置；无返回值，之后工作进程可据此加载。
def register() -> None:
    from vllm import ModelRegistry

    # Map the checkpoint's config.json architecture to our vLLM model that loads
    # PixelMaskDecoder and exposes a mask-producing pooler. No --hf-overrides
    # needed once this is registered (vLLM matches the arch name directly).
    ModelRegistry.register_model(
        "XiaomiLensQwen3_5ForConditionalGeneration",
        "xiaomi_lens.integrations.vllm.vllm_model:XiaomiLensForVLLM",
    )
