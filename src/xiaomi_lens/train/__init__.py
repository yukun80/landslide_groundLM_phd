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

# 【中文导读｜阅读第 5 步】
# 本文件：对外提供训练器和训练设置，按需加载以降低导入成本。
# 怎么读：先了解这个目录提供什么，随后去具体实现文件阅读；__init__.py 不是训练起点。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""Training for Xiaomi-Lens-0. Install with `pip install -e ".[train]"`."""


# Load torch / transformers only when a trainer symbol is requested.
# 【步骤说明】当访问包里尚未导入的名字时查找其所属模块，返回相应对象；未知名字抛出 AttributeError，避免提前加载所有依赖。
# 对照参数：name；下面保留原始英文说明与类型，方便对应调用处。
def __getattr__(name):
    if name == "XiaomiLensTrainer":
        from .trainer import XiaomiLensTrainer
        return XiaomiLensTrainer
    elif name == "XiaomiLensModelConfig":
        from .config import XiaomiLensModelConfig
        return XiaomiLensModelConfig
    elif name == "XiaomiLensScriptArguments":
        from .config import XiaomiLensScriptArguments
        return XiaomiLensScriptArguments
    elif name == "XiaomiLensTrainingConfig":
        from .config import XiaomiLensTrainingConfig
        return XiaomiLensTrainingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "XiaomiLensModelConfig",
    "XiaomiLensScriptArguments",
    "XiaomiLensTrainer",
    "XiaomiLensTrainingConfig",
]
