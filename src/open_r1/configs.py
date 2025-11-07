# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import trl


@dataclass
class DatasetConfig:
    """数据集混合中的数据集配置。"""

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """数据集混合的配置。"""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    ScriptArguments 的扩展版本，支持数据集混合。

    参数：
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            用于创建具有高级选项的数据集混合的配置。
            格式：
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    # 重写 dataset_name 使其可选
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "数据集名称。如果使用 dataset_mixture，可以省略。"}
    )
    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "用于创建具有高级选项（如打乱）的数据集混合的配置。"},
    )

    def __post_init__(self):
        # 验证至少提供了 dataset_name 或 dataset_mixture 中的一个
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("必须提供 `dataset_name` 或 `dataset_mixture` 中的至少一个")

        # 处理 dataset_mixture 的配置
        if self.dataset_mixture is not None:
            # 验证 dataset_mixture 是字典且包含 'datasets' 键
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture 必须是包含 'datasets' 键的字典。"
                    "预期格式：{'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            # 将数据集配置转换为 DatasetConfig 对象
            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' 必须是数据集配置的列表")

            # 创建 DatasetMixtureConfig 对象来管理数据集混合
            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # 检查列名在所有数据集配置中是否一致
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                # 确保所有数据集的列名完全相同
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "混合中所有数据集配置的列名必须一致。"
                        f"发现不同的列集合：{[list(cols) for cols in columns_sets]}"
                    )


# TODO: 使用 mixin 添加共享选项以减少代码重复
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    回调、基准测试等的参数
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "训练后要运行的基准测试。"},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "训练期间要运行的回调函数。"},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "要使用的聊天模板。"})
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "要推送模型的 Hub 分支。"}
    )
    num_completions_to_print: int = field(default=0, metadata={"help": "要打印的生成结果数量。"})
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "是否覆盖 Hub 修订版本。"})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "是否推送到 Hub 修订版本/分支。"})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "要使用的可选系统提示。"},
    )
    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": ("是否将唯一的提示记录到 wandb。这将为每个唯一的提示创建一个新的运行。")
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("存储运行的实体。")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("存储运行的项目。")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("存储运行的组。")},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    回调、基准测试等的参数
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "训练后要运行的基准测试。"},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "训练期间要运行的回调函数。"},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "要使用的聊天模板。"})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "用于基准测试的可选系统提示。"},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "要推送模型的 Hub 分支。"},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "是否覆盖 Hub 修订版本。"})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "是否推送到 Hub 修订版本/分支。"})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("存储运行的实体。")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("存储运行的项目。")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("存储运行的组。")},
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    GRPO 训练脚本的脚本参数。

    参数：
        reward_funcs (`list[str]`):
            奖励函数列表。可能的值：'accuracy'、'format'、'reasoning_steps'、'cosine'、'repetition_penalty'、'length'、'tag_count'、'code'、'ioi_code'、'code_format'、'soft_overlong_punishment'。
        cosine_min_value_wrong (`float`):
            错误答案余弦缩放的最小奖励。
        cosine_max_value_wrong (`float`):
            错误答案余弦缩放的最大奖励。
        cosine_min_value_correct (`float`):
            正确答案余弦缩放的最小奖励。
        cosine_max_value_correct (`float`):
            正确答案余弦缩放的最大奖励。
        cosine_max_len (`int`):
            余弦缩放的最大长度。
        code_language (`str`):
            代码格式奖励的语言。
        max_completion_len (`int`):
            生成结果中的最大令牌数。
        soft_punish_cache (`int`):
            生成结果中的最小令牌数。
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "奖励函数列表。可能的值：'accuracy'、'format'、'reasoning_steps'、'cosine'、'repetition_penalty'、'length'、'tag_count'、'code'、'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "错误答案的最小奖励"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "错误答案的最大奖励"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "正确答案的最小奖励"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "正确答案的最大奖励"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "缩放的最大长度"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "重复惩罚奖励的 n-gram 数量"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "重复惩罚奖励的最大（负）惩罚"},
    )
    code_language: str = field(
        default="python",
        # '(?:python|cpp)'
        metadata={
            "help": "代码格式奖励的语言。基于 E2B 支持的语言 https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "对于每个生成结果，并行评估这么多的测试用例，然后检查其中是否有任何失败（0分）：如果是则停止评估；否则继续评估下一批测试用例。这有助于避免过载评估服务器并节省错误解决方案的时间"
        },
    )
    code_eval_scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = field(
        default="weighted_sum",
        metadata={"help": "使用通过的测试用例的比例作为奖励。如果为 False，则使用 0/1 评分。"},
    )
    parallel_code_exec_per_proc: int = field(
        default=2,
        metadata={
            "help": "每个进程的并行 E2B 代码执行数。对于使用 8 个 GPU 进行训练的 E2B 免费爱好层，默认值 2 是合适的。"
        },
    )

    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "用作训练提示的列。"},
    )

    e2b_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "E2B 路由器的 URL。参见 scripts/e2b_router.py"},
    )

    morph_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "MorphCloud 路由器的 URL。参见 scripts/morph_router.py"},
    )

    code_provider: Optional[str] = field(
        default="e2b",
        metadata={
            "help": "代码执行的提供商。选项：'e2b'、'local'、'morph'。",
            "choices": ["e2b", "local", "morph"],
        },
    )

    ioi_provider: Optional[str] = field(
        default="piston",
        metadata={
            "help": "IOI 代码执行的提供商。选项：'piston'、'morph'。",
            "choices": ["piston", "morph"],
        },
    )

    max_completion_len: int = field(
        default=16384,
        metadata={"help": "生成结果中的最大字符数。"},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "生成结果中的最小字符数。"},
    )
