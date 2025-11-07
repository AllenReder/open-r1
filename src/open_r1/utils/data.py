import logging

import datasets
from datasets import DatasetDict, concatenate_datasets

from ..configs import ScriptArguments


logger = logging.getLogger(__name__)


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """根据配置加载单个数据集或数据集混合体。

    参数：
        args (ScriptArguments): 包含数据集配置的脚本参数。

    返回：
        DatasetDict: 加载的数据集。
    """
    # 如果只指定了单个数据集名称，不是混合体
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"正在加载数据集: {args.dataset_name}")
        return datasets.load_dataset(args.dataset_name, args.dataset_config)
    # 如果指定了数据集混合体，需要加载多个数据集并合并
    elif args.dataset_mixture:
        logger.info(f"创建包含 {len(args.dataset_mixture.datasets)} 个数据集的混合体")
        seed = args.dataset_mixture.seed
        datasets_list = []

        # 遍历混合体中的每个数据集配置
        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"为混合体加载数据集: {dataset_config.id} (配置: {dataset_config.config})")
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )
            # 如果指定了列，只选择这些列
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            # 如果指定了权重，按权重对数据集进行采样
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"按权重={dataset_config.weight} 对数据集 '{dataset_config.id}' (配置: {dataset_config.config}) 进行了下采样，得到 {len(ds)} 个样本"
                )

            datasets_list.append(ds)

        # 如果成功加载了数据集，将它们合并
        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"创建了包含 {len(combined_dataset)} 个样本的数据集混合体")

            # 如果指定了测试集划分大小，将数据集分成训练集和测试集
            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    f"将数据集划分为训练集和测试集，测试集大小: {args.dataset_mixture.test_split_size}"
                )
                return combined_dataset
            else:
                # 如果未指定测试集大小，仅返回训练集
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("从混合体配置中未加载到任何数据集")

    else:
        # 必须提供 `dataset_name` 或 `dataset_mixture` 之一
        raise ValueError("必须提供 `dataset_name` 或 `dataset_mixture` 中的至少一个")
