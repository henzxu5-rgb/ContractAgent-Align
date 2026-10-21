# 复现与公开材料

## 1. 不使用模型，直接核验已有结果

在仓库根目录、Python 3.11 环境执行：

```bash
python -m pip install -r contract_agent/requirements-local.txt
python -m unittest discover -s contract_agent/tests -q
python tools/verify_public_results.py
python -m contract_agent.server --port 8766
```

页面为只读回放，不产生 API 费用。仓库保留全部 1,700 条独立测试执行、236 条教师任务记录、200 条训练侧策略任务记录、400 对候选，以及正式训练 JSONL、配置、验收与评分报告。数据和测试参考答案公开；这不等于行动模型运行时能读取参考答案，隔离由环境和提示白名单实现。

`verify_public_results.py` 从保存的完整测试执行重算分数与任务覆盖，不调用模型；不重复耗时的 bootstrap。完整区间已保留，算法见 `contract_agent/analysis.py`。

## 2. 数据重建

数据已经包含在仓库中。需要从源头重建时，保留当前快照，在单独的 checkout 中执行：

```bash
python -m contract_agent.cli prepare
```

该步骤联网下载原始公开数据，校验固定 SHA256 并生成分离的原文、标注、任务索引及 manifest。详细来源和变更见 DATA_CARD.md。

## 3. 重新推理与训练

这部分需要另备 CUDA GPU 和基础模型，不属于只读体验。验证过的环境为 Torch 2.7.1+cu128、Transformers 4.55.4、TRL 0.21.0、PEFT 0.17.1、bitsandbytes 0.47.0。安装与实际 CUDA/驱动兼容的 Torch，再安装其余固定训练依赖；不要把普通 CPU 环境当作训练环境。

基础模型 revision 固定于 `contract_agent/settings.py`。如果提供本地模型目录，需使用项目下载器生成并验证下载 receipt：

```bash
python -m contract_agent.download_model --help
python -m contract_agent.training --help
python -m contract_agent.cli run --help
```

正式训练命令示例（自行指定可用 GPU，执行会消耗计算资源）：

```bash
python -m contract_agent.training sft --data contract_agent/artifacts/datasets/align-v1-sft.jsonl --run-id reproduction-sft
python -m contract_agent.training dpo --data contract_agent/artifacts/datasets/align-v1-rtx6000d-v2-raw.jsonl --sft-adapter contract_agent/artifacts/models/reproduction-sft --run-id reproduction-dpo-raw
python -m contract_agent.training dpo --data contract_agent/artifacts/datasets/align-v1-rtx6000d-v2-verified.jsonl --sft-adapter contract_agent/artifacts/models/reproduction-sft --run-id reproduction-dpo-verified
```

正式训练会核验技术验收状态、数据范围与掩码；缺少适配器直接失败。仓库中的 gate 是原实验的历史记录，不能代替新机器的 SFT/DPO 技术验收。不要用开发侧微型验收模型替代正式模型。两组 DPO 应从同一新 SFT checkpoint 开始，并复核参考模型不更新。

本轮三份正式适配器大权重不在 Git 中，其 SHA256 见 `contract_agent/artifacts/reports/model_checksums.json`。它们保存在本地完整交付包中；公开仓库能复算已有测试结果，但重新生成本轮模型的预测需要取得这些适配器，或明确作为新的重训实验。

## 4. 重新生产教师/Judge 数据

需要模型 API。凭据仅放本地环境变量，格式参考 `.env.example`；不要提交真实 `.env`。API 生产、训练和推理需要明确的新 run ID、预算与技术验收，不在浏览网页时自动启动。项目中的云端编排脚本为特定实验恢复工具，不是体验入口。

教师服务名称和 API 响应按当时返回记录保存。服务端别名及中断后的随机流会变化，因此不能保证重新采样逐字相同。公开材料保留实际候选、续跑、原始与交换后 Judge 判定，供直接核验。

## 5. 公开整理与谱系

发布采用显式文件白名单，仅包括当前 ContractAgent-Align 源码、文档及实验材料。凭据、账户余额响应、个人纪要、旧项目、模型缓存和优化器大文件均不进入发布目录。

JSON 中的本机绝对路径改为仓库相对路径；云端登录地址移除。原始私有记录保留，公开版本另生成 `PUBLICATION_MANIFEST.json`，分别记录原始与公开文件哈希。路径整理后的文件哈希可能与原始训练 receipt 中的输入哈希不同；候选动作、评分、测试任务与参考标注不因此重算或改写。
