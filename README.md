# BT-TWD-v2 第二篇论文实验工程

本仓库当前目标是服务第二篇论文实验：在共享全局 posterior、bucket tree 与基础 TWD 风险/阈值模块之上，搭建 governance 主流程，包括后决策验证、defer 生命周期闭合、上下文更新与样本级记录。

第一篇完整工程已有备份，因此本仓库只保留最小原方法 baseline，用于实验对比。

## 依赖

```bash
pip install numpy pandas scipy scikit-learn pyyaml psutil xgboost
```

如果配置中的 `global_estimator` 使用 `xgb`，需要安装 `xgboost`。

## 目录结构

- `bttwdlib/core/`：第二篇共享核心模块，包含数据、预处理、划分、全局 posterior、指标、IO、随机种子。
- `bttwdlib/bucket/`：规则 bucket tree、样本路由、结构保护。
- `bttwdlib/twd/`：风险计算、阈值搜索、P/BND/N 决策。
- `bttwdlib/baseline/`：第一篇原方法最小 baseline，包括原 BT-TWD、BSM、weak bucket、gain、ancestor threshold substitution。
- `bttwdlib/governance/`：第二篇主方法框架。
- `scripts/run_original_baseline.py`：运行第一篇原方法 baseline。
- `scripts/run_governance_experiments.py`：运行第二篇 governance 框架。
- `configs/`：数据集、成本和默认批量配置。
- `outputs/`：实验输出。
- `archive/`：旧 notebook、绘图、旧结果、临时脚本和兼容包装目录。

## 运行第一篇最小 baseline

```bash
python scripts/run_original_baseline.py --config configs/adult_bttwd.yaml
```

该入口只用于对比实验，不作为第二篇 governance 主流程依赖。

## 运行第二篇 governance 框架

批量运行 `configs/default.yaml` 中列出的数据集：

```bash
python scripts/run_governance_experiments.py --config configs/default.yaml
```

运行单个数据集：

```bash
python scripts/run_governance_experiments.py --config configs/adult_bttwd.yaml
```

输出位于：

- `outputs/governance/full/<dataset>/sample_records.csv`
- `outputs/governance/full/<dataset>/fold_summary.csv`
- `outputs/governance/full/dataset_summary.csv`
- 消融实验分别写入 `outputs/governance/no_cp/` 和 `outputs/governance/no_progressive/`

## Notebook 运行方式

每个数据集都有独立 notebook，位于 `notebooks/`。这些 notebook 只作为单数据集实验运行和结果查看入口，内部通过 `subprocess` 调用 `scripts/run_governance_experiments.py`，不复制核心算法逻辑。

建议先运行：

- `notebooks/run_adult_governance.ipynb`

每个 notebook 包含：

- 完整 governance 实验；
- 无 CP 消融；
- 无渐进更新消融；
- 读取并展示 `dataset_summary.csv`、`fold_summary.csv`。

如需批量运行，仍推荐使用命令行：

```bash
python scripts/run_governance_experiments.py --config configs/default.yaml
```

## 当前 governance 规则

- `post_validation.validate_post_decision`：BND 不做 CP 并进入 defer；P/N 经 CP prediction set 验证。
- `defer_lifecycle.resolve_deferred_sample`：defer 路径逐层记录风险证据，非 ROOT 层 P/N 需 CP 通过才闭合，ROOT 基于聚合 P/N 风险强制闭合。
- `context_update.update_decision_context`：记录路径、风险证据、自适应权重和聚合风险。
- `records.build_sample_record`：统一样本级过程记录。

当前版本只是第二篇实验框架，不代表最终算法实现。
