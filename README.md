# HW2 深度学习与空间智能

本仓库用于完成深度学习与空间智能 HW2。目前已实现 Task 1：在 `102 Category Flower Dataset` 上微调 ImageNet 预训练卷积神经网络，实现 102 类花卉图像分类，并保存实验曲线、指标和模型权重。

## 环境配置

建议使用 Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果使用 CUDA，请按你的显卡和 CUDA 版本安装对应的 PyTorch；如果使用 Apple Silicon，默认 PyTorch MPS 后端可用时会自动启用。

## Task 1：Flowers102 图像分类

### 已实现内容

- `ResNet-18/ResNet-34` Baseline，支持 ImageNet 预训练初始化。
- 随机初始化训练对照，用于“预训练消融实验”。
- `SE-ResNet-18/34` 注意力模型，用于与 Baseline 做 Accuracy 对比。
- 支持超参分析：学习率、batch size、优化器、epoch、scheduler 等均可通过 YAML 配置或命令行覆盖。
- 自动保存：
  - `checkpoints/task1/<experiment>/best.pt`
  - `checkpoints/task1/<experiment>/last.pt`
  - `runs/task1/<experiment>/metrics.csv`
  - `runs/task1/<experiment>/curves.png`
  - `runs/task1/<experiment>/summary.json`
- 支持 `wandb` 或 `swanlab` 记录训练/验证集 loss 和 accuracy 曲线。

### 训练命令

Baseline：

```bash
python -m src.task1.train --config configs/task1_resnet18_pretrained.yaml
```

随机初始化对照：

```bash
python -m src.task1.train --config configs/task1_resnet18_scratch.yaml
```

加入 SE 注意力：

```bash
python -m src.task1.train --config configs/task1_seresnet18_pretrained.yaml
```

超参实验示例：

```bash
python -m src.task1.train --config configs/task1_resnet18_pretrained_lr3e-4_bs64.yaml
```

也可以直接覆盖配置：

```bash
python -m src.task1.train --config configs/task1_resnet18_pretrained.yaml --override train.epochs=10 train.learning_rate=0.0003 train.batch_size=64
```

首次运行会自动下载 Flowers102 数据集到 `data/`。如果数据已手动准备好，可加 `--no-download`。

### 使用 wandb 或 swanlab

修改配置中的：

```yaml
logging:
  backend: wandb
```

或：

```yaml
logging:
  backend: swanlab
```

默认值为 `none`，不上传日志。

### 汇总实验结果

训练完多个实验后执行：

```bash
python -m src.task1.compare_experiments --runs-dir runs/task1 --out reports/task1/task1_results.csv
```

该命令会读取每个实验的 `summary.json`，生成报告表格，便于写入 PDF 实验报告。

## 实验报告与权重提交准备

Task 1 实验报告见：

- `reports/task1/report_template.md`

模型权重下载地址：

- [Google Drive](https://drive.google.com/file/d/1XR98f6fDGAXG2g9CKBrVLxt7xh-hWrS9/view?usp=drive_link)

