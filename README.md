# Muon优化器实验

## 项目简介

本项目实现了论文Algorithm 2中的Muon优化器及其变体：
- **StandardMuon**: 标准Muon优化器（SGD with Momentum + 正交化）
- **LoRAPreMuon**: 使用低秩分解的Muon优化器（论文方法）
- **KroneckerMuon**: 使用Kronecker积分解的Muon优化器（改进方法）
- **HybridKroneckerStandardMuon**: 混合优化器（隐藏层用Kronecker，输出层用标准Muon）

## 文件结构

```
muon_experiments/
├── muon_optimizers.py                    # 三种基础优化器的实现
├── muon_experiment.py                    # 实验工具函数
├── hybrid_kronecker_standard_muon.py     # 混合优化器实现
├── run_experiment_large.py               # 实验1: 大规模模型对比
├── run_experiment_equal_memory.py        # 实验2: 等显存公平对比
├── run_experiment_rank_sweep.py          # 实验3: Rank效率曲线
├── run_experiment_hybrid.py              # 实验4: 混合优化器实验
├── requirements.txt                      # 依赖包列表
└── README.md                             # 本文件
```

## 安装依赖

```bash
cd muon_experiments
pip install -r requirements.txt
```

或者手动安装：
```bash
pip install torch torchvision numpy matplotlib
```

## 数据准备

**提示**: 需要先下载数据集。

本项目支持两种数据集：
- **CIFAR-10**: 32×32 RGB图像，10个类别，输入维度3072
- **MNIST**: 28×28 灰度图像，10个类别，输入维度784

### 方法1: 自动下载（推荐）

运行实验脚本时，代码会自动下载所需数据集到 `data/` 文件夹。首次运行时会自动下载，后续运行会直接使用已下载的数据。

### 方法2: 手动下载

你也可以提前下载数据集，创建一个简单的下载脚本：

```python
# download_data.py
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 下载 CIFAR-10
print("正在下载 CIFAR-10 数据集...")
transform = transforms.ToTensor()
cifar10_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
print("CIFAR-10 下载完成！")

# 下载 MNIST
print("正在下载 MNIST 数据集...")
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
print("MNIST 下载完成！")
```

运行下载脚本：
```bash
python download_data.py
```

### 数据目录结构

下载完成后，`data/` 文件夹结构如下：
```
data/
├── cifar-10-batches-py/          # CIFAR-10 数据集
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   └── test_batch
└── MNIST/                         # MNIST 数据集
    └── raw/
        ├── train-images-idx3-ubyte
        ├── train-labels-idx1-ubyte
        ├── t10k-images-idx3-ubyte
        └── t10k-labels-idx1-ubyte
```

## 快速开始

### 步骤1: 下载数据

如果 `data/` 文件夹为空，首次运行实验时会自动下载数据。你也可以提前下载（见上面的"数据准备"部分）。

### 步骤2: 运行实验

1. **运行大规模对比实验**（推荐首次运行）:
```bash
python run_large.py
```

或者使用其他实验脚本：
```bash
python run_memory.py      # 等显存公平对比
python run_rank.py         # Rank效率曲线
python run_experiment_hybrid.py  # 混合优化器实验
```

### 查看结果

运行完成后，结果会保存在 `results/` 目录下，包含：
- 可视化图表（PNG格式）
- 文本摘要表格（TXT格式）

终端会显示最终的内存对比和准确率对比结果。

## 如何运行实验

**注意**: 首次运行前，请确保已下载数据集（见"数据准备"部分）。如果 `data/` 文件夹为空，代码会自动下载所需数据集。

### 实验1: 大规模模型对比 (`run_large.py`)

**目的**: 对比三种优化器在大规模模型上的内存占用和性能表现

**运行方式**:
```bash
python run_large.py
```

**实验参数** (可在脚本中修改):
- 模型结构: `3072 -> 4096 -> 2048 -> 10` (CIFAR-10)
- Epochs: 10
- Learning Rate: 0.001
- Momentum: 0.9
- Rank (LoRA-Pre): 128
- Batch Size: 64 (StandardMuon/LoRAPreMuon), 可单独设置 KroneckerMuon 的 batch size

**输出结果**:
- 终端显示: 每个优化器的训练进度、内存占用、测试准确率
- 保存文件: `results/large_epoch10_h4096_2048_.../` 目录下
  - `comparison_results.png`: 训练曲线和性能对比图
  - `summary_table.txt`: 文本格式的结果摘要

---

### 实验2: 等显存公平对比 (`run_memory.py`)

**目的**: 调整LoRA-Pre的rank，使其显存与KroneckerMuon相同，进行公平对比

**运行方式**:
```bash
python run_memory.py
```

**实验参数**:
- 模型结构: `3072 -> 4096 -> 2048 -> 10` (CIFAR-10)
- 自动寻找使LoRA-Pre显存匹配KroneckerMuon的rank值

**输出结果**:
- 终端显示: 显存匹配过程、最佳rank值、三个优化器的对比结果
- 保存文件: `results/equal_memory_epoch10_.../` 目录下

---

### 实验3: Rank效率曲线 (`run_rank.py`)

**目的**: 对比KroneckerMuon与不同rank下LoRA-Pre的性能，绘制效率曲线

**运行方式**:
```bash
python run_rank.py
```

**实验参数**:
- Rank范围: 默认测试多个rank值 (4, 8, 16, 32, 64, 128等)
- 模型结构: `3072 -> 4096 -> 2048 -> 10` (CIFAR-10)

**输出结果**:
- 终端显示: 每个rank的训练进度和结果
- 保存文件: `results/rank_sweep_.../` 目录下
  - `rank_efficiency_curve.png`: Rank-性能效率曲线图
  - `summary_table.txt`: 详细结果表格

---

### 实验4: 混合优化器实验 (`run_experiment_hybrid.py`)

**目的**: 测试混合优化器（隐藏层用Kronecker分解，输出层用标准Muon）的性能

**运行方式**:
```bash
python run_experiment_hybrid.py
```

**实验参数** (可在脚本中修改):
- 模型结构: `3072 -> 4096 -> 2048 -> 10` (CIFAR-10)
- Epochs: 10
- Learning Rate: 0.001
- Momentum: 0.9
- Batch Size: 64

**实验设计**:
- **HybridKroneckerStandardMuon**: 前两层（隐藏层）使用Kronecker分解，最后一层（输出层）使用标准Muon
- **StandardMuon**: 所有层使用标准Muon（baseline）
- **LoRAPreMuon**: 所有层使用LoRA-Pre分解（baseline）

**输出结果**:
- 终端显示: 
  - 每个优化器的训练进度
  - Kronecker分解的维度信息（隐藏层）
  - 显存占用对比
- 保存文件: `results/hybrid_epoch10_h4096_2048_.../` 目录下
  - `comparison_results.png`: 4个子图（训练损失、训练准确率、测试准确率、内存使用）
  - `summary_table.txt`: 结果摘要表格

---

## 修改实验参数

所有实验脚本都在 `__main__` 部分定义了实验参数，可以直接修改：

```python
# 通用参数
num_epochs = 10        # 训练轮数
hidden1 = 4096         # 第一个隐藏层维度
hidden2 = 2048         # 第二个隐藏层维度
lr = 1e-3              # 学习率
momentum = 0.9         # 动量参数
rank = 128             # LoRA-Pre的秩
gamma1 = None          # LoRA-Pre的gamma参数 (None表示自动耦合)
seed = 42              # 随机种子
dataset = 'cifar10'    # 数据集: 'cifar10' 或 'mnist'
batch_size = 64        # Batch size
```

## 算法说明

### StandardMuon

标准Muon优化器，算法：
```
m_t = µ·m_{t-1} + g_t
m_t = Newton-Schulz正交化(m_t)
θ_t = θ_{t-1} - γ·m_t
```

### LoRAPreMuon

使用低秩分解的Muon优化器（论文Algorithm 2）：
```
m_t = µ·m_B,t-1·m_A,t-1 + g_t
m_B,t = γ1·m_B,t-1 + (1-γ1)/(1-µ) · g_t @ m_A @ (m_A^T @ m_A)^(-1)
m_A,t = γ1·m_A,t-1 + (1-γ1)/(1-µ) · (m_B^T @ m_B)^(-1) @ m_B^T @ g_t
m_t = Newton-Schulz正交化(m_t)
θ_t = θ_{t-1} - γ·m_t
```

### KroneckerMuon

使用Kronecker积分解的Muon优化器：
```
m_t = µ·(m_f1 ⊗ m_f2) + g_t
交替更新 m_f1 和 m_f2
m_t = Newton-Schulz正交化(m_t)
θ_t = θ_{t-1} - γ·m_t
```

### HybridKroneckerStandardMuon

混合优化器：
- **隐藏层**: 使用Kronecker分解（与KroneckerMuon相同）
- **输出层**: 使用标准Muon（与StandardMuon相同）

## 实验结果

所有实验结果会保存在 `results/` 文件夹下，每个实验会创建一个带时间戳的子文件夹。

### 输出文件说明

#### 1. 可视化图表 (`comparison_results.png` 或类似名称)
- **训练损失曲线**: 显示各优化器在每个epoch的训练损失
- **训练准确率曲线**: 显示各优化器在每个epoch的训练准确率
- **测试准确率对比**: 柱状图显示最终测试准确率
- **内存使用对比**: 柱状图显示优化器状态内存占用

#### 2. 文本摘要 (`summary_table.txt`)
包含：
- 实验参数（epochs, hidden layers, lr, momentum等）
- 结果对比表格（优化器名称、内存占用、测试准确率）
- 显存节省倍数（相对于StandardMuon）

#### 3. 终端输出
运行时会实时显示：
- 实验配置信息
- 每个优化器的训练进度（每个epoch的loss和accuracy）
- 优化器状态内存占用
- 最终测试准确率
- 显存对比结果

## 实验参数说明

### 关键参数

1. **momentum (µ)**: 动量参数，范围 [0, 1)
   - 默认值：0.9
   - 控制历史动量的保留比例
   - 影响gamma1的自动计算：`gamma1 = 1 - sqrt(momentum)`

2. **rank**: LoRA-Pre的秩
   - 默认值：128
   - 影响内存压缩比和性能
   - 较小的rank节省更多内存，但可能影响性能

3. **gamma1 (γ1)**: LoRA-Pre的更新率
   - 默认值：None（自动耦合，计算为 `1 - sqrt(momentum)`）
   - 手动设置：范围 [0, 1)
   - 控制因子更新的平滑程度

4. **lr (γ)**: 学习率
   - 默认值：1e-3
   - 控制参数更新的步长

5. **batch_size**: 批次大小
   - 默认值：64
   - 在 `run_experiment_large.py` 中，可以为KroneckerMuon单独设置 `kronecker_batch_size`

6. **dataset**: 数据集选择
   - 可选：`'cifar10'` 或 `'mnist'`
   - CIFAR-10: 32×32 RGB图像，10类，输入维度3072
   - MNIST: 28×28 灰度图像，10类，输入维度784

### Kronecker分解相关

- **Kronecker分解维度**: 自动计算，将参数矩阵 `(m, n)` 分解为 `(m1, n1) ⊗ (m2, n2)`
- **等效rank**: 基于参数数量计算的等效低秩分解rank
- **压缩比**: 原始参数数量 / 分解后参数数量
- **Kronecker熵**: 用于评估Kronecker分解的适用性（值越大越好，接近1表示分解质量高）

## 预期结果

### 内存占用
- **StandardMuon**: 最高（存储完整动量矩阵）
- **LoRAPreMuon**: 中等（取决于rank值）
- **KroneckerMuon**: 最低（Kronecker分解压缩比高）
- **HybridKroneckerStandardMuon**: 介于KroneckerMuon和StandardMuon之间

### 性能表现
- 在相同条件下，三种方法应该达到相似的最终测试准确率
- 收敛速度可能略有不同
- KroneckerMuon在极低内存下仍能保持较好性能

## 常见问题

### Q: 数据下载失败怎么办？
A: 如果自动下载失败，可以：
1. 检查网络连接
2. 手动运行 `download_data.py` 脚本（见"数据准备"部分）
3. 或者手动从官网下载数据集并解压到 `data/` 文件夹

### Q: 如何修改模型大小？
A: 在实验脚本的 `__main__` 部分修改 `hidden1` 和 `hidden2` 参数。

### Q: 如何调整超参数？
A: 在对应实验脚本的 `__main__` 部分修改相应参数（lr, momentum, rank等）。

### Q: 结果保存在哪里？
A: 结果保存在 `results/` 文件夹下，每个实验有独立的带时间戳的子文件夹。

### Q: 如何确保结果可重复？
A: 代码中已设置随机种子（seed=42），相同参数下结果应该一致。

### Q: 如何切换数据集？
A: 在实验脚本中修改 `dataset` 参数为 `'cifar10'` 或 `'mnist'`。切换数据集时，确保对应的数据集已下载到 `data/` 文件夹。

### Q: KroneckerMuon的batch size可以单独设置吗？
A: 在 `run_experiment_large.py` 中可以设置 `kronecker_batch_size` 参数，其他优化器使用 `batch_size`。

### Q: 混合优化器是如何工作的？
A: `HybridKroneckerStandardMuon` 自动识别隐藏层和输出层，对隐藏层使用Kronecker分解，对输出层使用标准Muon。

### Q: 训练需要多长时间？
A: 取决于模型大小和epoch数。对于 `4096->2048->10` 的模型，10个epoch大约需要几分钟到十几分钟（取决于硬件）。

### Q: 如何查看Kronecker分解的详细信息？
A: 运行实验时，终端会显示每个层的Kronecker分解维度、压缩比和等效rank信息。

## 参考

- 论文: "Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation"
- Algorithm 2: Comparison of Muon and Muon with LoRA-Pre

## 作者

基于论文Algorithm 2实现，包含标准Muon、LoRA-Pre Muon和Kronecker Muon三种变体。

