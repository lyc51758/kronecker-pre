"""
Muon优化器完整实验流程
对比：StandardMuon、LoRAPreMuon、KroneckerMuon
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from muon_optimizers import StandardMuon, LoRAPreMuon, KroneckerMuon
import os
import random
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleMLP(nn.Module):
    """简单的MLP用于测试"""
    def __init__(self, hidden1=256, hidden2=128, dataset='cifar10'):
        super().__init__()
        # CIFAR-10: 32x32x3 = 3072, MNIST: 28x28 = 784
        input_size = 32*32*3 if dataset == 'cifar10' else 28*28
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 10)
        )
    
    def forward(self, x):
        return self.layers(x)


def train_one_epoch(model, train_loader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test_model(model, test_loader, device):
    """测试模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy


def measure_memory_usage(optimizer):
    """测量优化器状态的内存占用（MB）"""
    total_memory = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        total_memory += value.numel() * value.element_size()
    return total_memory / (1024 ** 2)


def initialize_optimizer_state(optimizer, model, data_loader, device):
    """初始化优化器状态"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()


def compare_muon_optimizers(num_epochs=10, hidden1=256, hidden2=128,
                           lr=1e-3, momentum=0.9, rank=128, gamma1=None, seed=42, dataset='cifar10', batch_size=64, kronecker_batch_size=None):
    """
    对比三种Muon优化器
    
    Args:
        num_epochs: 训练轮数
        hidden1: 第一个隐藏层维度
        hidden2: 第二个隐藏层维度
        lr: 学习率
        momentum: 动量参数
        rank: LoRA-Pre的秩
        gamma1: LoRA-Pre的gamma参数
        seed: 随机种子
    """
    # 设置随机种子
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"数据集: {dataset.upper()}")
    
    # 根据数据集设置输入大小和归一化参数
    if dataset == 'cifar10':
        input_size = 32*32*3
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2023, 0.1994, 0.2010)
        print(f"模型结构: 3072 -> {hidden1} -> {hidden2} -> 10")
    else:  # MNIST
        input_size = 28*28
        normalize_mean = (0.1307,)
        normalize_std = (0.3081,)
        print(f"模型结构: 784 -> {hidden1} -> {hidden2} -> 10")
    
    # 计算实际的gamma1（如果未指定，使用耦合关系）
    actual_gamma1 = gamma1 if gamma1 is not None else (1 - math.sqrt(momentum))
    print(f"超参数: lr={lr}, momentum={momentum}, rank={rank}, gamma1={actual_gamma1:.4f} (自动耦合)" if gamma1 is None else f"超参数: lr={lr}, momentum={momentum}, rank={rank}, gamma1={gamma1}")
    print(f"随机种子: {seed}")
    print(f"Batch Size: {batch_size}" + (f" (KroneckerMuon: {kronecker_batch_size})" if kronecker_batch_size is not None and kronecker_batch_size != batch_size else ""))
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    else:  # MNIST
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 为 KroneckerMuon 创建单独的 DataLoader（如果指定了不同的 batch size）
    if kronecker_batch_size is not None and kronecker_batch_size != batch_size:
        generator_kron = torch.Generator()
        generator_kron.manual_seed(seed)  # 使用相同的种子保证数据顺序一致
        train_loader_kron = DataLoader(train_dataset, batch_size=kronecker_batch_size, shuffle=True, generator=generator_kron)
    else:
        train_loader_kron = train_loader  # 使用相同的 DataLoader
    
    results = {}
    
    # 1. Kronecker Muon (先跑)
    print("\n" + "="*60)
    print("测试Kronecker Muon优化器")
    print("="*60)
    if kronecker_batch_size is not None and kronecker_batch_size != batch_size:
        print(f"KroneckerMuon 使用单独的 Batch Size: {kronecker_batch_size}")
    print("Kronecker分解维度信息:")
    model3 = SimpleMLP(hidden1=hidden1, hidden2=hidden2, dataset=dataset).to(device)
    optimizer3 = KroneckerMuon(model3.parameters(), lr=lr, momentum=momentum, gamma1=gamma1)
    
    initialize_optimizer_state(optimizer3, model3, train_loader_kron, device)
    memory3 = measure_memory_usage(optimizer3)
    print(f"优化器状态内存: {memory3:.2f} MB")
    
    train_losses3, train_accs3 = [], []
    for epoch in range(num_epochs):
        loss, acc = train_one_epoch(model3, train_loader_kron, optimizer3, device)
        train_losses3.append(loss)
        train_accs3.append(acc)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}, Train Acc={acc:.2f}%")
    
    test_acc3 = test_model(model3, test_loader, device)
    print(f"Test Accuracy: {test_acc3:.2f}%")
    
    results['KroneckerMuon'] = {
        'memory_mb': memory3,
        'train_losses': train_losses3,
        'train_accs': train_accs3,
        'test_acc': test_acc3
    }
    
    # 2. 标准Muon
    print("\n" + "="*60)
    print("测试标准Muon优化器")
    print("="*60)
    model1 = SimpleMLP(hidden1=hidden1, hidden2=hidden2, dataset=dataset).to(device)
    optimizer1 = StandardMuon(model1.parameters(), lr=lr, momentum=momentum)
    
    initialize_optimizer_state(optimizer1, model1, train_loader, device)
    memory1 = measure_memory_usage(optimizer1)
    print(f"优化器状态内存: {memory1:.2f} MB")
    
    train_losses1, train_accs1 = [], []
    for epoch in range(num_epochs):
        loss, acc = train_one_epoch(model1, train_loader, optimizer1, device)
        train_losses1.append(loss)
        train_accs1.append(acc)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}, Train Acc={acc:.2f}%")
    
    test_acc1 = test_model(model1, test_loader, device)
    print(f"Test Accuracy: {test_acc1:.2f}%")
    
    results['StandardMuon'] = {
        'memory_mb': memory1,
        'train_losses': train_losses1,
        'train_accs': train_accs1,
        'test_acc': test_acc1
    }
    
    # 3. LoRA-Pre Muon
    print("\n" + "="*60)
    print("测试LoRA-Pre Muon优化器")
    print("="*60)
    model2 = SimpleMLP(hidden1=hidden1, hidden2=hidden2, dataset=dataset).to(device)
    optimizer2 = LoRAPreMuon(model2.parameters(), lr=lr, momentum=momentum, 
                            rank=rank, gamma1=gamma1)
    
    initialize_optimizer_state(optimizer2, model2, train_loader, device)
    memory2 = measure_memory_usage(optimizer2)
    print(f"优化器状态内存: {memory2:.2f} MB")
    
    train_losses2, train_accs2 = [], []
    for epoch in range(num_epochs):
        loss, acc = train_one_epoch(model2, train_loader, optimizer2, device)
        train_losses2.append(loss)
        train_accs2.append(acc)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}, Train Acc={acc:.2f}%")
    
    test_acc2 = test_model(model2, test_loader, device)
    print(f"Test Accuracy: {test_acc2:.2f}%")
    
    results['LoRAPreMuon'] = {
        'memory_mb': memory2,
        'train_losses': train_losses2,
        'train_accs': train_accs2,
        'test_acc': test_acc2
    }
    
    # 打印对比结果
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"{'方法':<20} {'内存(MB)':<15} {'测试准确率':<15}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['memory_mb']:<15.2f} {metrics['test_acc']:<15.2f}%")
    
    return results, {
        'num_epochs': num_epochs,
        'hidden1': hidden1,
        'hidden2': hidden2,
        'dataset': dataset,
        'lr': lr,
        'momentum': momentum,
        'rank': rank,
        'gamma1': gamma1
    }


def plot_results(results, save_dir):
    """绘制实验结果"""
    # 训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    for name, metrics in results.items():
        if 'train_losses' in metrics:
            epochs = range(1, len(metrics['train_losses']) + 1)
            ax1.plot(epochs, metrics['train_losses'], label=name, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for name, metrics in results.items():
        if 'train_accs' in metrics:
            epochs = range(1, len(metrics['train_accs']) + 1)
            ax2.plot(epochs, metrics['train_accs'], label=name, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy (%)')
    ax2.set_title('Training Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {os.path.join(save_dir, 'training_curves.png')}")
    plt.close()
    
    # 内存对比
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    memories = [results[name]['memory_mb'] for name in names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(names, memories, color=colors[:len(names)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Optimizer State Memory Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mem in zip(bars, memories):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.2f} MB',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'memory_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"内存对比图已保存到: {os.path.join(save_dir, 'memory_comparison.png')}")
    plt.close()
    
    # 性能总结
    fig, ax = plt.subplots(figsize=(10, 6))
    test_accs = [results[name].get('test_acc', 0) for name in names]
    memories = [results[name]['memory_mb'] for name in names]
    
    for i, (name, acc, mem) in enumerate(zip(names, test_accs, memories)):
        ax.scatter(mem, acc, s=200, alpha=0.6, color=colors[i % len(colors)],
                  label=name, edgecolors='black', linewidth=2)
        ax.annotate(name, (mem, acc), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Memory Usage (MB)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Memory-Performance Trade-off')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
    print(f"性能总结图已保存到: {os.path.join(save_dir, 'performance_summary.png')}")
    plt.close()


def create_summary_table(results, exp_params, save_path):
    """创建文本格式的总结表"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Muon优化器对比实验结果总结\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("实验参数:\n")
        f.write(f"  训练轮数: {exp_params['num_epochs']}\n")
        f.write(f"  模型结构: 784 -> {exp_params['hidden1']} -> {exp_params['hidden2']} -> 10\n")
        f.write(f"  学习率: {exp_params['lr']}\n")
        f.write(f"  动量: {exp_params['momentum']}\n")
        f.write(f"  LoRA-Pre秩: {exp_params['rank']}\n")
        gamma1_val = exp_params.get('gamma1', '自动耦合')
        if gamma1_val != '自动耦合':
            f.write(f"  Gamma1: {gamma1_val}\n")
        else:
            momentum_val = exp_params.get('momentum', 0.9)
            f.write(f"  Gamma1: {1 - math.sqrt(momentum_val):.4f} (自动耦合: 1 - sqrt(momentum))\n")
        f.write("\n")
        
        f.write("结果对比:\n")
        f.write(f"{'方法':<20} {'内存(MB)':<15} {'测试准确率(%)':<15} {'压缩比':<15}\n")
        f.write("-" * 70 + "\n")
        
        baseline_memory = results.get('StandardMuon', {}).get('memory_mb', 0)
        
        for name, metrics in results.items():
            memory = metrics.get('memory_mb', 0)
            test_acc = metrics.get('test_acc', 0)
            
            if baseline_memory > 0 and name != 'StandardMuon':
                compression = baseline_memory / memory if memory > 0 else 0
                f.write(f"{name:<20} {memory:<15.2f} {test_acc:<15.2f} {compression:<15.2f}x\n")
            else:
                f.write(f"{name:<20} {memory:<15.2f} {test_acc:<15.2f} {'1.00x':<15}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"总结表已保存到: {save_path}")


if __name__ == "__main__":
    print("="*70)
    print("Muon优化器对比实验")
    print("="*70)
    print("对比三种优化器：")
    print("1. 标准Muon (StandardMuon)")
    print("2. LoRA-Pre Muon (LoRAPreMuon)")
    print("3. Kronecker Muon (KroneckerMuon)")
    print("="*70)
    
    # 实验参数
    num_epochs = 10
    hidden1 = 256
    hidden2 = 128
    lr = 1e-3
    momentum = 0.9
    rank = 128
    # gamma1会自动与momentum耦合（gamma1 = 1 - sqrt(momentum)）
    # 如果需要手动指定，可以取消下面的注释
    # gamma1 = 1 - math.sqrt(momentum)  # 与momentum耦合
    gamma1 = None  # None表示使用自动耦合
    seed = 42
    
    # 数据集选择
    dataset = 'cifar10'  # 可选: 'cifar10' 或 'mnist'
    
    # 运行实验
    results, exp_params = compare_muon_optimizers(
        num_epochs=num_epochs,
        hidden1=hidden1,
        hidden2=hidden2,
        lr=lr,
        momentum=momentum,
        rank=rank,
        gamma1=gamma1,
        seed=seed,
        dataset=dataset
    )
    
    # 生成结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_gamma1 = gamma1 if gamma1 is not None else (1 - math.sqrt(momentum))
    folder_name = f"muon_epoch{num_epochs}_h{hidden1}_{hidden2}_m{momentum}_r{rank}_g{actual_gamma1:.3f}_{timestamp}"
    results_dir = os.path.join("results", folder_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n结果将保存到: {results_dir}")
    
    # 生成可视化结果
    print("\n生成可视化结果...")
    plot_results(results, results_dir)
    create_summary_table(results, exp_params, os.path.join(results_dir, 'summary_table.txt'))
    
    print(f"\n所有结果已保存到: {results_dir}")

